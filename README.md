# RustFox

**A tiered inference core for machines that shouldn't be able to run it.**

RustFox treats storage, RAM, and whatever GPU you happen to have as **one memory
hierarchy** rather than three separate places, and it spreads a model across all
of them so that no single component drowns. Pure C11, no dependencies beyond
libc. Linux and Docker first, VPS second, native Windows third.

The target is not a workstation. The target is a 4 GB Celeron laptop, a €4/month
VPS, an old office box someone was about to throw away.

> **Status: early, but the hard part works.** Hardware probing, cgroup
> awareness, the residency governor, the GGUF parser, the quantised kernels,
> and both weight backends — resident and streaming — are implemented and
> exercised in CI on x86_64, arm64, musl, macOS and Windows. CI asserts that
> streaming returns byte-for-byte what resident returns while staying inside a
> budget tight enough to force eviction on every pass.
>
> **It generates text.** `fox run model.gguf -p "..."` works, resident or
> streamed. What is not done yet is the performance work that makes streaming
> cheap rather than merely correct — async prefetch, the layer-contiguous
> repack, and vectorised TQ1_0. See [Roadmap](#roadmap) for exactly what exists.

---

## Why this exists

[Colibrì](https://github.com/JustVugg/colibri) showed that a tiny C engine can
run enormous MoE models by streaming experts off disk. That works because
**MoE is sparse**: a top-8-of-256 router touches roughly 3% of the weights per
token, so the disk only has to move 3% of the model per token.

**Dense models get no such discount.** Every weight is read on every token. If
your model does not fit in RAM, the disk has to move the entire non-resident
remainder for each token you generate. That is the whole problem, and no amount
of clever scheduling makes it go away.

What *does* make it go away is making the model smaller. This is why 1-bit is
not merely one option among many here — it is the enabling condition:

| Quantisation | 27B dense | Fits in 4 GB RAM? |
|---|---|---|
| fp16 | ~54 GB | no |
| Q8_0 | ~28 GB | no |
| Q4_K | ~15 GB | no |
| **TQ2_0 (~2.06 bpw)** | **~7.0 GB** | not quite |
| **TQ1_0 (~1.69 bpw)** | **~5.7 GB** | not quite — but streamable |

At 1.69 bpw a 27B model becomes something a 4 GB box can *stream*, and a 4B
model at 0.85 GB becomes something it holds entirely in RAM and runs fast.
That second case is the sweet spot, and RustFox is built to find it for you and
tell you about it before you spend an hour downloading anything.

---

## The honest arithmetic

RustFox will not pretend a Celeron is an H100. Here is the model it uses, and
it is the same model the planner prints.

For a dense model, per generated token:

```
streamed_bytes = model_bytes - resident_bytes
disk_time      = streamed_bytes / measured_seq_read_bandwidth
compute_time   = model_flops / measured_cpu_throughput
token_time     ≈ max(disk_time, compute_time)     [reads overlap compute]
```

Worked example — 27B at TQ1_0 (5.7 GB) on a SATA SSD at 500 MB/s with 2.5 GB
resident:

```
streamed  = 5.7 - 2.5 = 3.2 GB
disk_time = 3.2 GB / 500 MB/s = 6.4 s
                    → about 0.16 tokens/second
```

That is slow, and RustFox will say so instead of letting you find out after the
download. Three things change that number, in order of leverage:

1. **Make it resident.** Every GB you move from disk to RAM is a GB you stop
   re-reading on every single token. This is what the governor exists to
   maximise safely.
2. **Batch or speculate.** Streaming cost is paid *per layer pass*, not per
   token. Verifying 8 speculative tokens in one pass costs one disk pass, not
   eight — so acceptance-weighted speculative decoding is worth far more on a
   streaming engine than on a resident one.
3. **Use a smaller model.** A 4B at TQ1_0 is fully resident on the same box and
   runs perhaps a hundred times faster. Often this is the right answer, and an
   engine that will not tell you so is not being your friend.

For a **MoE** model the first line changes to `streamed_bytes = active_experts ×
expert_bytes`, which is why MoE streams so much better, and why the placement
policy branches on architecture rather than pretending they are the same.

---

## What works today

### `fox probe` — what this machine actually is

Not what `/proc/meminfo` says on the host. What *this process* is allowed to
use, which inside a container is a completely different number.

```
$ fox probe --bench
rustfox 0.1.0 — system probe

  host
    os               Linux 6.8.0-45-generic
    container        yes  (cgroup v2)

  cpu
    model            Intel(R) Celeron(R) N4100 CPU @ 1.10GHz
    vendor / uarch   GenuineIntel / goldmont-plus
    cores            4 physical, 4 logical
    caches           L1d 24.00 KiB  L2 4.00 MiB  L3 0 B
    features         sse2 ssse3 sse4.1 sse4.2 popcnt

  memory
    usable total     512.00 MiB  (host has 3.81 GiB)
    available now    498.00 MiB
    cgroup limit     512.00 MiB  (using 14.00 MiB)

  storage  (/var/cache/rustfox)
    device / fs      sda / ext4
    kind             solid state
    O_DIRECT         yes
    seq read         486 MiB/s
    4K random read   18412 IOPS, 51 us median
```

`--json` gives the same thing machine-readably. `--bench` measures storage for
real — through `O_DIRECT` where the filesystem allows it, falling back to
`POSIX_FADV_DONTNEED` cache eviction, so the number is the SSD rather than the
page cache.

Note the `goldmont-plus` line. That family tops out at SSE4.2 with **no AVX at
all**, which means the fast path for ternary weights there is a `pshufb` lookup
table, not a vector multiply. The planner needs to know this, so the prober
reports it.

### `fox watch` — the governor, live

```
$ fox watch --seconds 20
residency governor: floor 32.00 MiB, ceiling 320.00 MiB, tick 250 ms
time     act       budget     mem%     io%  majflt/s  why
   0.00s hold  176.00 MiB     0.00    0.00       0.0  quiet, waiting out hysteresis
   0.50s grow  182.40 MiB     0.00    0.00       0.0  quiet: growing
   1.75s grow  208.00 MiB     0.00   22.31       0.0  io stall high and memory quiet: growing fast
   4.25s CUT   166.40 MiB     3.84    9.10      12.0  memory stall above target
   4.50s hold  166.40 MiB     2.10    4.02       0.0  cooling down after a cut
```

---

## The governor

This is the piece that keeps the machine alive, and it is the part of RustFox
worth stealing.

**It runs AIMD over the residency budget** — additive increase, multiplicative
decrease — exactly the way TCP runs it over a congestion window. Creep upward
while the system is quiet; cut hard and immediately at the first sign of
distress. The engine gets slower under pressure. The machine does not fall over.

The signal is **PSI** (`/proc/pressure/*`, or the cgroup's own
`memory.pressure` when we are containerised), not free memory. PSI answers the
question `MemFree` cannot: *how much wall time did tasks spend blocked waiting
for this resource?* A box can look like it has RAM to spare while every
allocation stalls behind reclaim, and it can look nearly full while running
perfectly happily on clean page cache. PSI knows the difference.

Cut signals:

- memory PSI `some avg10` above target, or `full avg10` above zero at all
- **major fault rate** above target — every major fault is a page we believed
  was resident and was not, paid for with a synchronous trip to the SSD. This
  is distinct from our own deliberate streaming reads, which are explicit
  `pread`s into a managed buffer and never fault.
- swap-in rate above zero-ish
- `memory.events` `high` or `max` incremented since the last tick — the kernel
  already had to intervene, so we are past the line, not approaching it

**I/O pressure is deliberately not a cut signal.** Streaming weights off storage
is the design; a busy disk is the engine working, not the engine misbehaving.
And the cure for I/O stall is *more* resident memory, not less. So high I/O
stall with quiet memory means growing is urgent, and it multiplies the additive
step rather than triggering a decrease. Getting this backwards would make the
engine throttle itself precisely when it should be claiming more RAM.

---

## Building

Nothing is built on your machine unless you want it to be — CI does it.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/fox probe --bench
```

Or just `make test`.

### Docker

```bash
docker build -f docker/Dockerfile -t rustfox .
docker run --rm --memory=512m rustfox probe --json
```

The image is a fully static musl binary on Alpine, running as a non-root user.
CI asserts that a probe inside `--memory=512m` reports 512 MiB rather than the
runner's host RAM — because a memory governor that reads the wrong number is
worse than no governor at all.

### Windows

Native builds work and are covered in CI, but WSL2 or Docker Desktop is the
recommended path: cgroup limits and PSI telemetry only exist on Linux, and
without them the governor falls back to a much blunter available-memory
heuristic.

---

## Roadmap

| | Component | State |
|---|---|---|
| ✅ | Platform layer — POSIX / Windows, `O_DIRECT`, cache eviction | done |
| ✅ | CPU probe — CPUID with XCR0 validation, uarch table, arm64 HWCAP | done |
| ✅ | Memory probe — cgroup v1/v2, container detection, host vs effective | done |
| ✅ | Storage probe — device, rotational, measured bandwidth and latency | done |
| ✅ | PSI / pressure telemetry, cgroup-local where applicable | done |
| ✅ | AIMD residency governor | done |
| ✅ | GGUF parser — metadata, arrays, tensor table, all block layouts | done |
| ✅ | Placement planner with tok/s prediction | done |
| ✅ | Ternary kernels TQ1_0 / TQ2_0, SSSE3 path for TQ2_0 | done |
| ✅ | F32, F16, Q4_0/1, Q5_0/1, Q8_0, Q4_K, Q6_K kernels | done |
| ✅ | Threadpool and multi-threaded GEMV | done |
| ✅ | Weights lease API — resident and streaming behind one contract | done |
| ✅ | Streaming backend — LRU slots, hard budget, verified vs resident | done |
| ✅ | Tokenizer | done |
| ✅ | Transformer forward pass, GQA attention, KV cache | done |
| ✅ | Sampler — greedy, top-k, top-p, repeat penalty | done |
| ✅ | `fox run` — text in, text out | done |
| 🚧 | Async prefetch — overlap layer N+1 with the compute of layer N | next |
| 🚧 | Layer-contiguous repack (`.foxpack`) for sequential streaming | planned |
| 🚧 | Vectorised TQ1_0 | planned |
| 🚧 | `io_uring` fast path on Linux | planned |
| 🚧 | Speculative decoding — the multiplier for streamed dense models | planned |
| 🚧 | Vulkan backend for integrated GPUs | later |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design reasoning.

---

## Credits

The tiering idea and the "one hierarchy, not three tiers" framing come from
[Colibrì](https://github.com/JustVugg/colibri) (Apache-2.0), which got there
first and proved it works. The transformer-level structure follows
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and its GGUF format.
RustFox is not a fork of either; it is a different bet — dense-first and
1-bit-first, for machines much smaller than either was aiming at.

Apache-2.0. See [LICENSE](LICENSE).
