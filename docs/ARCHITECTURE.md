# Architecture

The source deliberately carries no comments. This file is where the reasoning
lives — specifically the decisions that are not obvious from reading the code,
and the ones where the obvious thing is wrong.

---

## 1. Layering

```
   cli.c            fox probe / fox watch
     │
   governor.c       AIMD control over the residency budget
   pressure.c       PSI, major faults, cgroup events
   sysprobe.c       assembles the machine picture, prints it, serialises it
     │
   cpuid.c          ISA feature detection (arch-specific, OS-portable)
   platform.c       OS-independent: small files, cache dir, storage benchmark
   platform_posix.c │ one of these two compiles to a stub
   platform_win32.c │ depending on the target
```

Everything platform-conditional lives behind `src/platform.h`. Nothing above
that line contains an `#ifdef` for an operating system. `pressure.c` is a good
example: it reads only through `fox_read_small_file`, so it compiles and runs
everywhere, and simply reports "unsupported" where the files do not exist.

Both `platform_posix.c` and `platform_win32.c` are always in the source list;
each wraps itself in a single `#if` and produces an almost-empty translation
unit on the wrong target. This is deliberate — it means adding a platform
function fails to link loudly on every platform that forgot it, rather than
silently on one.

---

## 2. Why PSI and not free memory

`MemFree` is close to meaningless as a control signal. Linux will happily drive
it near zero with page cache, which is *good* — that cache is our streamed
weights and it is reclaimable on demand. Conversely a box can show gigabytes
"free" while every allocation stalls behind direct reclaim.

PSI measures the thing we actually care about: **wall time tasks spent blocked
waiting for a resource.** `some` means at least one task stalled; `full` means
every runnable task stalled and the machine got nothing done at all.

Two corollaries that shaped the code:

- **Inside a container, `/proc/pressure/*` is the host's.** It reports the
  noisy neighbour, not us. `read_psi()` therefore tries
  `/sys/fs/cgroup/<resource>.pressure` first and only falls back to `/proc`.
  Getting this backwards on a VPS would have the governor throttling itself
  because of someone else's workload.

- **Major faults deserve their own signal.** `pgmajfault` counts pages we
  believed were resident and were not. Each one is a synchronous trip to the
  SSD in the middle of a matrix multiply. This is categorically different from
  our own streaming reads, which are explicit `pread`s into a buffer we own and
  never fault. A major fault storm means our accounting is wrong, so it is a
  hard cut signal.

---

## 3. Why AIMD

The residency budget is a congestion window. We want to use as much memory as
possible without pushing the system into reclaim — which is precisely TCP's
problem of using as much bandwidth as possible without pushing the link into
loss.

Additive increase, multiplicative decrease has the property we need: it
approaches the limit slowly enough to notice the edge, and retreats fast enough
that overshoot is brief. Symmetric (additive/additive) control oscillates
across the cliff; multiplicative increase overshoots hard and thrashes.

`FOX_GOV_COOLDOWN_TICKS` exists because PSI averages over 10 seconds. Right
after a cut, `avg10` still reflects the pre-cut world, so growing again
immediately would be reacting to stale data. The cooldown waits for the signal
to catch up.

### I/O pressure is not a cut signal

This is the one that is easy to get backwards, and getting it backwards would
be actively harmful.

Streaming weights off storage **is the design**. A busy disk means the engine is
working. More importantly, the cure for I/O stall is *more* resident memory,
not less — every byte we hold is a byte we stop re-reading every token.

So high I/O stall with quiet memory is not distress, it is an argument for
claiming more RAM, and it multiplies the additive step by 4. An engine that cut
its budget on I/O pressure would starve itself exactly when it needed to grow.

---

## 4. Why the CPU probe checks XCR0

A CPUID feature bit says the silicon can execute an instruction. It does not
say the operating system has enabled saving the wider register state across
context switches. On a kernel booted with `noxsave`, or in some hypervisors,
`CPUID.1:ECX.AVX` is set and executing an AVX instruction faults.

`probe_x86()` therefore requires `OSXSAVE`, reads `XCR0`, and only reports AVX
when bits 1–2 (XMM, YMM) are set, and AVX-512 when bits 5–7 (opmask, ZMM_Hi256,
Hi16_ZMM) are also set. This is the same check llama.cpp and every other
serious runtime performs, and it is not optional.

### Why there is a microarchitecture table at all

The immediate reason is Goldmont Plus. The Celeron N4100 and its relatives are
an extremely common floor-of-the-market target, and that family has **SSE4.2 and
no AVX whatsoever**. On such a chip the fast path for ternary weights is a
`pshufb` lookup table over packed 2-bit codes — a table lookup and an add, no
multiply anywhere — and that is a *first-class* kernel there rather than a
fallback for something better.

Naming the microarchitecture lets the planner reason about this out loud rather
than inferring it from a feature bitmask.

---

## 5. Measuring storage honestly

A storage benchmark on a machine with spare RAM measures `memcpy` unless you
take steps. The probe takes them, in this order:

1. `O_DIRECT` (Linux) or `FILE_FLAG_NO_BUFFERING` (Windows) or `F_NOCACHE`
   (macOS). When granted, the page cache is out of the path entirely.
2. If the filesystem refuses — tmpfs, overlayfs and several network filesystems
   do, and that is normal rather than an error — fall back to
   `POSIX_FADV_DONTNEED` after an `fsync` to evict our own pages.
3. If neither is available, warn loudly and mark the result optimistic.

`fox_storage_info.measured` and `.supports_odirect` are reported so nothing
downstream has to guess how much to trust the number. When the benchmark is
skipped the timing fields stay at exactly zero — the planner never invents a
bandwidth it did not measure. This is asserted in the test suite.

The random-read median is taken rather than the mean because SSD latency
distributions have a long tail that a mean smears into uselessness.

---

## 6. Container awareness

Effective memory is `min(host RAM, cgroup limit)`, and effective availability is
further clamped to `cgroup limit − cgroup current`. Two layouts have to work:

- **Container-rooted** — the cgroup namespace makes `/sys/fs/cgroup/memory.max`
  ours. This is the Docker and Kubernetes case.
- **Host layout** — `/proc/self/cgroup` must be followed to the right
  subdirectory under `/sys/fs/cgroup`.

cgroup v1 spells "unlimited" as a huge page-aligned integer rather than the
literal `max`, so `linux_cgroup_mem()` treats anything above 2^62 as unlimited.
A governor that believed a limit of 9223372036854771712 bytes would never
throttle at all.

`memory.high` matters more than `memory.max` for our purposes: it is where the
kernel begins throttling rather than where it starts killing, so the default
ceiling is placed 10% below it when it is set.

CPU is handled the same way — `sched_getaffinity` rather than
`_SC_NPROCESSORS_ONLN`, because `--cpuset-cpus` makes the latter a lie, plus
`cpu.max` for the quota.

---

## 7. Placement policy (design, not yet implemented)

Streaming works differently for the two architectures, so placement branches on
them rather than pretending they are the same.

**Dense.** Every weight is touched every token, so access-frequency ranking is
degenerate — everything is equally hot. What differs is how *well* each tensor
streams:

- **Pin** norms, RoPE tables, attention Q/K/V/O, and the output head. These are
  small relative to their per-token cost and scattered enough that streaming
  them wastes I/O on seeks.
- **Stream** the FFN `gate`/`up`/`down` projections. They are typically 65–70%
  of parameters, individually large, and contiguous — ideal for one big
  sequential read prefetched a layer ahead.
- **`mmap`** the token embedding and let the page cache handle it. Access is
  sparse: only the rows for tokens actually seen.

**MoE.** Pin the router, shared experts, and attention; stream routed experts
under a heat-weighted LRU. This is the shape Colibrì established and there is
no reason to deviate from it.

The prefetch depth follows from the measurement, not from a constant: read a
layer ahead while computing the current one, and size the ring so that
`layer_bytes / seq_read_bandwidth ≈ layer_compute_time`. Deeper costs memory
for no gain; shallower leaves the disk idle.

---

## 8. Testing strategy

Two things are hard to test about this system and both are addressed directly.

**Hardware-dependent probing** cannot assert absolute values — CI runners differ.
So the tests assert *invariants* instead: physical cores never exceed logical
cores, effective total never exceeds host total, page size is a power of two,
unmeasured bandwidth is exactly zero rather than a guess.

**Container awareness** is the one thing that would silently break in the field
while passing every local test, so CI runs the real image under
`--memory=512m` and asserts the probe reports 512 MiB — not the runner's host
RAM. A memory governor that reads the wrong number is worse than no governor.

The governor's own tests drive it through synthetic `fox_sysinfo` values, since
`fox_governor_config_default` is pure and takes the probe as an argument
precisely so it can be tested without one.
