# Ternary Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide correct TQ1_0/TQ2_0 dot products with a scalar fallback and an SSSE3 LUT fast path for weak x86 hardware.

**Architecture:** The kernel API consumes canonical 256-element blocks and int8 activations, returning a scaled float dot product. Scalar decoding is the reference path; the SSSE3 implementation accelerates TQ2_0 packed 2-bit extraction with `pshufb` and uses runtime CPU feature detection, while TQ1_0 initially shares a verified scalar decoder until its packed trit path has matching differential tests.

**Tech Stack:** C11, x86 SSSE3 intrinsics guarded by compile-time architecture checks, existing CPU feature bitmask, CTest.

## Global Constraints

- Match canonical TQ1_0 (256 values, 54 bytes) and TQ2_0 (256 values, 66 bytes) layouts.
- Never execute SSSE3 instructions unless runtime feature detection reports `FOX_CPU_SSSE3`.
- Keep scalar and optimized paths bit-for-bit equivalent for integer accumulation.
- Reject null pointers and lengths that are not whole blocks.

---

### Task 1: Reference kernels

**Files:**
- Modify: `include/fox/fox.h`
- Create: `src/ternary.c`
- Create: `src/ternary_internal.h`
- Test: `tests/test_ternary.c`

- [ ] Add `fox_tq1_dot_i8` and `fox_tq2_dot_i8` APIs plus block-count variants.
- [ ] Decode little-endian FP16 scales and canonical packed values in scalar code.
- [ ] Test zero blocks, signed activations, known all-zero/all-one blocks, and invalid arguments.

### Task 2: SSSE3 LUT path

**Files:**
- Modify: `src/ternary.c`
- Modify: `tests/test_ternary.c`
- Modify: `CMakeLists.txt`

- [ ] Add a `pshufb` LUT path for TQ2_0 under x86 SSSE3 compile guards.
- [ ] Select the optimized path only when `fox_probe_cpu()` reports SSSE3.
- [ ] Differential-test optimized and scalar results across deterministic random blocks.
- [ ] Register source and tests, then build with warnings as errors.
