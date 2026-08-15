# GGUF Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, read-only GGUF parser to `foxcore` that exposes validated model metadata and tensor placement data, including TQ1_0/TQ2_0 tensor types.

**Architecture:** `fox_gguf_open()` reads a file into a private parser context, validates the GGUF header, typed key/value records, tensor descriptors, alignment, and tensor data extents before returning. Metadata is retained in a compact typed table; tensors retain owned names, dimensions, type, relative offset, and computed byte size. No weight decoding or memory mapping is included; kernels and the placement planner consume the validated descriptors later.

**Tech Stack:** C11, CMake, existing `fox_status`/`fox_fail` error model, TAP-style C tests.

## Global Constraints

- Support GGUF versions 2 and 3; reject other versions.
- Treat all serialized integers as little-endian and reject truncation, overflow, invalid counts, invalid strings, invalid alignment, and tensor extents outside the file.
- Keep parser APIs independent from `llama.cpp`; use GGUF enum values compatible with canonical `gguf.h`.
- Recognize `GGML_TYPE_TQ1_0` (34) and `GGML_TYPE_TQ2_0` (35); decoding remains outside this feature.
- Do not read or allocate the tensor data blob; parser memory usage must depend on headers, names, and metadata only.

---

### Task 1: Public GGUF contracts

**Files:**
- Modify: `include/fox/fox.h`
- Modify: `CMakeLists.txt`
- Create: `src/gguf.c`
- Create: `src/gguf_internal.h`

**Interfaces:**
- Produces opaque `fox_gguf`, `fox_gguf_open`, `fox_gguf_close`, count accessors, metadata lookup, tensor access, and type/block-size helpers.
- `fox_gguf_tensor` contains `name`, `n_dims`, `ne[GGUF_MAX_DIMS]`, `type`, `offset`, and `size_bytes`.

- [ ] Define public enums for GGUF scalar types and GGML tensor types, preserving canonical numeric IDs.
- [ ] Define `fox_gguf_open(const char *, fox_gguf **)` and null-safe close/accessors.
- [ ] Implement bounded binary readers and checked arithmetic in `src/gguf.c`.
- [ ] Parse header, metadata, tensor descriptors, and aligned data start without loading weights.
- [ ] Compute tensor byte sizes for supported dense and quantized types; reject types whose layout is unknown instead of guessing.

### Task 2: Typed metadata and tensor queries

**Files:**
- Modify: `src/gguf.c`
- Modify: `include/fox/fox.h`
- Test: `tests/test_gguf.c`

**Interfaces:**
- `fox_gguf_find(const fox_gguf *, const char *)` returns a metadata index or `SIZE_MAX`.
- Typed getters return `FOX_ERR_ARG`, `FOX_ERR_NOTFOUND`, or `FOX_ERR_FORMAT` as appropriate.
- `fox_gguf_tensor_at()` and `fox_gguf_tensor_find()` expose validated descriptors.

- [ ] Add a fixture builder in the test executable for minimal valid GGUF v3 files.
- [ ] Test scalar metadata, string metadata, arrays, tensor dimensions, offsets, alignment, and TQ1_0/TQ2_0 descriptors.
- [ ] Test duplicate keys, missing NUL assumptions, oversized counts, truncated records, bad alignment, and out-of-file tensor extents.
- [ ] Implement metadata lookup and typed getters with no borrowed pointers into temporary buffers.

### Task 3: Build and regression verification

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `docs/ARCHITECTURE.md`
- Test: `tests/test_gguf.c`

- [ ] Register `gguf.c` in `foxcore` and `test_gguf` in CTest.
- [ ] Document parser ownership, no-weight-loading behavior, and the TQ type boundary.
- [ ] Build with normal warnings and `FOX_WERROR=ON`.
- [ ] Run the full CTest suite and verify the parser tests pass on generated fixtures.

