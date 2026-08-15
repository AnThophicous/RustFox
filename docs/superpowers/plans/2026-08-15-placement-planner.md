# Placement Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn validated GGUF tensor descriptors plus hardware/storage measurements into explainable placement decisions and a conservative tok/s estimate.

**Architecture:** The planner is pure policy: it never opens model weights or changes the governor. It classifies tensors by names and dimensions, selects pinned tensors within a residency budget, marks sparse embeddings as mmap, streams the remainder, and computes a lower-bound token rate from measured sequential bandwidth and supplied compute throughput.

**Tech Stack:** C11, existing GGUF and `fox_sysinfo` APIs, CTest.

## Global Constraints

- Never claim unmeasured storage throughput; zero bandwidth produces zero predicted tok/s.
- Never place more pinned bytes than the configured budget.
- Use checked arithmetic and return explicit errors for invalid configuration or allocation failure.
- Keep placement explainable through stable role and reason fields.

---

### Task 1: Planner contract and classification

**Files:**
- Modify: `include/fox/fox.h`
- Create: `src/planner.c`
- Create: `src/planner_internal.h`
- Test: `tests/test_planner.c`

- [ ] Add `fox_plan_config`, `fox_plan_item`, `fox_plan_summary`, opaque `fox_plan`, and open/build/query/free APIs.
- [ ] Classify names containing norm/rope/attn/q_proj/k_proj/v_proj/o_proj/output as hot, embeddings as mmap, and gate/up/down/ffn/expert as stream candidates.
- [ ] Add tests proving budget, role, and unknown-name behavior.

### Task 2: Placement and prediction

**Files:**
- Modify: `src/planner.c`
- Modify: `tests/test_planner.c`
- Modify: `docs/ARCHITECTURE.md`

- [ ] Select hot tensors by priority until budget is exhausted, then mmap embeddings and stream the rest.
- [ ] Calculate streamed bytes, resident bytes, I/O seconds, overlap with compute seconds, and predicted tok/s.
- [ ] Test zero bandwidth, zero compute rate, tiny budget, and overflow-safe totals.

### Task 3: Build integration

**Files:**
- Modify: `CMakeLists.txt`

- [ ] Register planner source and test target.
- [ ] Run `git diff --check`, configure with `FOX_WERROR=ON`, build, and run CTest.

