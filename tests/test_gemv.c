#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ROWS      37
#define COLS      512
#define TQ2_BYTES 66
#define BLOCKS    (COLS / 256)
#define ROW_BYTES (BLOCKS * TQ2_BYTES)

static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static void mark_slice(void *ctx, int worker, size_t begin, size_t end)
{
    int *hits = (int *)ctx;
    size_t i;
    (void)worker;
    for (i = begin; i < end; i++) hits[i]++;
}

static int coverage_is_exact(fox_threadpool *tp, size_t n)
{
    int *hits = (int *)calloc(n ? n : 1, sizeof(int));
    size_t i;
    int ok = 1;

    if (!hits) return 0;
    if (fox_parallel_for(tp, n, mark_slice, hits) != FOX_OK) {
        free(hits);
        return 0;
    }
    for (i = 0; i < n; i++)
        if (hits[i] != 1) { ok = 0; break; }
    free(hits);
    return ok;
}

int main(void)
{
    fox_threadpool *tp;
    uint8_t *w;
    int8_t q[COLS];
    float x[COLS];
    float serial[ROWS], parallel[ROWS], direct[ROWS];
    float scale = 0.0f;
    uint32_t seed = 0xC0FFEEu;
    size_t i;
    int r, ok;

    tp = fox_threadpool_create(4);
    CHECK(tp != NULL, "threadpool is created");
    CHECK(fox_threadpool_size(tp) >= 1, "threadpool reports its width");
    CHECK(fox_threadpool_size(NULL) == 1, "a null pool behaves as one thread");

    CHECK(coverage_is_exact(tp, 1000),
          "parallel_for covers every index exactly once");
    CHECK(coverage_is_exact(tp, 1),
          "a single item is still visited exactly once");
    CHECK(coverage_is_exact(tp, 3),
          "fewer items than threads leaves no worker double-counting");
    CHECK(coverage_is_exact(tp, 4097),
          "an index count coprime with the thread count still splits cleanly");
    CHECK(coverage_is_exact(NULL, 500),
          "a null pool runs the whole range inline");
    CHECK(fox_parallel_for(tp, 0, mark_slice, NULL) == FOX_OK,
          "an empty range is a no-op, not an error");
    CHECK(fox_parallel_for(tp, 10, NULL, NULL) == FOX_ERR_ARG,
          "a null callback is rejected");

    for (i = 0; i < COLS; i++) x[i] = 0.0f;
    CHECK(fox_quantize_activations_i8(x, COLS, q, &scale) == FOX_OK &&
          scale == 0.0f && q[0] == 0 && q[COLS - 1] == 0,
          "an all-zero activation vector quantises to zero with a zero scale");

    for (i = 0; i < COLS; i++) x[i] = (float)i - 200.0f;
    CHECK(fox_quantize_activations_i8(x, COLS, q, &scale) == FOX_OK,
          "activation quantisation succeeds");
    CHECK(scale > 0.0f, "a non-zero vector gets a positive scale");
    ok = 1;
    for (i = 0; i < COLS; i++) if (q[i] < -127 || q[i] > 127) ok = 0;
    CHECK(ok, "every quantised activation stays inside the int8 range");
    CHECK(q[COLS - 1] == 127,
          "the largest magnitude maps to the top of the range");

    w = (uint8_t *)malloc(ROWS * ROW_BYTES);
    CHECK(w != NULL, "weight buffer allocated");
    if (!w) { fox_threadpool_destroy(tp); CHECK_DONE(); }

    for (i = 0; i < (size_t)ROWS * ROW_BYTES; i++)
        w[i] = (uint8_t)(rng_next(&seed) & 0xFF);
    for (r = 0; r < ROWS; r++) {
        int b;
        for (b = 0; b < BLOCKS; b++) {
            uint8_t *blk = w + (size_t)r * ROW_BYTES + (size_t)b * TQ2_BYTES;
            blk[64] = 0x00;
            blk[65] = 0x3C;
        }
    }

    for (i = 0; i < COLS; i++)
        x[i] = (float)((int)(rng_next(&seed) & 0xFF) - 128) * 0.01f;
    CHECK(fox_quantize_activations_i8(x, COLS, q, &scale) == FOX_OK,
          "activations quantised for the gemv comparison");

    CHECK(fox_gemv_tq2_i8(NULL, q, scale, w, ROWS, COLS, serial) == FOX_OK,
          "single-threaded gemv runs");
    CHECK(fox_gemv_tq2_i8(tp, q, scale, w, ROWS, COLS, parallel) == FOX_OK,
          "multi-threaded gemv runs");

    ok = 1;
    for (r = 0; r < ROWS; r++) if (serial[r] != parallel[r]) ok = 0;
    CHECK(ok,
          "threading changes nothing about the result; row work is disjoint so "
          "the sum order per row is identical either way");

    for (r = 0; r < ROWS; r++) {
        float acc = 0.0f;
        fox_tq2_dot_i8(q, w + (size_t)r * ROW_BYTES, COLS, &acc);
        direct[r] = acc * scale;
    }
    ok = 1;
    for (r = 0; r < ROWS; r++) if (direct[r] != serial[r]) ok = 0;
    CHECK(ok, "gemv rows match the per-row dot product exactly");

    CHECK(fox_gemv_tq2_i8(tp, q, scale, w, ROWS, 255, serial) == FOX_ERR_ARG,
          "a column count that is not a whole number of blocks is rejected");
    CHECK(fox_gemv_tq2_i8(tp, NULL, scale, w, ROWS, COLS, serial) == FOX_ERR_ARG,
          "null activations are rejected");
    CHECK(fox_gemv_tq2_i8(tp, q, scale, NULL, ROWS, COLS, serial) == FOX_ERR_ARG,
          "null weights are rejected");
    CHECK(fox_gemv_tq2_i8(tp, q, scale, w, 0, COLS, serial) == FOX_ERR_ARG,
          "zero rows is rejected");

    memset(parallel, 0, sizeof(parallel));
    CHECK(fox_gemv_tq2_f32(tp, x, w, ROWS, COLS, parallel) == FOX_OK,
          "the float entry point quantises and dispatches");
    ok = 1;
    for (r = 0; r < ROWS; r++) if (parallel[r] != serial[r]) ok = 0;
    CHECK(ok,
          "the float entry point reproduces the pre-quantised path exactly, "
          "so callers can hoist quantisation out of a loop without drift");

    free(w);
    fox_threadpool_destroy(tp);
    fox_threadpool_destroy(NULL);

    CHECK_DONE();
}
