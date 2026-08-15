#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define N 64

static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static void put_half(uint8_t *p, uint16_t h)
{
    p[0] = (uint8_t)(h & 0xFF);
    p[1] = (uint8_t)(h >> 8);
}

static void ref_q4_0(const uint8_t *blk, float *out)
{
    float d = fox_f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    int j;
    for (j = 0; j < 32; j++) {
        int nib = (j < 16) ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
        out[j] = (float)(nib - 8) * d;
    }
}

static void ref_q8_0(const uint8_t *blk, float *out)
{
    float d = fox_f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    int j;
    for (j = 0; j < 32; j++) out[j] = (float)(int8_t)blk[2 + j] * d;
}

static void ref_q5_0(const uint8_t *blk, float *out)
{
    float d = fox_f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
    uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                  ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    int j;
    for (j = 0; j < 32; j++) {
        int nib, bit;
        if (j < 16) {
            nib = blk[6 + j] & 0x0F;
            bit = (int)((qh >> j) & 1u);
        } else {
            nib = blk[6 + j - 16] >> 4;
            bit = (int)((qh >> (j - 16 + 16)) & 1u);
        }
        out[j] = (float)((nib | (bit << 4)) - 16) * d;
    }
}

static int probe_one_hot(fox_ggml_type type, const uint8_t *blk,
                         const float *expected)
{
    float x[32];
    float got = 0.0f;
    int k;

    for (k = 0; k < 32; k++) {
        memset(x, 0, sizeof(x));
        x[k] = 1.0f;
        if (fox_gemv(NULL, type, x, blk, 1, 32, &got) != FOX_OK) return 0;
        if (got != expected[k]) return 0;
    }
    return 1;
}

int main(void)
{
    uint8_t q4[18], q8[34], q5[22];
    float ref[32];
    float x[N], w32[N];
    float out_a[3], out_b[3];
    float deq[N];
    uint8_t f16row[N * 2];
    fox_threadpool *tp;
    uint32_t seed = 0x51EEDu;
    size_t i;
    int ok, k;

    CHECK(fox_f16_to_f32(0x3C00) == 1.0f, "0x3C00 is 1.0");
    CHECK(fox_f16_to_f32(0x0000) == 0.0f, "0x0000 is zero");
    CHECK(fox_f16_to_f32(0xBC00) == -1.0f, "0xBC00 is -1.0");
    CHECK(fox_f16_to_f32(0x4000) == 2.0f, "0x4000 is 2.0");
    CHECK(fox_f16_to_f32(0x3800) == 0.5f, "0x3800 is 0.5");
    CHECK(fox_f16_to_f32(0x0001) > 0.0f, "the smallest subnormal is positive");
    CHECK(fox_f32_to_f16(1.0f) == 0x3C00, "1.0 encodes back to 0x3C00");
    CHECK(fox_f32_to_f16(-2.0f) == 0xC000, "-2.0 encodes back to 0xC000");
    CHECK(fox_f16_to_f32(fox_f32_to_f16(0.25f)) == 0.25f,
          "an exactly representable value survives the round trip");

    CHECK(fox_row_bytes(FOX_GGML_F32, 64) == 256, "F32 rows are 4 bytes each");
    CHECK(fox_row_bytes(FOX_GGML_F16, 64) == 128, "F16 rows are 2 bytes each");
    CHECK(fox_row_bytes(FOX_GGML_Q4_0, 64) == 36, "Q4_0 is 18 bytes per 32");
    CHECK(fox_row_bytes(FOX_GGML_Q4_0, 33) == 0,
          "a column count that is not a whole block reports zero");
    CHECK(fox_gemv_supports(FOX_GGML_TQ2_0) && fox_gemv_supports(FOX_GGML_Q8_0),
          "ternary and Q8_0 are supported");
    CHECK(!fox_gemv_supports(FOX_GGML_Q6_K),
          "an unimplemented type says no rather than pretending");

    put_half(q4, 0x3C00);
    for (i = 0; i < 16; i++) q4[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q4_0(q4, ref);
    CHECK(probe_one_hot(FOX_GGML_Q4_0, q4, ref),
          "Q4_0 places element j in the low nibble and j+16 in the high nibble, "
          "verified one position at a time");

    put_half(q8, 0x3800);
    for (i = 0; i < 32; i++) q8[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q8_0(q8, ref);
    CHECK(probe_one_hot(FOX_GGML_Q8_0, q8, ref),
          "Q8_0 elements land in order and the scale applies");

    put_half(q5, 0x3C00);
    for (i = 0; i < 20; i++) q5[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q5_0(q5, ref);
    CHECK(probe_one_hot(FOX_GGML_Q5_0, q5, ref),
          "Q5_0 takes its fifth bit from qh bit j for the low half and bit j+16 "
          "for the high half");

    for (i = 0; i < N; i++) {
        x[i]   = (float)((int)(rng_next(&seed) & 0x3F) - 32) * 0.125f;
        w32[i] = (float)((int)(rng_next(&seed) & 0x3F) - 32) * 0.0625f;
    }
    {
        float want = 0.0f, got = 0.0f;
        for (i = 0; i < N; i++) want += w32[i] * x[i];
        CHECK(fox_gemv(NULL, FOX_GGML_F32, x, w32, 1, N, &got) == FOX_OK &&
              got == want, "the F32 kernel is a plain dot product");
    }

    for (i = 0; i < N; i++) put_half(f16row + i * 2, fox_f32_to_f16(w32[i]));
    {
        float want = 0.0f, got = 0.0f;
        for (i = 0; i < N; i++)
            want += fox_f16_to_f32((uint16_t)f16row[i * 2] |
                                   ((uint16_t)f16row[i * 2 + 1] << 8)) * x[i];
        CHECK(fox_gemv(NULL, FOX_GGML_F16, x, f16row, 1, N, &got) == FOX_OK &&
              got == want, "the F16 kernel converts then multiplies");
    }

    CHECK(fox_dequant_row(FOX_GGML_F16, f16row, N, deq) == FOX_OK, "F16 dequant runs");
    ok = 1;
    for (i = 0; i < N; i++)
        if (deq[i] != fox_f16_to_f32((uint16_t)f16row[i * 2] |
                                     ((uint16_t)f16row[i * 2 + 1] << 8))) ok = 0;
    CHECK(ok, "dequant and the kernel agree on what each F16 weight is");

    CHECK(fox_dequant_row(FOX_GGML_Q4_0, q4, 32, deq) == FOX_OK, "Q4_0 dequant runs");
    ref_q4_0(q4, ref);
    ok = 1;
    for (k = 0; k < 32; k++) if (deq[k] != ref[k]) ok = 0;
    CHECK(ok, "Q4_0 dequant matches the reference element for element");

    CHECK(fox_dequant_row(FOX_GGML_Q6_K, q4, 256, deq) == FOX_ERR_ARG ||
          fox_dequant_row(FOX_GGML_Q6_K, q4, 256, deq) == FOX_ERR_UNSUPPORTED,
          "an unimplemented dequant refuses instead of returning noise");

    tp = fox_threadpool_create(4);
    {
        uint8_t rows[3 * 34];
        for (i = 0; i < sizeof(rows); i++) rows[i] = (uint8_t)(rng_next(&seed) & 0xFF);
        for (k = 0; k < 3; k++) put_half(rows + (size_t)k * 34, 0x3C00);

        CHECK(fox_gemv(NULL, FOX_GGML_Q8_0, x, rows, 3, 32, out_a) == FOX_OK,
              "multi-row gemv runs single-threaded");
        CHECK(fox_gemv(tp, FOX_GGML_Q8_0, x, rows, 3, 32, out_b) == FOX_OK,
              "multi-row gemv runs threaded");
        ok = 1;
        for (k = 0; k < 3; k++) if (out_a[k] != out_b[k]) ok = 0;
        CHECK(ok, "threading does not change a single bit of the result");

        ok = 1;
        for (k = 0; k < 3; k++) {
            float want = 0.0f;
            ref_q8_0(rows + (size_t)k * 34, ref);
            for (i = 0; i < 32; i++) want += ref[i] * x[i];
            if (out_a[k] != want) ok = 0;
        }
        CHECK(ok, "each output row is the dot product of that row alone");
    }

    CHECK(fox_gemv(tp, FOX_GGML_Q6_K, x, q4, 1, 256, out_a) == FOX_ERR_UNSUPPORTED,
          "an unsupported type is named in the error rather than crashing");
    CHECK(fox_gemv(tp, FOX_GGML_Q4_0, x, q4, 1, 33, out_a) == FOX_ERR_ARG,
          "a ragged column count is rejected");
    CHECK(fox_gemv(tp, FOX_GGML_F32, NULL, w32, 1, N, out_a) == FOX_ERR_ARG,
          "null activations are rejected");

    fox_threadpool_destroy(tp);
    CHECK_DONE();
}
