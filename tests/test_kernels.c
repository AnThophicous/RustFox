#include "fox/fox.h"
#include "check.h"

#include <math.h>
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

static float half_at(const uint8_t *p)
{
    return fox_f16_to_f32((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static void ref_q4_0(const uint8_t *blk, float *out)
{
    float d = half_at(blk);
    int j;
    for (j = 0; j < 32; j++) {
        int nib = (j < 16) ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
        out[j] = (float)(nib - 8) * d;
    }
}

static void ref_q8_0(const uint8_t *blk, float *out)
{
    float d = half_at(blk);
    int j;
    for (j = 0; j < 32; j++) out[j] = (float)(int8_t)blk[2 + j] * d;
}

static void ref_q5_0(const uint8_t *blk, float *out)
{
    float d = half_at(blk);
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
            bit = (int)((qh >> j) & 1u);
        }
        out[j] = (float)((nib | (bit << 4)) - 16) * d;
    }
}

static void ref_q6_k(const uint8_t *blk, float *out)
{
    float d = half_at(blk + 208);
    int i;

    for (i = 0; i < 256; i++) {
        int half   = i / 128;
        int within = i % 128;
        int group  = within / 32;
        int l      = within % 32;
        const uint8_t *ql = blk + (size_t)half * 64;
        const uint8_t *qh = blk + 128 + (size_t)half * 32;
        const int8_t  *sc = (const int8_t *)(blk + 192) + (size_t)half * 8;
        int is = l / 16;
        int lo, hib, sidx, q;

        if (group == 0)      { lo = ql[l]      & 0x0F; hib = (qh[l] >> 0) & 3; sidx = is + 0; }
        else if (group == 1) { lo = ql[l + 32] & 0x0F; hib = (qh[l] >> 2) & 3; sidx = is + 2; }
        else if (group == 2) { lo = ql[l]      >> 4;   hib = (qh[l] >> 4) & 3; sidx = is + 4; }
        else                 { lo = ql[l + 32] >> 4;   hib = (qh[l] >> 6) & 3; sidx = is + 6; }

        q = (lo | (hib << 4)) - 32;
        out[i] = d * (float)sc[sidx] * (float)q;
    }
}

static void ref_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = (uint8_t)(q[j]     & 63);
        *m = (uint8_t)(q[j + 4] & 63);
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)   | ((q[j]     >> 6) << 4));
    }
}

static void ref_q4_k(const uint8_t *blk, float *out)
{
    float d    = half_at(blk);
    float dmin = half_at(blk + 2);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int i;

    for (i = 0; i < 256; i++) {
        int pair   = i / 64;
        int within = i % 64;
        int upper  = within / 32;
        int l      = within % 32;
        int j      = pair * 2 + upper;
        uint8_t sc, m;
        int nib;

        ref_scale_min(j, scales, &sc, &m);
        nib = upper ? (qs[pair * 32 + l] >> 4) : (qs[pair * 32 + l] & 0x0F);
        out[i] = (d * (float)sc) * (float)nib - dmin * (float)m;
    }
}

static int close_enough(float got, float want)
{
    float limit = 0.12f * (1.0f + fabsf(want));
    return fabsf(got - want) <= limit;
}

static int probe_one_hot_tol(fox_ggml_type type, const uint8_t *blk,
                             const float *expected, size_t n, float tol)
{
    static float x[256];
    float got = 0.0f;
    size_t k;

    if (n > 256) return 0;

    for (k = 0; k < n; k++) {
        memset(x, 0, n * sizeof(float));
        x[k] = 1.0f;
        if (fox_gemv(NULL, type, x, blk, 1, n, &got) != FOX_OK) return 0;
        if (tol == 0.0f) {
            if (got != expected[k]) return 0;
        } else {
            float slack = tol * (1.0f + fabsf(expected[k]));
            if (fabsf(got - expected[k]) > slack) return 0;
        }
    }
    return 1;
}

static int probe_one_hot(fox_ggml_type type, const uint8_t *blk,
                         const float *expected, size_t n)
{
    return probe_one_hot_tol(type, blk, expected, n, 0.0f);
}

int main(void)
{
    uint8_t q4[18], q8[34], q5[22], q6k[210], q4k[144];
    float ref[256], deq[256];
    float x[N], w32[N];
    float qk_x[256];
    float out_a[3], out_b[3];
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
    CHECK(fox_row_bytes(FOX_GGML_Q6_K, 256) == 210, "Q6_K is 210 bytes per 256");
    CHECK(fox_row_bytes(FOX_GGML_Q4_K, 512) == 288, "Q4_K is 144 bytes per 256");
    CHECK(fox_row_bytes(FOX_GGML_Q4_0, 33) == 0,
          "a column count that is not a whole block reports zero");
    CHECK(fox_gemv_supports(FOX_GGML_TQ2_0) && fox_gemv_supports(FOX_GGML_Q8_0) &&
          fox_gemv_supports(FOX_GGML_Q6_K) && fox_gemv_supports(FOX_GGML_Q4_K),
          "ternary, legacy and the two K-quants that matter are supported");
    CHECK(!fox_gemv_supports(FOX_GGML_Q2_K) && !fox_gemv_supports(FOX_GGML_IQ4_XS),
          "an unimplemented type says no rather than pretending");

    put_half(q4, 0x3C00);
    for (i = 0; i < 16; i++) q4[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q4_0(q4, ref);
    CHECK(probe_one_hot(FOX_GGML_Q4_0, q4, ref, 32),
          "Q4_0 places element j in the low nibble and j+16 in the high nibble, "
          "verified one position at a time");

    put_half(q8, 0x3800);
    for (i = 0; i < 32; i++) q8[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q8_0(q8, ref);
    CHECK(probe_one_hot(FOX_GGML_Q8_0, q8, ref, 32),
          "Q8_0 elements land in order and the scale applies");

    put_half(q5, 0x3C00);
    for (i = 0; i < 20; i++) q5[2 + i] = (uint8_t)(rng_next(&seed) & 0xFF);
    ref_q5_0(q5, ref);
    CHECK(probe_one_hot(FOX_GGML_Q5_0, q5, ref, 32),
          "Q5_0 takes its fifth bit from qh bit j for both halves");

    for (i = 0; i < 208; i++) q6k[i] = (uint8_t)(rng_next(&seed) & 0xFF);
    put_half(q6k + 208, 0x3C00);
    ref_q6_k(q6k, ref);
    CHECK(probe_one_hot_tol(FOX_GGML_Q6_K, q6k, ref, 256, 1e-3f),
          "Q6_K interleaves four groups of 32 across two 128-element halves, "
          "each with its own scale slot; probed at all 256 positions");

    for (i = 0; i < 144; i++) q4k[i] = (uint8_t)(rng_next(&seed) & 0xFF);
    put_half(q4k, 0x3C00);
    put_half(q4k + 2, 0x3800);
    ref_q4_k(q4k, ref);
    CHECK(probe_one_hot_tol(FOX_GGML_Q4_K, q4k, ref, 256, 1e-3f),
          "Q4_K unpacks its 6-bit scales and mins from the 12 packed bytes and "
          "applies them to the right 32-element run; probed at all 256 positions");

    CHECK(fox_dequant_row(FOX_GGML_Q6_K, q6k, 256, deq) == FOX_OK, "Q6_K dequant runs");
    ref_q6_k(q6k, ref);
    ok = 1;
    for (k = 0; k < 256; k++) if (deq[k] != ref[k]) ok = 0;
    CHECK(ok, "Q6_K dequant matches the reference element for element");

    CHECK(fox_dequant_row(FOX_GGML_Q4_K, q4k, 256, deq) == FOX_OK, "Q4_K dequant runs");
    ref_q4_k(q4k, ref);
    ok = 1;
    for (k = 0; k < 256; k++) if (deq[k] != ref[k]) ok = 0;
    CHECK(ok, "Q4_K dequant matches the reference element for element");

    for (i = 0; i < 256; i++)
        qk_x[i] = (float)((int)(rng_next(&seed) & 0xFF) - 128) * 0.01f;
    ref_q6_k(q6k, ref);
    {
        float want = 0.0f, got = 0.0f;
        for (i = 0; i < 256; i++) want += ref[i] * qk_x[i];
        CHECK(fox_gemv(NULL, FOX_GGML_Q6_K, qk_x, q6k, 1, 256, &got) == FOX_OK &&
              close_enough(got, want),
              "Q6_K integer SIMD GEMV stays within the quantized activation error bound");
    }
    ref_q4_k(q4k, ref);
    {
        float want = 0.0f, got = 0.0f;
        for (i = 0; i < 256; i++) want += ref[i] * qk_x[i];
        CHECK(fox_gemv(NULL, FOX_GGML_Q4_K, qk_x, q4k, 1, 256, &got) == FOX_OK &&
              close_enough(got, want),
              "Q4_K integer SIMD GEMV stays within the quantized activation error bound");
    }

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
        for (i = 0; i < N; i++) want += half_at(f16row + i * 2) * x[i];
        CHECK(fox_gemv(NULL, FOX_GGML_F16, x, f16row, 1, N, &got) == FOX_OK &&
              got == want, "the F16 kernel converts then multiplies");
    }

    CHECK(fox_dequant_row(FOX_GGML_F16, f16row, N, deq) == FOX_OK, "F16 dequant runs");
    ok = 1;
    for (i = 0; i < N; i++) if (deq[i] != half_at(f16row + i * 2)) ok = 0;
    CHECK(ok, "dequant and the kernel agree on what each F16 weight is");

    CHECK(fox_dequant_row(FOX_GGML_Q4_0, q4, 32, deq) == FOX_OK, "Q4_0 dequant runs");
    ref_q4_0(q4, ref);
    ok = 1;
    for (k = 0; k < 32; k++) if (deq[k] != ref[k]) ok = 0;
    CHECK(ok, "Q4_0 dequant matches the reference element for element");

    CHECK(fox_dequant_row(FOX_GGML_Q2_K, q6k, 256, deq) == FOX_ERR_UNSUPPORTED,
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

    CHECK(fox_gemv(tp, FOX_GGML_Q2_K, x, q6k, 1, 256, out_a) == FOX_ERR_UNSUPPORTED,
          "an unsupported type is named in the error rather than crashing");
    CHECK(fox_gemv(tp, FOX_GGML_Q4_0, x, q4, 1, 33, out_a) == FOX_ERR_ARG,
          "a ragged column count is rejected");
    CHECK(fox_gemv(tp, FOX_GGML_F32, NULL, w32, 1, N, out_a) == FOX_ERR_ARG,
          "null activations are rejected");

    fox_threadpool_destroy(tp);
    CHECK_DONE();
}
