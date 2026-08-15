#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TQ1_BYTES 54
#define TQ2_BYTES 66
#define BLOCKS    3
#define N         (256 * BLOCKS)

static const uint8_t POW3[5] = { 1, 3, 9, 27, 81 };

static int32_t tq1_weight(uint8_t code, int level)
{
    uint8_t q = (uint8_t)(code * POW3[level]);
    return (int32_t)(((uint32_t)q * 3u) >> 8) - 1;
}

static void set_half_one(uint8_t *p)
{
    p[0] = 0x00;
    p[1] = 0x3C;
}

static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float tq1_reference(const int8_t *x, const uint8_t *blocks, size_t n)
{
    float total = 0.0f;
    size_t base;

    for (base = 0; base < n; base += 256) {
        const uint8_t *b = blocks + (base / 256) * TQ1_BYTES;
        int32_t sum = 0;
        size_t idx = 0;
        int l;
        int m;

        for (l = 0; l < 5; l++)
            for (m = 0; m < 32; m++)
                sum += tq1_weight(b[m], l) * x[base + idx++];
        for (l = 0; l < 5; l++)
            for (m = 0; m < 16; m++)
                sum += tq1_weight(b[32 + m], l) * x[base + idx++];
        for (l = 0; l < 4; l++)
            for (m = 0; m < 4; m++)
                sum += tq1_weight(b[48 + m], l) * x[base + idx++];

        total += (float)sum;
    }
    return total;
}

static float tq2_reference(const int8_t *x, const uint8_t *blocks, size_t n)
{
    float total = 0.0f;
    size_t base;

    for (base = 0; base < n; base += 256) {
        const uint8_t *b = blocks + (base / 256) * TQ2_BYTES;
        int32_t sum = 0;
        size_t idx = 0;
        int j, l, m;

        for (j = 0; j < 64; j += 32)
            for (l = 0; l < 4; l++)
                for (m = 0; m < 32; m++)
                    sum += ((int32_t)((b[j + m] >> (l * 2)) & 3) - 1) * x[base + idx++];

        total += (float)sum;
    }
    return total;
}

int main(void)
{
    int8_t x[N];
    uint8_t tq1[TQ1_BYTES * BLOCKS];
    uint8_t tq2[TQ2_BYTES * BLOCKS];
    float a = 0.0f, b = 0.0f, want;
    uint32_t seed = 0x1234567u;
    size_t i;
    int k;

    for (i = 0; i < N; i++) x[i] = 1;

    memset(tq1, 0, sizeof(tq1));
    memset(tq2, 0, sizeof(tq2));

    for (k = 0; k < BLOCKS; k++) {
        uint8_t *b1 = tq1 + (size_t)k * TQ1_BYTES;
        uint8_t *b2 = tq2 + (size_t)k * TQ2_BYTES;

        memset(b1, 255, 48);
        memset(b1 + 48, 253, 4);
        set_half_one(b1 + 52);

        memset(b2, 0xAA, 64);
        set_half_one(b2 + 64);
    }

    CHECK(tq1_weight(255, 0) == 1 && tq1_weight(255, 4) == 1,
          "255 decodes to +1 at every base-3 level; the encoder scales by "
          "ceil(q*256/243), so an all-plus-one block is 255, not 242");
    CHECK(tq1_weight(253, 0) == 1 && tq1_weight(253, 3) == 1,
          "253 is the all-plus-one qh byte, scaled by ceil(q*256/81)");
    CHECK(tq1_weight(0, 0) == -1 && tq1_weight(0, 4) == -1,
          "a zero byte decodes to -1, not to zero");

    CHECK(fox_tq1_dot_i8(x, tq1, N, &a) == FOX_OK && a == (float)N,
          "TQ1_0 all-plus-one against all-ones activations sums to N");
    CHECK(fox_tq2_dot_i8_scalar(x, tq2, N, &a) == FOX_OK && a == (float)N,
          "TQ2_0 scalar all-plus-one sums to N");
    CHECK(fox_tq2_dot_i8(x, tq2, N, &b) == FOX_OK && b == (float)N,
          "TQ2_0 dispatch agrees on the all-plus-one case");

    memset(tq1, 0, sizeof(tq1));
    memset(tq2, 0, sizeof(tq2));
    for (k = 0; k < BLOCKS; k++) {
        set_half_one(tq1 + (size_t)k * TQ1_BYTES + 52);
        set_half_one(tq2 + (size_t)k * TQ2_BYTES + 64);
    }
    CHECK(fox_tq1_dot_i8(x, tq1, N, &a) == FOX_OK && a == -(float)N,
          "an all-zero TQ1_0 block is all minus one, so it sums to -N");
    CHECK(fox_tq2_dot_i8(x, tq2, N, &a) == FOX_OK && a == -(float)N,
          "an all-zero TQ2_0 block is all minus one, so it sums to -N");

    for (i = 0; i < N; i++) x[i] = (int8_t)((int)(rng_next(&seed) & 0xFF) - 128);
    for (i = 0; i < sizeof(tq1); i++) tq1[i] = (uint8_t)(rng_next(&seed) & 0xFF);
    for (i = 0; i < sizeof(tq2); i++) tq2[i] = (uint8_t)(rng_next(&seed) & 0xFF);
    for (k = 0; k < BLOCKS; k++) {
        set_half_one(tq1 + (size_t)k * TQ1_BYTES + 52);
        set_half_one(tq2 + (size_t)k * TQ2_BYTES + 64);
    }

    want = tq1_reference(x, tq1, N);
    CHECK(fox_tq1_dot_i8_scalar(x, tq1, N, &b) == FOX_OK && b == want,
          "TQ1_0 scalar matches an independent reference on pseudorandom data");
    CHECK(fox_tq1_dot_i8(x, tq1, N, &a) == FOX_OK && a == b,
          "TQ1_0 dispatch is bit-for-bit equal to the scalar reference path");

    want = tq2_reference(x, tq2, N);
    CHECK(fox_tq2_dot_i8_scalar(x, tq2, N, &a) == FOX_OK && a == want,
          "TQ2_0 scalar matches an independent reference on pseudorandom data");
    CHECK(fox_tq2_dot_i8(x, tq2, N, &b) == FOX_OK && b == want,
          "the SSSE3 path agrees with the reference bit for bit; this is the "
          "check that catches the interleaved layout being read sequentially");

    CHECK(fox_tq1_dot_i8(x, tq1, 255, &a) == FOX_ERR_ARG,
          "a partial TQ1_0 block is rejected");
    CHECK(fox_tq2_dot_i8(x, tq2, 255, &a) == FOX_ERR_ARG,
          "a partial TQ2_0 block is rejected");
    CHECK(fox_tq2_dot_i8(NULL, tq2, 256, &a) == FOX_ERR_ARG,
          "a null activation pointer is rejected");
    CHECK(fox_tq2_dot_i8(x, NULL, 256, &a) == FOX_ERR_ARG,
          "a null weight pointer is rejected");
    CHECK(fox_tq2_dot_i8(x, tq2, 256, NULL) == FOX_ERR_ARG,
          "a null result pointer is rejected");
    CHECK(fox_tq2_dot_i8(x, tq2, 0, &a) == FOX_ERR_ARG,
          "a zero-length dot product is rejected");

    CHECK_DONE();
}
