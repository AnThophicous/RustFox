#include "fox/fox.h"
#include "fox_internal.h"
#include "platform.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(FOX_ARCH_X86)
#  include <tmmintrin.h>
#  define FOX_TERNARY_X86 1
#endif

#define TQ_BLOCK        256
#define TQ1_0_BYTES     54
#define TQ2_0_BYTES     66
#define TQ1_0_QS_BYTES  48
#define TQ1_0_QH_BYTES  4

static const uint8_t tq_pow3[5] = { 1, 3, 9, 27, 81 };

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15);
    int exp = (int)((h >> 10) & 0x1F);
    uint32_t frac = h & 0x3FFu;
    union { uint32_t u; float f; } value;

    if (exp == 0) {
        if (frac == 0) {
            value.u = sign << 31;
            return value.f;
        }
        while ((frac & 0x400u) == 0) {
            frac <<= 1;
            exp--;
        }
        exp++;
        frac &= 0x3FFu;
    } else if (exp == 31) {
        value.u = (sign << 31) | 0x7F800000u | (frac << 13);
        return value.f;
    }

    value.u = (sign << 31) | ((uint32_t)(exp + (127 - 15)) << 23) | (frac << 13);
    return value.f;
}

static float block_scale(const uint8_t *block, size_t offset)
{
    uint16_t h = (uint16_t)block[offset] | ((uint16_t)block[offset + 1] << 8);
    return half_to_float(h);
}

static int32_t tq1_weight(uint8_t code, size_t level)
{
    uint8_t q = (uint8_t)(code * tq_pow3[level]);
    return (int32_t)(((uint32_t)q * 3u) >> 8) - 1;
}

static fox_status check_args(const int8_t *activations, const void *data,
                             size_t n, float *out)
{
    if (!activations || !data || !out) return FOX_ERR_ARG;
    if (n == 0 || n % TQ_BLOCK != 0) return FOX_ERR_ARG;
    return FOX_OK;
}

static int32_t tq1_block_scalar(const int8_t *x, const uint8_t *block)
{
    const uint8_t *qs = block;
    const uint8_t *qh = block + TQ1_0_QS_BYTES;
    int32_t sum = 0;
    size_t l, m, idx = 0;

    for (l = 0; l < 5; l++)
        for (m = 0; m < 32; m++)
            sum += tq1_weight(qs[m], l) * x[idx++];

    for (l = 0; l < 5; l++)
        for (m = 0; m < 16; m++)
            sum += tq1_weight(qs[32 + m], l) * x[idx++];

    for (l = 0; l < 4; l++)
        for (m = 0; m < 4; m++)
            sum += tq1_weight(qh[m], l) * x[idx++];

    return sum;
}

static int32_t tq2_block_scalar(const int8_t *x, const uint8_t *qs)
{
    int32_t sum = 0;
    size_t j, l, m, idx = 0;

    for (j = 0; j < 64; j += 32)
        for (l = 0; l < 4; l++)
            for (m = 0; m < 32; m++)
                sum += ((int32_t)((qs[j + m] >> (l * 2)) & 3) - 1) * x[idx++];

    return sum;
}

fox_status fox_tq1_dot_i8_scalar(const int8_t *activations, const void *block_data,
                                 size_t n, float *out)
{
    const uint8_t *blocks = (const uint8_t *)block_data;
    fox_status st = check_args(activations, block_data, n, out);
    size_t base;

    if (st != FOX_OK) return st;

    *out = 0.0f;
    for (base = 0; base < n; base += TQ_BLOCK) {
        const uint8_t *block = blocks + (base / TQ_BLOCK) * TQ1_0_BYTES;
        int32_t sum = tq1_block_scalar(activations + base, block);
        *out += block_scale(block, TQ1_0_BYTES - 2) * (float)sum;
    }
    return FOX_OK;
}

fox_status fox_tq2_dot_i8_scalar(const int8_t *activations, const void *block_data,
                                 size_t n, float *out)
{
    const uint8_t *blocks = (const uint8_t *)block_data;
    fox_status st = check_args(activations, block_data, n, out);
    size_t base;

    if (st != FOX_OK) return st;

    *out = 0.0f;
    for (base = 0; base < n; base += TQ_BLOCK) {
        const uint8_t *block = blocks + (base / TQ_BLOCK) * TQ2_0_BYTES;
        int32_t sum = tq2_block_scalar(activations + base, block);
        *out += block_scale(block, TQ2_0_BYTES - 2) * (float)sum;
    }
    return FOX_OK;
}

fox_status fox_tq1_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out)
{
    return fox_tq1_dot_i8_scalar(activations, block_data, n, out);
}

#if defined(FOX_TERNARY_X86)

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("ssse3")))
#endif
static int32_t tq2_block_ssse3(const int8_t *x, const uint8_t *qs)
{
    const __m128i mask   = _mm_set1_epi8(3);
    const __m128i ones8  = _mm_set1_epi8(1);
    const __m128i ones16 = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    int group, m0;

#define TQ2_LANE(SHIFT, LEVEL)                                                 \
    do {                                                                       \
        __m128i codes = _mm_and_si128(_mm_srli_epi16(packed, (SHIFT)), mask);  \
        __m128i a = _mm_loadu_si128((const __m128i *)(xa + (LEVEL) * 32));     \
        __m128i p = _mm_sub_epi16(_mm_maddubs_epi16(codes, a),                 \
                                  _mm_maddubs_epi16(ones8, a));                \
        acc = _mm_add_epi32(acc, _mm_madd_epi16(p, ones16));                   \
    } while (0)

    for (group = 0; group < 2; group++) {
        for (m0 = 0; m0 < 32; m0 += 16) {
            const int8_t *xa = x + group * 128 + m0;
            __m128i packed =
                _mm_loadu_si128((const __m128i *)(qs + group * 32 + m0));
            TQ2_LANE(0, 0);
            TQ2_LANE(2, 1);
            TQ2_LANE(4, 2);
            TQ2_LANE(6, 3);
        }
    }

#undef TQ2_LANE

    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
}

static int ternary_have_ssse3(void)
{
    static int cached = -1;

    if (cached < 0) {
        fox_cpu_info cpu;
        fox_probe_cpu(&cpu);
        cached = (cpu.features & FOX_CPU_SSSE3) ? 1 : 0;
    }
    return cached;
}

#endif

fox_status fox_tq2_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out)
{
#if defined(FOX_TERNARY_X86)
    const uint8_t *blocks = (const uint8_t *)block_data;
    fox_status st = check_args(activations, block_data, n, out);
    size_t base;

    if (st != FOX_OK) return st;
    if (!ternary_have_ssse3())
        return fox_tq2_dot_i8_scalar(activations, block_data, n, out);

    *out = 0.0f;
    for (base = 0; base < n; base += TQ_BLOCK) {
        const uint8_t *block = blocks + (base / TQ_BLOCK) * TQ2_0_BYTES;
        int32_t sum = tq2_block_ssse3(activations + base, block);
        *out += block_scale(block, TQ2_0_BYTES - 2) * (float)sum;
    }
    return FOX_OK;
#else
    return fox_tq2_dot_i8_scalar(activations, block_data, n, out);
#endif
}
