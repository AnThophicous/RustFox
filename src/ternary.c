#include "fox/fox.h"
#include "platform.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <tmmintrin.h>
#define FOX_X86_SSSE3 1
#endif

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15);
    int exp = (int)((h >> 10) & 0x1f);
    uint32_t frac = h & 0x3ffu;
    uint32_t bits;
    union { uint32_t u; float f; } value;

    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        while ((frac & 0x400u) == 0) { frac <<= 1; exp--; }
        exp++;
        frac &= 0x3ffu;
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (frac << 13);
        value.u = bits;
        return value.f;
    }
    bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13);
    value.u = bits;
    return value.f;
}

static float tq_scale(const uint8_t *block, size_t offset)
{
    uint16_t h = (uint16_t)block[offset] | ((uint16_t)block[offset + 1] << 8);
    return half_to_float(h);
}

static fox_status validate_args(const int8_t *activations, const void *data,
                                size_t n, size_t block, float *out)
{
    if (!activations || !data || !out) return FOX_ERR_ARG;
    if (n == 0 || n % block != 0) return FOX_ERR_ARG;
    return FOX_OK;
}

fox_status fox_tq1_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out)
{
    const uint8_t *blocks = (const uint8_t *)block_data;
    size_t base, i, j;
    int32_t sum;
    fox_status status = validate_args(activations, block_data, n, 256, out);
    if (status != FOX_OK) return status;
    *out = 0.0f;
    for (base = 0; base < n; base += 256) {
        const uint8_t *block = blocks + (base / 256) * 54;
        sum = 0;
        for (i = 0; i < 48; i++) {
            uint8_t packed = block[i];
            for (j = 0; j < 5; j++) {
                int32_t q = (int32_t)(packed % 3) - 1;
                sum += q * activations[base + i * 5 + j];
                packed = (uint8_t)(packed / 3);
            }
        }
        for (i = 0; i < 16; i++) {
            int32_t q = (int32_t)((block[48 + i / 4] >> ((i % 4) * 2)) & 3) - 1;
            sum += q * activations[base + 240 + i];
        }
        *out += tq_scale(block, 52) * (float)sum;
    }
    return FOX_OK;
}

fox_status fox_tq2_dot_i8_scalar(const int8_t *activations, const void *block_data,
                                 size_t n, float *out)
{
    const uint8_t *blocks = (const uint8_t *)block_data;
    size_t base, i, j;
    fox_status status = validate_args(activations, block_data, n, 256, out);
    if (status != FOX_OK) return status;
    *out = 0.0f;
    for (base = 0; base < n; base += 256) {
        const uint8_t *block = blocks + (base / 256) * 66;
        int32_t sum = 0;
        for (i = 0; i < 64; i++) {
            uint8_t packed = block[i];
            for (j = 0; j < 4; j++) {
                sum += ((int32_t)(packed & 3) - 1) * activations[base + i * 4 + j];
                packed >>= 2;
            }
        }
        *out += tq_scale(block, 64) * (float)sum;
    }
    return FOX_OK;
}

#if defined(FOX_X86_SSSE3)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("ssse3")))
#endif
static void tq2_dot_ssse3_one(const int8_t *x, const uint8_t *block, float *out)
{
    const __m128i mask = _mm_set1_epi8(3);
    const __m128i lut = _mm_setr_epi8(-1, 0, 1, 2, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0);
    int8_t decoded[256];
    int8_t lane[16];
    int32_t sum = 0;
    size_t i, j;
    for (i = 0; i < 64; i += 16) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(block + i));
        __m128i codes0 = _mm_and_si128(packed, mask);
        __m128i codes1 = _mm_and_si128(_mm_srli_epi16(packed, 2), mask);
        __m128i codes2 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        __m128i codes3 = _mm_and_si128(_mm_srli_epi16(packed, 6), mask);
        _mm_storeu_si128((__m128i *)lane, _mm_shuffle_epi8(lut, codes0));
        for (j = 0; j < 16; j++) decoded[(i + j) * 4] = lane[j];
        _mm_storeu_si128((__m128i *)lane, _mm_shuffle_epi8(lut, codes1));
        for (j = 0; j < 16; j++) decoded[(i + j) * 4 + 1] = lane[j];
        _mm_storeu_si128((__m128i *)lane, _mm_shuffle_epi8(lut, codes2));
        for (j = 0; j < 16; j++) decoded[(i + j) * 4 + 2] = lane[j];
        _mm_storeu_si128((__m128i *)lane, _mm_shuffle_epi8(lut, codes3));
        for (j = 0; j < 16; j++) decoded[(i + j) * 4 + 3] = lane[j];
    }
    for (i = 0; i < 256; i++) sum += (int32_t)decoded[i] * x[i];
    *out += tq_scale(block, 64) * (float)sum;
}
#endif

fox_status fox_tq2_dot_i8(const int8_t *activations, const void *block_data,
                          size_t n, float *out)
{
    fox_cpu_info cpu;
    fox_status status = validate_args(activations, block_data, n, 256, out);
    if (status != FOX_OK) return status;
    *out = 0.0f;
#if defined(FOX_X86_SSSE3)
    fox_probe_cpu(&cpu);
    if (cpu.features & FOX_CPU_SSSE3) {
        size_t base;
        const uint8_t *blocks = (const uint8_t *)block_data;
        for (base = 0; base < n; base += 256)
            tq2_dot_ssse3_one(activations + base, blocks + (base / 256) * 66, out);
        return FOX_OK;
    }
#else
    (void)cpu;
#endif
    return fox_tq2_dot_i8_scalar(activations, block_data, n, out);
}
