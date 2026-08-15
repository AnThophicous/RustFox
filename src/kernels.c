#include "fox_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#if defined(FOX_ARCH_X86) && \
    (defined(__SSE2__) || defined(_M_X64) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#  include <emmintrin.h>
#  define FOX_KERNEL_SSE2 1
#endif

float fox_f16_to_f32(uint16_t h)
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

uint16_t fox_f32_to_f16(float f)
{
    union { uint32_t u; float f; } value;
    uint32_t sign, mant;
    int exp;

    value.f = f;
    sign = (value.u >> 16) & 0x8000u;
    exp  = (int)((value.u >> 23) & 0xFF) - 127 + 15;
    mant = value.u & 0x7FFFFFu;

    if (((value.u >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));

    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        {
            int shift = 14 - exp;
            uint32_t round = 1u << (shift - 1);
            uint32_t out = (mant + round) >> shift;
            return (uint16_t)(sign | out);
        }
    }

    {
        uint32_t out = (uint32_t)exp << 10;
        uint32_t rounded = (mant + 0x0FFFu + ((mant >> 13) & 1u)) >> 13;
        out += rounded;
        return (uint16_t)(sign | out);
    }
}

static uint16_t load_half(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float dot_f32(const uint8_t *w, const float *x, size_t n)
{
    const float *fw = (const float *)(const void *)w;
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) sum += fw[i] * x[i];
    return sum;
}

static float dot_f16(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) sum += fox_f16_to_f32(load_half(w + i * 2)) * x[i];
    return sum;
}

static float dot_q8_0(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 32;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 34;
        const int8_t *qs = (const int8_t *)(blk + 2);
        float d = fox_f16_to_f32(load_half(blk));
        float acc = 0.0f;
        size_t j;
        for (j = 0; j < 32; j++) acc += (float)qs[j] * x[b * 32 + j];
        sum += d * acc;
    }
    return sum;
}

static float dot_q4_0(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 32;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 18;
        const uint8_t *qs = blk + 2;
        float d = fox_f16_to_f32(load_half(blk));
        const float *xb = x + b * 32;
        float acc = 0.0f;
        size_t j;
        for (j = 0; j < 16; j++) {
            int lo = (int)(qs[j] & 0x0F) - 8;
            int hi = (int)(qs[j] >> 4) - 8;
            acc += (float)lo * xb[j] + (float)hi * xb[j + 16];
        }
        sum += d * acc;
    }
    return sum;
}

static float dot_q4_1(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 32;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 20;
        const uint8_t *qs = blk + 4;
        float d = fox_f16_to_f32(load_half(blk));
        float m = fox_f16_to_f32(load_half(blk + 2));
        const float *xb = x + b * 32;
        float acc = 0.0f;
        float msum = 0.0f;
        size_t j;
        for (j = 0; j < 16; j++) {
            int lo = (int)(qs[j] & 0x0F);
            int hi = (int)(qs[j] >> 4);
            acc  += (float)lo * xb[j] + (float)hi * xb[j + 16];
            msum += xb[j] + xb[j + 16];
        }
        sum += d * acc + m * msum;
    }
    return sum;
}

static float dot_q5_0(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 32;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 22;
        const uint8_t *qs = blk + 6;
        float d = fox_f16_to_f32(load_half(blk));
        const float *xb = x + b * 32;
        uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                      ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
        float acc = 0.0f;
        size_t j;
        for (j = 0; j < 16; j++) {
            uint8_t xh0 = (uint8_t)(((qh >> j) << 4) & 0x10);
            uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            int lo = (int)((qs[j] & 0x0F) | xh0) - 16;
            int hi = (int)((qs[j] >> 4)   | xh1) - 16;
            acc += (float)lo * xb[j] + (float)hi * xb[j + 16];
        }
        sum += d * acc;
    }
    return sum;
}

static float dot_q5_1(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 32;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 24;
        const uint8_t *qs = blk + 8;
        float d = fox_f16_to_f32(load_half(blk));
        float m = fox_f16_to_f32(load_half(blk + 2));
        const float *xb = x + b * 32;
        uint32_t qh = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8) |
                      ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
        float acc = 0.0f;
        float msum = 0.0f;
        size_t j;
        for (j = 0; j < 16; j++) {
            uint8_t xh0 = (uint8_t)(((qh >> j) << 4) & 0x10);
            uint8_t xh1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            int lo = (int)((qs[j] & 0x0F) | xh0);
            int hi = (int)((qs[j] >> 4)   | xh1);
            acc  += (float)lo * xb[j] + (float)hi * xb[j + 16];
            msum += xb[j] + xb[j + 16];
        }
        sum += d * acc + m * msum;
    }
    return sum;
}

static void dequant_block_q6k(const uint8_t *blk, float *out)
{
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t  *sc = (const int8_t *)(blk + 192);
    float d = fox_f16_to_f32(load_half(blk + 208));
    float *y = out;
    int n, l;

    for (n = 0; n < 256; n += 128) {
        for (l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;

            y[l]      = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = (uint8_t)(q[j]     & 63);
        *m = (uint8_t)(q[j + 4] & 63);
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)   | ((q[j]     >> 6) << 4));
    }
}

static void dequant_block_q4k(const uint8_t *blk, float *out)
{
    float d    = fox_f16_to_f32(load_half(blk));
    float dmin = fox_f16_to_f32(load_half(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *q = blk + 16;
    float *y = out;
    int j, l, is = 0;

    for (j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        float d1, m1, d2, m2;

        scale_min_k4(is + 0, scales, &sc, &m);
        d1 = d * (float)sc;
        m1 = dmin * (float)m;
        scale_min_k4(is + 1, scales, &sc, &m);
        d2 = d * (float)sc;
        m2 = dmin * (float)m;

        for (l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
        for (l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4)   - m2;

        q  += 32;
        is += 2;
    }
}

static void dequant_block_q5k(const uint8_t *blk, float *out)
{
    float d    = fox_f16_to_f32(load_half(blk));
    float dmin = fox_f16_to_f32(load_half(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *qh = blk + 16;
    const uint8_t *q = blk + 48;
    int group, l;

    for (group = 0; group < 8; group++) {
        uint8_t sc, m;
        const uint8_t *qg = q + (group & 3) * 32;

        scale_min_k4(group, scales, &sc, &m);
        for (l = 0; l < 32; l++) {
            int nib = group < 4 ? qg[l] & 0x0F : qg[l] >> 4;
            int high = (qh[l] >> group) & 1;
            int value = nib | (high << 4);
            out[group * 32 + l] = d * (float)sc * (float)value -
                                   dmin * (float)m;
        }
    }
}

static float dot_q4_k(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 144;
        float d    = fox_f16_to_f32(load_half(blk));
        float dmin = fox_f16_to_f32(load_half(blk + 2));
        const uint8_t *scales = blk + 4;
        const float *xb = x + b * 256;
        int p;

        for (p = 0; p < 4; p++) {
            const uint8_t *q = blk + 16 + (size_t)p * 32;
            const float *xlo = xb + (size_t)p * 64;
            const float *xhi = xlo + 32;
            float acc_lo = 0.0f, acc_hi = 0.0f;
            float sum_lo = 0.0f, sum_hi = 0.0f;
            uint8_t sc, m;
            float d1, m1, d2, m2;
            int l;

            for (l = 0; l < 32; l++) {
                float a = xlo[l];
                float c = xhi[l];
                acc_lo += (float)(q[l] & 0x0F) * a;
                acc_hi += (float)(q[l] >> 4) * c;
                sum_lo += a;
                sum_hi += c;
            }

            scale_min_k4(p * 2 + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = dmin * (float)m;
            scale_min_k4(p * 2 + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = dmin * (float)m;

            sum += d1 * acc_lo - m1 * sum_lo;
            sum += d2 * acc_hi - m2 * sum_hi;
        }
    }
    return sum;
}

static float dot_q5_k(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 176;
        float d    = fox_f16_to_f32(load_half(blk));
        float dmin = fox_f16_to_f32(load_half(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *q = blk + 48;
        const float *xb = x + b * 256;
        int group;

        for (group = 0; group < 8; group++) {
            const uint8_t *qg = q + (group & 3) * 32;
            const float *xg = xb + group * 32;
            float acc = 0.0f;
            float xsum = 0.0f;
            uint8_t sc, m;
            int l;

            for (l = 0; l < 32; l++) {
                int nib = group < 4 ? qg[l] & 0x0F : qg[l] >> 4;
                int high = (qh[l] >> group) & 1;
                acc += (float)(nib | (high << 4)) * xg[l];
                xsum += xg[l];
            }
            scale_min_k4(group, scales, &sc, &m);
            sum += d * (float)sc * acc - dmin * (float)m * xsum;
        }
    }
    return sum;
}

static float dot_q6_k(const uint8_t *w, const float *x, size_t n)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 210;
        float d = fox_f16_to_f32(load_half(blk + 208));
        const float *xb = x + b * 256;
        int half;

        for (half = 0; half < 2; half++) {
            const uint8_t *ql = blk + (size_t)half * 64;
            const uint8_t *qh = blk + 128 + (size_t)half * 32;
            const int8_t  *sc = (const int8_t *)(blk + 192) + (size_t)half * 8;
            const float *xh = xb + (size_t)half * 128;
            int sub;

            for (sub = 0; sub < 2; sub++) {
                int base = sub * 16;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                int l;

                for (l = base; l < base + 16; l++) {
                    int q0 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q1 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;

                    a0 += (float)q0 * xh[l];
                    a1 += (float)q1 * xh[l + 32];
                    a2 += (float)q2 * xh[l + 64];
                    a3 += (float)q3 * xh[l + 96];
                }

                sum += d * (float)sc[sub + 0] * a0;
                sum += d * (float)sc[sub + 2] * a1;
                sum += d * (float)sc[sub + 4] * a2;
                sum += d * (float)sc[sub + 6] * a3;
            }
        }
    }
    return sum;
}

#if defined(FOX_KERNEL_SSE2)

static __m128i kernel_sign_extend_low_i8(__m128i values)
{
    __m128i zero = _mm_setzero_si128();
    return _mm_unpacklo_epi8(values, _mm_cmpgt_epi8(zero, values));
}

static __m128i kernel_sign_extend_high_i8(__m128i values)
{
    __m128i zero = _mm_setzero_si128();
    return _mm_unpackhi_epi8(values, _mm_cmpgt_epi8(zero, values));
}

static int32_t kernel_dot_i8_vectors(__m128i a, __m128i b)
{
    __m128i ones = _mm_set1_epi16(1);
    __m128i low = _mm_mullo_epi16(kernel_sign_extend_low_i8(a),
                                  kernel_sign_extend_low_i8(b));
    __m128i high = _mm_mullo_epi16(kernel_sign_extend_high_i8(a),
                                   kernel_sign_extend_high_i8(b));
    __m128i acc = _mm_add_epi32(_mm_madd_epi16(low, ones),
                                _mm_madd_epi16(high, ones));

    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
}

static int32_t kernel_sum_i8(const int8_t *x)
{
    __m128i ones = _mm_set1_epi8(1);
    return kernel_dot_i8_vectors(ones, _mm_loadu_si128((const __m128i *)x)) +
           kernel_dot_i8_vectors(ones, _mm_loadu_si128((const __m128i *)(x + 16)));
}

static int32_t kernel_dot_u4_i8(const uint8_t *a, const int8_t *b, int high_nibble)
{
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i ones = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    int half;

    for (half = 0; half < 2; half++) {
        __m128i qa = _mm_loadu_si128((const __m128i *)(a + half * 16));
        __m128i xb = _mm_loadu_si128((const __m128i *)(b + half * 16));
        __m128i q = high_nibble
                  ? _mm_and_si128(_mm_srli_epi16(qa, 4), mask)
                  : _mm_and_si128(qa, mask);
        __m128i qlo = _mm_unpacklo_epi8(q, zero);
        __m128i qhi = _mm_unpackhi_epi8(q, zero);
        __m128i xlo = kernel_sign_extend_low_i8(xb);
        __m128i xhi = kernel_sign_extend_high_i8(xb);

        acc = _mm_add_epi32(acc,
                            _mm_madd_epi16(_mm_mullo_epi16(qlo, xlo), ones));
        acc = _mm_add_epi32(acc,
                            _mm_madd_epi16(_mm_mullo_epi16(qhi, xhi), ones));
    }

    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
}

static int32_t kernel_dot_u1_i8(const uint8_t *a, const int8_t *b, int bit)
{
    __m128i zero = _mm_setzero_si128();
    __m128i one = _mm_set1_epi8(1);
    __m128i shift = _mm_cvtsi32_si128(bit);
    __m128i ones = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    int half;

    for (half = 0; half < 2; half++) {
        __m128i bits = _mm_loadu_si128((const __m128i *)(a + half * 16));
        __m128i xb = _mm_loadu_si128((const __m128i *)(b + half * 16));
        __m128i q = _mm_and_si128(_mm_srl_epi16(bits, shift), one);
        __m128i qlo = _mm_unpacklo_epi8(q, zero);
        __m128i qhi = _mm_unpackhi_epi8(q, zero);
        __m128i xlo = kernel_sign_extend_low_i8(xb);
        __m128i xhi = kernel_sign_extend_high_i8(xb);

        acc = _mm_add_epi32(acc,
                            _mm_madd_epi16(_mm_mullo_epi16(qlo, xlo), ones));
        acc = _mm_add_epi32(acc,
                            _mm_madd_epi16(_mm_mullo_epi16(qhi, xhi), ones));
    }

    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return _mm_cvtsi128_si32(acc);
}

static int32_t kernel_q6_group_dot(const int8_t *x, const uint8_t *ql,
                                   const uint8_t *qh, const int8_t *sc,
                                   int sub)
{
    const int base = sub * 16;
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i mask2 = _mm_set1_epi8(0x03);
    const __m128i thirty_two = _mm_set1_epi8(32);
    __m128i low = _mm_loadu_si128((const __m128i *)(ql + base));
    __m128i high = _mm_loadu_si128((const __m128i *)(ql + base + 32));
    __m128i bits = _mm_loadu_si128((const __m128i *)(qh + base));
    __m128i q0, q1, q2, q3;
    int32_t sum = 0;

    q0 = _mm_or_si128(_mm_and_si128(low, mask4),
                      _mm_slli_epi16(_mm_and_si128(bits, mask2), 4));
    q1 = _mm_or_si128(_mm_and_si128(high, mask4),
                      _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(bits, 2), mask2), 4));
    q2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(low, 4), mask4),
                      _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(bits, 4), mask2), 4));
    q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(high, 4), mask4),
                      _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(bits, 6), mask2), 4));

    q0 = _mm_sub_epi8(q0, thirty_two);
    q1 = _mm_sub_epi8(q1, thirty_two);
    q2 = _mm_sub_epi8(q2, thirty_two);
    q3 = _mm_sub_epi8(q3, thirty_two);

    sum += (int32_t)sc[sub + 0] * kernel_dot_i8_vectors(
        q0, _mm_loadu_si128((const __m128i *)(x + base)));
    sum += (int32_t)sc[sub + 2] * kernel_dot_i8_vectors(
        q1, _mm_loadu_si128((const __m128i *)(x + base + 32)));
    sum += (int32_t)sc[sub + 4] * kernel_dot_i8_vectors(
        q2, _mm_loadu_si128((const __m128i *)(x + base + 64)));
    sum += (int32_t)sc[sub + 6] * kernel_dot_i8_vectors(
        q3, _mm_loadu_si128((const __m128i *)(x + base + 96)));

    return sum;
}

static float dot_q4_k_i8(const uint8_t *w, const int8_t *x,
                         size_t n, const float *ascale)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 144;
        float d = fox_f16_to_f32(load_half(blk));
        float dmin = fox_f16_to_f32(load_half(blk + 2));
        const uint8_t *scales = blk + 4;
        const int8_t *xb = x + b * 256;
        int p;

        for (p = 0; p < 4; p++) {
            const uint8_t *q = blk + 16 + (size_t)p * 32;
            const int8_t *xlo = xb + (size_t)p * 64;
            const int8_t *xhi = xlo + 32;
            uint8_t sc, m;

            scale_min_k4(p * 2 + 0, scales, &sc, &m);
            sum += ascale[b] * (d * (float)sc *
                                (float)kernel_dot_u4_i8(q, xlo, 0) -
                                dmin * (float)m * (float)kernel_sum_i8(xlo));
            scale_min_k4(p * 2 + 1, scales, &sc, &m);
            sum += ascale[b] * (d * (float)sc *
                                (float)kernel_dot_u4_i8(q, xhi, 1) -
                                dmin * (float)m * (float)kernel_sum_i8(xhi));
        }
    }
    return sum;
}

static float dot_q5_k_i8(const uint8_t *w, const int8_t *x,
                         size_t n, const float *ascale)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 176;
        float d = fox_f16_to_f32(load_half(blk));
        float dmin = fox_f16_to_f32(load_half(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *q = blk + 48;
        const int8_t *xb = x + b * 256;
        int group;

        for (group = 0; group < 8; group++) {
            const int8_t *xg = xb + group * 32;
            const uint8_t *qg = q + (group & 3) * 32;
            uint8_t sc, m;

            scale_min_k4(group, scales, &sc, &m);
            sum += ascale[b] *
                   (d * (float)sc * (float)(kernel_dot_u4_i8(
                       qg, xg, group >= 4) +
                       16 * kernel_dot_u1_i8(qh, xg, group)) -
                    dmin * (float)m * (float)kernel_sum_i8(xg));
        }
    }
    return sum;
}

static float dot_q6_k_i8(const uint8_t *w, const int8_t *x,
                         size_t n, const float *ascale)
{
    float sum = 0.0f;
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const uint8_t *blk = w + b * 210;
        float d = fox_f16_to_f32(load_half(blk + 208));
        const int8_t *xb = x + b * 256;
        int half;

        for (half = 0; half < 2; half++) {
            const uint8_t *ql = blk + (size_t)half * 64;
            const uint8_t *qh = blk + 128 + (size_t)half * 32;
            const int8_t *sc = (const int8_t *)(blk + 192) + (size_t)half * 8;
            int sub;

            for (sub = 0; sub < 2; sub++)
                sum += ascale[b] * d *
                       (float)kernel_q6_group_dot(xb + (size_t)half * 128,
                                                  ql, qh, sc, sub);
        }
    }
    return sum;
}

#endif

typedef float (*dot_fn)(const uint8_t *w, const float *x, size_t n);

static dot_fn dot_for(fox_ggml_type type)
{
    switch (type) {
    case FOX_GGML_F32:  return dot_f32;
    case FOX_GGML_F16:  return dot_f16;
    case FOX_GGML_Q8_0: return dot_q8_0;
    case FOX_GGML_Q4_0: return dot_q4_0;
    case FOX_GGML_Q4_1: return dot_q4_1;
    case FOX_GGML_Q5_0: return dot_q5_0;
    case FOX_GGML_Q5_1: return dot_q5_1;
    case FOX_GGML_Q5_K: return dot_q5_k;
    case FOX_GGML_Q6_K: return dot_q6_k;
    case FOX_GGML_Q4_K: return dot_q4_k;
    default: return NULL;
    }
}

int fox_gemv_supports(fox_ggml_type type)
{
    if (type == FOX_GGML_TQ1_0 || type == FOX_GGML_TQ2_0) return 1;
    return dot_for(type) != NULL;
}

size_t fox_row_bytes(fox_ggml_type type, size_t n_cols)
{
    uint32_t block = fox_ggml_type_block_size(type);
    uint32_t bytes = fox_ggml_type_block_bytes(type);

    if (block == 0 || bytes == 0) return 0;
    if (n_cols % block != 0) return 0;
    return (n_cols / block) * (size_t)bytes;
}

typedef struct {
    dot_fn         dot;
    const uint8_t *weights;
    const float   *x;
    float         *out;
    size_t         n_cols;
    size_t         row_bytes;
} kernel_job;

static void kernel_rows(void *vjob, int worker, size_t begin, size_t end)
{
    kernel_job *job = (kernel_job *)vjob;
    size_t r;

    (void)worker;
    for (r = begin; r < end; r++)
        job->out[r] = job->dot(job->weights + r * job->row_bytes,
                               job->x, job->n_cols);
}

#if defined(FOX_KERNEL_SSE2)

static void quantize_activations_k(const float *x, size_t n,
                                   int8_t *q, float *scales)
{
    size_t b, blocks = n / 256;

    for (b = 0; b < blocks; b++) {
        const float *xb = x + b * 256;
        int8_t *qb = q + b * 256;
        float amax = 0.0f;
        float inv;
        size_t i;

        for (i = 0; i < 256; i++) {
            float a = xb[i] < 0.0f ? -xb[i] : xb[i];
            if (a > amax) amax = a;
        }
        if (!(amax > 0.0f)) {
            memset(qb, 0, 256);
            scales[b] = 0.0f;
            continue;
        }

        inv = 127.0f / amax;
        for (i = 0; i < 256; i++) {
            float v = xb[i] * inv;
            int r = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (r >  127) r =  127;
            if (r < -127) r = -127;
            qb[i] = (int8_t)r;
        }
        scales[b] = amax / 127.0f;
    }
}

typedef float (*dot_i8_fn)(const uint8_t *w, const int8_t *x,
                           size_t n, const float *ascale);

typedef struct {
    dot_i8_fn      dot;
    const uint8_t *weights;
    const int8_t   *x;
    const float   *ascale;
    float          *out;
    size_t         n_cols;
    size_t         row_bytes;
} kernel_i8_job;

static void kernel_i8_rows(void *vjob, int worker, size_t begin, size_t end)
{
    kernel_i8_job *job = (kernel_i8_job *)vjob;
    size_t r;

    (void)worker;
    for (r = begin; r < end; r++)
        job->out[r] = job->dot(job->weights + r * job->row_bytes,
                               job->x, job->n_cols, job->ascale);
}

static fox_status gemv_qk_i8(fox_threadpool *tp, fox_ggml_type type,
                             const int8_t *x, const float *ascale,
                             const void *weights, size_t n_rows,
                             size_t n_cols, float *out)
{
    kernel_i8_job job;

    if (!x || !weights || !out || n_rows == 0 || n_cols == 0)
        return FOX_ERR_ARG;
    if (n_cols % 256 != 0)
        return fox_fail(FOX_ERR_ARG,
                        "gemv: %llu columns is not a whole %s block",
                        (unsigned long long)n_cols, fox_ggml_type_name(type));

    job.dot = type == FOX_GGML_Q4_K ? dot_q4_k_i8 :
              type == FOX_GGML_Q5_K ? dot_q5_k_i8 : dot_q6_k_i8;
    job.weights = (const uint8_t *)weights;
    job.x = x;
    job.ascale = ascale;
    job.out = out;
    job.n_cols = n_cols;
    job.row_bytes = fox_row_bytes(type, n_cols);
    if (job.row_bytes == 0) return FOX_ERR_ARG;

    return fox_parallel_for(tp, n_rows, kernel_i8_rows, &job);
}

#endif

fox_status fox_gemv(fox_threadpool *tp, fox_ggml_type type, const float *x,
                    const void *weights, size_t n_rows, size_t n_cols,
                    float *out)
{
    kernel_job job;
    size_t row_bytes;

    if (!x || !weights || !out) return FOX_ERR_ARG;
    if (n_rows == 0 || n_cols == 0) return FOX_ERR_ARG;

#if defined(FOX_KERNEL_SSE2)
    if (type == FOX_GGML_Q4_K || type == FOX_GGML_Q5_K ||
        type == FOX_GGML_Q6_K) {
        int8_t *qx;
        float *ascale;
        fox_status st;

        if (fox_row_bytes(type, n_cols) == 0)
            return fox_fail(FOX_ERR_ARG,
                            "gemv: %llu columns is not a whole number of %s blocks",
                            (unsigned long long)n_cols, fox_ggml_type_name(type));

        qx = (int8_t *)malloc(n_cols);
        ascale = (float *)malloc((n_cols / 256) * sizeof(float));
        if (!qx || !ascale) {
            free(qx);
            free(ascale);
            return fox_fail(FOX_ERR_NOMEM, "gemv: activation buffer");
        }
        quantize_activations_k(x, n_cols, qx, ascale);
        st = gemv_qk_i8(tp, type, qx, ascale, weights, n_rows, n_cols, out);
        free(qx);
        free(ascale);
        return st;
    }
#endif

    if (type == FOX_GGML_TQ1_0)
        return fox_gemv_tq1_f32(tp, x, weights, n_rows, n_cols, out);
    if (type == FOX_GGML_TQ2_0)
        return fox_gemv_tq2_f32(tp, x, weights, n_rows, n_cols, out);

    job.dot = dot_for(type);
    if (!job.dot)
        return fox_fail(FOX_ERR_UNSUPPORTED,
                        "gemv: no kernel for %s yet; this build handles F32, "
                        "F16, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, "
                        "Q8_0, TQ1_0 and TQ2_0",
                        fox_ggml_type_name(type));

    row_bytes = fox_row_bytes(type, n_cols);
    if (row_bytes == 0)
        return fox_fail(FOX_ERR_ARG,
                        "gemv: %llu columns is not a whole number of %s blocks",
                        (unsigned long long)n_cols, fox_ggml_type_name(type));

    job.weights   = (const uint8_t *)weights;
    job.x         = x;
    job.out       = out;
    job.n_cols    = n_cols;
    job.row_bytes = row_bytes;

    return fox_parallel_for(tp, n_rows, kernel_rows, &job);
}

fox_status fox_dequant_row(fox_ggml_type type, const void *row, size_t n,
                           float *out)
{
    const uint8_t *w = (const uint8_t *)row;
    size_t i, b, blocks;

    if (!row || !out || n == 0) return FOX_ERR_ARG;
    if (fox_row_bytes(type, n) == 0)
        return fox_fail(FOX_ERR_ARG,
                        "dequant: %llu elements is not a whole number of %s blocks",
                        (unsigned long long)n, fox_ggml_type_name(type));

    switch (type) {
    case FOX_GGML_F32:
        memcpy(out, w, n * sizeof(float));
        return FOX_OK;

    case FOX_GGML_F16:
        for (i = 0; i < n; i++) out[i] = fox_f16_to_f32(load_half(w + i * 2));
        return FOX_OK;

    case FOX_GGML_Q8_0:
        blocks = n / 32;
        for (b = 0; b < blocks; b++) {
            const uint8_t *blk = w + b * 34;
            const int8_t *qs = (const int8_t *)(blk + 2);
            float d = fox_f16_to_f32(load_half(blk));
            for (i = 0; i < 32; i++) out[b * 32 + i] = (float)qs[i] * d;
        }
        return FOX_OK;

    case FOX_GGML_Q4_0:
        blocks = n / 32;
        for (b = 0; b < blocks; b++) {
            const uint8_t *blk = w + b * 18;
            const uint8_t *qs = blk + 2;
            float d = fox_f16_to_f32(load_half(blk));
            for (i = 0; i < 16; i++) {
                out[b * 32 + i]      = (float)((int)(qs[i] & 0x0F) - 8) * d;
                out[b * 32 + i + 16] = (float)((int)(qs[i] >> 4)   - 8) * d;
            }
        }
        return FOX_OK;

    case FOX_GGML_Q6_K:
        blocks = n / 256;
        for (b = 0; b < blocks; b++)
            dequant_block_q6k(w + b * 210, out + b * 256);
        return FOX_OK;

    case FOX_GGML_Q5_K:
        blocks = n / 256;
        for (b = 0; b < blocks; b++)
            dequant_block_q5k(w + b * 176, out + b * 256);
        return FOX_OK;

    case FOX_GGML_Q4_K:
        blocks = n / 256;
        for (b = 0; b < blocks; b++)
            dequant_block_q4k(w + b * 144, out + b * 256);
        return FOX_OK;

    default:
        return fox_fail(FOX_ERR_UNSUPPORTED,
                        "dequant: %s is not implemented yet",
                        fox_ggml_type_name(type));
    }
}
