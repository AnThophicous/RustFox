#include "fox_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

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

fox_status fox_gemv(fox_threadpool *tp, fox_ggml_type type, const float *x,
                    const void *weights, size_t n_rows, size_t n_cols,
                    float *out)
{
    kernel_job job;
    size_t row_bytes;

    if (!x || !weights || !out) return FOX_ERR_ARG;
    if (n_rows == 0 || n_cols == 0) return FOX_ERR_ARG;

    if (type == FOX_GGML_TQ1_0)
        return fox_gemv_tq1_f32(tp, x, weights, n_rows, n_cols, out);
    if (type == FOX_GGML_TQ2_0)
        return fox_gemv_tq2_f32(tp, x, weights, n_rows, n_cols, out);

    job.dot = dot_for(type);
    if (!job.dot)
        return fox_fail(FOX_ERR_UNSUPPORTED,
                        "gemv: no kernel for %s yet; this build handles F32, "
                        "F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ1_0 and TQ2_0",
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

    default:
        return fox_fail(FOX_ERR_UNSUPPORTED,
                        "dequant: %s is not implemented yet",
                        fox_ggml_type_name(type));
    }
}
