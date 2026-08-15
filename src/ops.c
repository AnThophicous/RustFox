#include "fox_internal.h"
#include "model_internal.h"

#include <math.h>
#include <string.h>

void fox_rmsnorm(float *out, const float *x, const float *weight,
                 size_t n, float eps)
{
    double sum = 0.0;
    float scale;
    size_t i;

    for (i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    scale = (float)(1.0 / sqrt(sum / (double)n + (double)eps));

    if (weight) {
        for (i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    } else {
        for (i = 0; i < n; i++) out[i] = x[i] * scale;
    }
}

void fox_softmax(float *x, size_t n)
{
    float max = x[0];
    float sum = 0.0f;
    size_t i;

    for (i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (i = 0; i < n; i++) x[i] *= inv;
    }
}

void fox_silu_mul(float *gate, const float *up, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void fox_add(float *dst, const float *src, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) dst[i] += src[i];
}

void fox_rope(float *vec, size_t n_heads, size_t head_dim, uint32_t pos,
              float freq_base)
{
    size_t h, i;
    size_t half = head_dim / 2;

    for (h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (i = 0; i < half; i++) {
            float exponent = -2.0f * (float)i / (float)head_dim;
            float theta = (float)pos * powf(freq_base, exponent);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = v[2 * i];
            float x1 = v[2 * i + 1];
            v[2 * i]     = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

float fox_dot_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
