#include "fox_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

#define GEMV_BLOCK     256
#define GEMV_TQ1_BYTES 54
#define GEMV_TQ2_BYTES 66

fox_status fox_quantize_activations_i8(const float *x, size_t n,
                                       int8_t *q, float *scale)
{
    float amax = 0.0f;
    float inv;
    size_t i;

    if (!x || !q || !scale || n == 0) return FOX_ERR_ARG;

    for (i = 0; i < n; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }

    if (!(amax > 0.0f)) {
        memset(q, 0, n);
        *scale = 0.0f;
        return FOX_OK;
    }

    inv = 127.0f / amax;
    for (i = 0; i < n; i++) {
        float v = x[i] * inv;
        int r = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
        if (r >  127) r =  127;
        if (r < -127) r = -127;
        q[i] = (int8_t)r;
    }

    *scale = amax / 127.0f;
    return FOX_OK;
}

typedef struct {
    const int8_t  *q;
    const uint8_t *weights;
    float         *out;
    size_t         n_cols;
    size_t         row_bytes;
    float          act_scale;
    int            ternary_one_bit;
} gemv_job;

static void gemv_rows(void *vjob, int worker, size_t begin, size_t end)
{
    gemv_job *job = (gemv_job *)vjob;
    size_t r;

    (void)worker;

    for (r = begin; r < end; r++) {
        const uint8_t *row = job->weights + r * job->row_bytes;
        float acc = 0.0f;

        if (job->ternary_one_bit)
            fox_tq1_dot_i8(job->q, row, job->n_cols, &acc);
        else
            fox_tq2_dot_i8(job->q, row, job->n_cols, &acc);

        job->out[r] = acc * job->act_scale;
    }
}

static fox_status gemv_i8(fox_threadpool *tp, const int8_t *q, float act_scale,
                          const void *weights, size_t n_rows, size_t n_cols,
                          float *out, int one_bit)
{
    gemv_job job;

    if (!q || !weights || !out) return FOX_ERR_ARG;
    if (n_rows == 0 || n_cols == 0) return FOX_ERR_ARG;
    if (n_cols % GEMV_BLOCK != 0)
        return fox_fail(FOX_ERR_ARG,
                        "gemv: %llu columns is not a multiple of the %d-element block",
                        (unsigned long long)n_cols, GEMV_BLOCK);

    job.q               = q;
    job.weights         = (const uint8_t *)weights;
    job.out             = out;
    job.n_cols          = n_cols;
    job.row_bytes       = (n_cols / GEMV_BLOCK) *
                          (size_t)(one_bit ? GEMV_TQ1_BYTES : GEMV_TQ2_BYTES);
    job.act_scale       = act_scale;
    job.ternary_one_bit = one_bit;

    return fox_parallel_for(tp, n_rows, gemv_rows, &job);
}

fox_status fox_gemv_tq1_i8(fox_threadpool *tp, const int8_t *q, float act_scale,
                           const void *weights, size_t n_rows, size_t n_cols,
                           float *out)
{
    return gemv_i8(tp, q, act_scale, weights, n_rows, n_cols, out, 1);
}

fox_status fox_gemv_tq2_i8(fox_threadpool *tp, const int8_t *q, float act_scale,
                           const void *weights, size_t n_rows, size_t n_cols,
                           float *out)
{
    return gemv_i8(tp, q, act_scale, weights, n_rows, n_cols, out, 0);
}

static fox_status gemv_f32(fox_threadpool *tp, const float *x, const void *weights,
                           size_t n_rows, size_t n_cols, float *out, int one_bit)
{
    int8_t *q;
    float scale = 0.0f;
    fox_status st;

    if (!x || !weights || !out) return FOX_ERR_ARG;
    if (n_rows == 0 || n_cols == 0) return FOX_ERR_ARG;

    q = (int8_t *)malloc(n_cols);
    if (!q) return fox_fail(FOX_ERR_NOMEM, "gemv: activation buffer");

    st = fox_quantize_activations_i8(x, n_cols, q, &scale);
    if (st == FOX_OK)
        st = gemv_i8(tp, q, scale, weights, n_rows, n_cols, out, one_bit);

    free(q);
    return st;
}

fox_status fox_gemv_tq1_f32(fox_threadpool *tp, const float *x, const void *weights,
                            size_t n_rows, size_t n_cols, float *out)
{
    return gemv_f32(tp, x, weights, n_rows, n_cols, out, 1);
}

fox_status fox_gemv_tq2_f32(fox_threadpool *tp, const float *x, const void *weights,
                            size_t n_rows, size_t n_cols, float *out)
{
    return gemv_f32(tp, x, weights, n_rows, n_cols, out, 0);
}
