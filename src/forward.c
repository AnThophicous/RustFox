#include "fox_internal.h"
#include "model_internal.h"
#include "platform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct fox_context {
    fox_model          *model;
    fox_context_config  cfg;
    fox_threadpool     *pool;

    uint32_t            pos;
    size_t              n_q_embd;
    size_t              n_kv_embd;
    size_t              gqa;
    size_t              wide;

    float              *kcache;
    float              *vcache;

    float              *x;
    float              *xb;
    float              *xb2;
    float              *q;
    float              *kbuf;
    float              *vbuf;
    float              *hb;
    float              *hb2;
    float              *att;
    float              *norm;
    float              *logits;
    float              *rope_cos;
    float              *rope_sin;

    double              last_seconds;
    uint64_t            reads_at_start;
};

void fox_context_config_default(fox_context_config *cfg, const fox_model *m)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_ctx     = m ? m->info.n_ctx_train : 2048;
    cfg->n_batch   = 32;
    cfg->n_threads = 0;
    if (cfg->n_ctx == 0) cfg->n_ctx = 2048;
}

static void *alloc_f32(size_t count)
{
    return calloc(count ? count : 1, sizeof(float));
}

fox_status fox_context_create(fox_model *m, const fox_context_config *cfg,
                              fox_context **out)
{
    fox_context *ctx;
    fox_context_config c;
    const fox_model_info *mi;
    size_t nb;

    if (!m || !out) return fox_fail(FOX_ERR_ARG, "context: null argument");
    *out = NULL;

    if (cfg) c = *cfg;
    else     fox_context_config_default(&c, m);
    if (c.n_ctx == 0)   c.n_ctx = m->info.n_ctx_train;
    if (c.n_ctx == 0)   c.n_ctx = 2048;
    if (c.n_batch == 0) c.n_batch = 32;
    if (c.n_batch > c.n_ctx) c.n_batch = c.n_ctx;

    mi = &m->info;

    ctx = (fox_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return fox_fail(FOX_ERR_NOMEM, "context: handle");

    ctx->model     = m;
    ctx->cfg       = c;
    ctx->n_q_embd  = (size_t)mi->n_head * mi->head_dim;
    ctx->n_kv_embd = (size_t)mi->n_head_kv * mi->head_dim;
    ctx->gqa       = mi->n_head / mi->n_head_kv;
    ctx->wide      = ctx->n_q_embd > mi->n_embd ? ctx->n_q_embd : mi->n_embd;

    ctx->pool = fox_threadpool_create(c.n_threads);
    if (!ctx->pool) { free(ctx); return fox_fail(FOX_ERR_NOMEM, "context: pool"); }

    nb = c.n_batch;

    ctx->kcache = (float *)alloc_f32((size_t)mi->n_layer * c.n_ctx * ctx->n_kv_embd);
    ctx->vcache = (float *)alloc_f32((size_t)mi->n_layer * c.n_ctx * ctx->n_kv_embd);
    ctx->x      = (float *)alloc_f32(nb * mi->n_embd);
    ctx->xb     = (float *)alloc_f32(nb * ctx->wide);
    ctx->xb2    = (float *)alloc_f32(nb * ctx->wide);
    ctx->q      = (float *)alloc_f32(nb * ctx->n_q_embd);
    ctx->kbuf   = (float *)alloc_f32(nb * ctx->n_kv_embd);
    ctx->vbuf   = (float *)alloc_f32(nb * ctx->n_kv_embd);
    ctx->hb     = (float *)alloc_f32(nb * mi->n_ff);
    ctx->hb2    = (float *)alloc_f32(nb * mi->n_ff);
    ctx->att    = (float *)alloc_f32((size_t)fox_threadpool_size(ctx->pool) * c.n_ctx);
    ctx->norm   = (float *)alloc_f32(ctx->wide);
    ctx->logits = (float *)alloc_f32(mi->n_vocab);
    ctx->rope_cos = (float *)alloc_f32((size_t)c.n_ctx * (mi->head_dim / 2));
    ctx->rope_sin = (float *)alloc_f32((size_t)c.n_ctx * (mi->head_dim / 2));

    if (ctx->rope_cos && ctx->rope_sin) {
        size_t half = mi->head_dim / 2;
        size_t p, i;
        for (p = 0; p < c.n_ctx; p++) {
            for (i = 0; i < half; i++) {
                double exponent = -2.0 * (double)i / (double)mi->head_dim;
                double theta = (double)p * pow((double)mi->rope_freq_base, exponent);
                ctx->rope_cos[p * half + i] = (float)cos(theta);
                ctx->rope_sin[p * half + i] = (float)sin(theta);
            }
        }
    }

    if (!ctx->kcache || !ctx->vcache || !ctx->x || !ctx->xb || !ctx->xb2 ||
        !ctx->q || !ctx->kbuf || !ctx->vbuf || !ctx->hb || !ctx->hb2 ||
        !ctx->att || !ctx->norm || !ctx->logits ||
        !ctx->rope_cos || !ctx->rope_sin) {
        fox_context_destroy(ctx);
        return fox_fail(FOX_ERR_NOMEM,
                        "context: cannot allocate working memory for %u context "
                        "tokens; try a smaller --ctx", c.n_ctx);
    }

    *out = ctx;
    return FOX_OK;
}

void fox_context_destroy(fox_context *ctx)
{
    if (!ctx) return;
    free(ctx->kcache); free(ctx->vcache);
    free(ctx->x); free(ctx->xb); free(ctx->xb2);
    free(ctx->q); free(ctx->kbuf); free(ctx->vbuf);
    free(ctx->hb); free(ctx->hb2);
    free(ctx->att); free(ctx->norm); free(ctx->logits);
    free(ctx->rope_cos); free(ctx->rope_sin);
    if (ctx->pool) fox_threadpool_destroy(ctx->pool);
    free(ctx);
}

void fox_context_reset(fox_context *ctx)
{
    if (ctx) ctx->pos = 0;
}

uint32_t fox_context_position(const fox_context *ctx)
{
    return ctx ? ctx->pos : 0;
}

double fox_context_last_eval_seconds(const fox_context *ctx)
{
    return ctx ? ctx->last_seconds : 0.0;
}

uint64_t fox_context_weight_reads(const fox_context *ctx)
{
    return ctx ? fox_weights_reads(ctx->model->weights) : 0;
}

static fox_status norm_weights(fox_context *ctx, size_t tensor_index,
                               size_t n, const float **out)
{
    fox_weights_lease lease;
    fox_gguf_tensor t;
    fox_status st;

    st = fox_gguf_tensor_at(ctx->model->gguf, tensor_index, &t);
    if (st != FOX_OK) return st;

    st = fox_weights_acquire(ctx->model->weights, tensor_index, &lease);
    if (st != FOX_OK) return st;

    st = fox_dequant_row(t.type, lease.data, n, ctx->norm);
    fox_weights_release(ctx->model->weights, &lease);
    if (st != FOX_OK) return st;

    *out = ctx->norm;
    return FOX_OK;
}

static fox_status project(fox_context *ctx, size_t tensor_index, size_t tokens,
                          const float *in, size_t in_stride,
                          float *outbuf, size_t out_stride)
{
    fox_weights_lease lease;
    fox_gguf_tensor t;
    fox_status st;
    size_t i;

    st = fox_gguf_tensor_at(ctx->model->gguf, tensor_index, &t);
    if (st != FOX_OK) return st;
    if (t.n_dims < 2)
        return fox_fail(FOX_ERR_FORMAT, "forward: '%s' is not a matrix", t.name);

    st = fox_weights_acquire(ctx->model->weights, tensor_index, &lease);
    if (st != FOX_OK) return st;

    for (i = 0; i < tokens; i++) {
        st = fox_gemv(ctx->pool, t.type, in + i * in_stride, lease.data,
                      (size_t)t.ne[1], (size_t)t.ne[0], outbuf + i * out_stride);
        if (st != FOX_OK) break;
    }

    fox_weights_release(ctx->model->weights, &lease);
    return st;
}

typedef struct {
    fox_context *ctx;
    size_t       layer;
    size_t       token;
    uint32_t     seq_pos;
} attn_job;

static void attention_heads(void *vjob, int worker, size_t begin, size_t end)
{
    attn_job *job = (attn_job *)vjob;
    fox_context *ctx = job->ctx;
    const fox_model_info *mi = &ctx->model->info;
    size_t head_dim = mi->head_dim;
    size_t n_ctx = ctx->cfg.n_ctx;
    size_t layer_base = job->layer * n_ctx * ctx->n_kv_embd;
    float *att = ctx->att + (size_t)worker * n_ctx;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t upto = (size_t)job->seq_pos + 1;
    size_t h;

    for (h = begin; h < end; h++) {
        const float *qh = ctx->q + job->token * ctx->n_q_embd + h * head_dim;
        float *dst = ctx->xb + job->token * ctx->wide + h * head_dim;
        size_t kvh = h / ctx->gqa;
        size_t j, d;

        for (j = 0; j < upto; j++) {
            const float *kj = ctx->kcache + layer_base + j * ctx->n_kv_embd +
                              kvh * head_dim;
            att[j] = fox_dot_f32(qh, kj, head_dim) * scale;
        }

        fox_softmax(att, upto);

        for (d = 0; d < head_dim; d++) dst[d] = 0.0f;
        for (j = 0; j < upto; j++) {
            const float *vj = ctx->vcache + layer_base + j * ctx->n_kv_embd +
                              kvh * head_dim;
            float a = att[j];
            for (d = 0; d < head_dim; d++) dst[d] += a * vj[d];
        }
    }
}

static fox_status embed(fox_context *ctx, const uint32_t *tokens, size_t n)
{
    fox_model *m = ctx->model;
    fox_weights_lease lease;
    fox_gguf_tensor t;
    fox_status st;
    size_t row_bytes, i;

    st = fox_gguf_tensor_at(m->gguf, m->token_embd, &t);
    if (st != FOX_OK) return st;

    row_bytes = fox_row_bytes(t.type, m->info.n_embd);
    if (row_bytes == 0)
        return fox_fail(FOX_ERR_FORMAT, "forward: token_embd row is not aligned "
                                        "to a whole number of blocks");

    st = fox_weights_acquire(m->weights, m->token_embd, &lease);
    if (st != FOX_OK) return st;

    for (i = 0; i < n; i++) {
        if (tokens[i] >= m->info.n_vocab) {
            st = fox_fail(FOX_ERR_ARG, "forward: token %u is outside the %u "
                                       "entry vocabulary",
                          tokens[i], m->info.n_vocab);
            break;
        }
        st = fox_dequant_row(t.type,
                             (const uint8_t *)lease.data + (size_t)tokens[i] * row_bytes,
                             m->info.n_embd, ctx->x + i * m->info.n_embd);
        if (st != FOX_OK) break;
    }

    fox_weights_release(m->weights, &lease);
    return st;
}

static fox_status forward_batch(fox_context *ctx, const uint32_t *tokens,
                                size_t n, int want_logits)
{
    fox_model *m = ctx->model;
    const fox_model_info *mi = &m->info;
    size_t n_embd = mi->n_embd;
    size_t n_ctx = ctx->cfg.n_ctx;
    const float *w;
    fox_status st;
    uint32_t l;
    size_t i;
    size_t prefetched = FOX_TENSOR_ABSENT;

    if ((size_t)ctx->pos + n > n_ctx)
        return fox_fail(FOX_ERR_ARG,
                        "forward: %llu tokens past position %u would overrun the "
                        "%llu token context",
                        (unsigned long long)n, ctx->pos,
                        (unsigned long long)n_ctx);

    st = embed(ctx, tokens, n);
    if (st != FOX_OK) return st;

    for (l = 0; l < mi->n_layer; l++) {
        const fox_layer_tensors *lt = &m->layers[l];
        size_t layer_base = (size_t)l * n_ctx * ctx->n_kv_embd;

        if (prefetched == lt->attn_norm) {
            (void)fox_weights_prefetch_wait(m->weights, prefetched);
            prefetched = FOX_TENSOR_ABSENT;
        }

        st = norm_weights(ctx, lt->attn_norm, n_embd, &w);
        if (st != FOX_OK) return st;
        for (i = 0; i < n; i++)
            fox_rmsnorm(ctx->xb + i * ctx->wide, ctx->x + i * n_embd, w,
                        n_embd, mi->rms_eps);

        if (l + 1 < mi->n_layer) {
            size_t next_norm = m->layers[l + 1].attn_norm;
            if (fox_weights_prefetch_async(m->weights, next_norm) == FOX_OK)
                prefetched = next_norm;
        }

        st = project(ctx, lt->attn_q, n, ctx->xb, ctx->wide, ctx->q, ctx->n_q_embd);
        if (st != FOX_OK) return st;
        st = project(ctx, lt->attn_k, n, ctx->xb, ctx->wide, ctx->kbuf, ctx->n_kv_embd);
        if (st != FOX_OK) return st;
        st = project(ctx, lt->attn_v, n, ctx->xb, ctx->wide, ctx->vbuf, ctx->n_kv_embd);
        if (st != FOX_OK) return st;

        for (i = 0; i < n; i++) {
            uint32_t p = ctx->pos + (uint32_t)i;
            float *kdst = ctx->kcache + layer_base + (size_t)p * ctx->n_kv_embd;
            float *vdst = ctx->vcache + layer_base + (size_t)p * ctx->n_kv_embd;

            const float *rc = ctx->rope_cos + (size_t)p * (mi->head_dim / 2);
            const float *rs = ctx->rope_sin + (size_t)p * (mi->head_dim / 2);

            fox_rope(ctx->q + i * ctx->n_q_embd, mi->n_head, mi->head_dim, rc, rs);
            fox_rope(ctx->kbuf + i * ctx->n_kv_embd, mi->n_head_kv, mi->head_dim,
                     rc, rs);

            memcpy(kdst, ctx->kbuf + i * ctx->n_kv_embd,
                   ctx->n_kv_embd * sizeof(float));
            memcpy(vdst, ctx->vbuf + i * ctx->n_kv_embd,
                   ctx->n_kv_embd * sizeof(float));
        }

        for (i = 0; i < n; i++) {
            attn_job job;
            job.ctx     = ctx;
            job.layer   = l;
            job.token   = i;
            job.seq_pos = ctx->pos + (uint32_t)i;
            st = fox_parallel_for(ctx->pool, mi->n_head, attention_heads, &job);
            if (st != FOX_OK) return st;
        }

        st = project(ctx, lt->attn_out, n, ctx->xb, ctx->wide, ctx->xb2, ctx->wide);
        if (st != FOX_OK) return st;
        for (i = 0; i < n; i++)
            fox_add(ctx->x + i * n_embd, ctx->xb2 + i * ctx->wide, n_embd);

        st = norm_weights(ctx, lt->ffn_norm, n_embd, &w);
        if (st != FOX_OK) return st;
        for (i = 0; i < n; i++)
            fox_rmsnorm(ctx->xb + i * ctx->wide, ctx->x + i * n_embd, w,
                        n_embd, mi->rms_eps);

        st = project(ctx, lt->ffn_gate, n, ctx->xb, ctx->wide, ctx->hb, mi->n_ff);
        if (st != FOX_OK) return st;
        st = project(ctx, lt->ffn_up, n, ctx->xb, ctx->wide, ctx->hb2, mi->n_ff);
        if (st != FOX_OK) return st;

        for (i = 0; i < n; i++)
            fox_silu_mul(ctx->hb + i * mi->n_ff, ctx->hb2 + i * mi->n_ff, mi->n_ff);

        st = project(ctx, lt->ffn_down, n, ctx->hb, mi->n_ff, ctx->xb2, ctx->wide);
        if (st != FOX_OK) return st;
        for (i = 0; i < n; i++)
            fox_add(ctx->x + i * n_embd, ctx->xb2 + i * ctx->wide, n_embd);
    }

    ctx->pos += (uint32_t)n;

    if (!want_logits) return FOX_OK;

    st = norm_weights(ctx, m->output_norm, n_embd, &w);
    if (st != FOX_OK) return st;
    fox_rmsnorm(ctx->xb, ctx->x + (n - 1) * n_embd, w, n_embd, mi->rms_eps);

    return project(ctx, m->output, 1, ctx->xb, ctx->wide, ctx->logits,
                   (size_t)mi->n_vocab);
}

fox_status fox_eval(fox_context *ctx, const uint32_t *tokens, size_t n_tokens,
                    float **logits_out)
{
    uint64_t t0;
    size_t done = 0;
    fox_status st = FOX_OK;

    if (!ctx || !tokens || n_tokens == 0)
        return fox_fail(FOX_ERR_ARG, "eval: nothing to evaluate");

    t0 = fox_now_ns();
    ctx->reads_at_start = fox_weights_reads(ctx->model->weights);

    while (done < n_tokens) {
        size_t chunk = n_tokens - done;
        int last;
        if (chunk > ctx->cfg.n_batch) chunk = ctx->cfg.n_batch;
        last = (done + chunk == n_tokens);

        st = forward_batch(ctx, tokens + done, chunk, last);
        if (st != FOX_OK) return st;
        done += chunk;
    }

    ctx->last_seconds = (double)(fox_now_ns() - t0) / 1e9;
    if (logits_out) *logits_out = ctx->logits;
    return st;
}
