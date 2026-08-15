#include "fox_internal.h"
#include "platform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FOX_TOPK_MAX 512

struct fox_sampler {
    fox_sampler_config cfg;
    size_t   n_vocab;
    uint64_t rng;
    float   *work;
    uint32_t *history;
    size_t   hist_cap;
    size_t   hist_len;
    size_t   hist_head;
};

void fox_sampler_config_default(fox_sampler_config *cfg)
{
    if (!cfg) return;
    cfg->temperature    = 0.8f;
    cfg->top_k          = 40;
    cfg->top_p          = 0.95f;
    cfg->repeat_penalty = 1.1f;
    cfg->repeat_last_n  = 64;
    cfg->seed           = 0;
}

static uint64_t rng_next(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static float rng_float(uint64_t *s)
{
    return (float)((rng_next(s) >> 40) / 16777216.0);
}

fox_sampler *fox_sampler_create(const fox_sampler_config *cfg, size_t n_vocab)
{
    fox_sampler *s;

    if (n_vocab == 0) { fox_fail(FOX_ERR_ARG, "sampler: empty vocabulary"); return NULL; }

    s = (fox_sampler *)calloc(1, sizeof(*s));
    if (!s) { fox_fail(FOX_ERR_NOMEM, "sampler: handle"); return NULL; }

    if (cfg) s->cfg = *cfg;
    else     fox_sampler_config_default(&s->cfg);

    if (s->cfg.top_k < 0) s->cfg.top_k = 0;
    if (s->cfg.top_k > FOX_TOPK_MAX) s->cfg.top_k = FOX_TOPK_MAX;
    if (s->cfg.top_p <= 0.0f || s->cfg.top_p > 1.0f) s->cfg.top_p = 1.0f;
    if (s->cfg.repeat_penalty < 1.0f) s->cfg.repeat_penalty = 1.0f;

    s->n_vocab  = n_vocab;
    s->rng      = s->cfg.seed ? s->cfg.seed : 0x9E3779B97F4A7C15ull;
    s->hist_cap = s->cfg.repeat_last_n ? s->cfg.repeat_last_n : 1;

    s->work    = (float *)calloc(n_vocab, sizeof(float));
    s->history = (uint32_t *)calloc(s->hist_cap, sizeof(uint32_t));

    if (!s->work || !s->history) {
        fox_sampler_destroy(s);
        fox_fail(FOX_ERR_NOMEM, "sampler: buffers");
        return NULL;
    }
    return s;
}

void fox_sampler_destroy(fox_sampler *s)
{
    if (!s) return;
    free(s->work);
    free(s->history);
    free(s);
}

void fox_sampler_accept(fox_sampler *s, uint32_t token)
{
    if (!s || s->hist_cap == 0) return;
    s->history[s->hist_head] = token;
    s->hist_head = (s->hist_head + 1) % s->hist_cap;
    if (s->hist_len < s->hist_cap) s->hist_len++;
}

static void apply_repeat_penalty(fox_sampler *s, float *logits)
{
    size_t i;

    if (s->cfg.repeat_penalty <= 1.0f) return;

    for (i = 0; i < s->hist_len; i++) {
        uint32_t tok = s->history[i];
        if (tok >= s->n_vocab) continue;
        if (logits[tok] > 0.0f) logits[tok] /= s->cfg.repeat_penalty;
        else                    logits[tok] *= s->cfg.repeat_penalty;
    }
}

typedef struct { float logit; uint32_t id; } cand;

static size_t select_top_k(const float *logits, size_t n, int k, cand *out)
{
    size_t filled = 0;
    size_t i, j;

    for (i = 0; i < n; i++) {
        float v = logits[i];

        if (filled < (size_t)k) {
            j = filled++;
        } else {
            if (v <= out[filled - 1].logit) continue;
            j = filled - 1;
        }
        while (j > 0 && out[j - 1].logit < v) {
            out[j] = out[j - 1];
            j--;
        }
        out[j].logit = v;
        out[j].id    = (uint32_t)i;
    }
    return filled;
}

fox_status fox_sampler_pick(fox_sampler *s, const float *logits, uint32_t *out)
{
    cand top[FOX_TOPK_MAX];
    size_t n_top, i;
    float max, sum, cum, r;
    int k;

    if (!s || !logits || !out) return FOX_ERR_ARG;

    memcpy(s->work, logits, s->n_vocab * sizeof(float));
    apply_repeat_penalty(s, s->work);

    if (s->cfg.temperature <= 0.0f) {
        size_t best = 0;
        for (i = 1; i < s->n_vocab; i++)
            if (s->work[i] > s->work[best]) best = i;
        *out = (uint32_t)best;
        return FOX_OK;
    }

    k = s->cfg.top_k;
    if (k <= 0 || (size_t)k > s->n_vocab) k = (int)(s->n_vocab < FOX_TOPK_MAX
                                                    ? s->n_vocab : FOX_TOPK_MAX);

    n_top = select_top_k(s->work, s->n_vocab, k, top);
    if (n_top == 0) return fox_fail(FOX_ERR_INTERNAL, "sampler: no candidates");

    max = top[0].logit;
    sum = 0.0f;
    for (i = 0; i < n_top; i++) {
        top[i].logit = expf((top[i].logit - max) / s->cfg.temperature);
        sum += top[i].logit;
    }
    if (!(sum > 0.0f)) {
        *out = top[0].id;
        return FOX_OK;
    }
    for (i = 0; i < n_top; i++) top[i].logit /= sum;

    if (s->cfg.top_p < 1.0f) {
        cum = 0.0f;
        for (i = 0; i < n_top; i++) {
            cum += top[i].logit;
            if (cum >= s->cfg.top_p) { n_top = i + 1; break; }
        }
        sum = 0.0f;
        for (i = 0; i < n_top; i++) sum += top[i].logit;
        if (sum > 0.0f)
            for (i = 0; i < n_top; i++) top[i].logit /= sum;
    }

    r = rng_float(&s->rng);
    cum = 0.0f;
    for (i = 0; i < n_top; i++) {
        cum += top[i].logit;
        if (r <= cum) { *out = top[i].id; return FOX_OK; }
    }

    *out = top[n_top - 1].id;
    return FOX_OK;
}
