#include "fox/fox.h"
#include "check.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYER 2
#define N_EMBD  8
#define N_HEAD  2
#define N_KV    1
#define N_FF    16
#define N_VOCAB 16
#define HEAD_D  (N_EMBD / N_HEAD)
#define N_Q     (N_HEAD * HEAD_D)
#define N_KVE   (N_KV * HEAD_D)
#define ALIGN   32

typedef struct {
    char     name[64];
    uint64_t ne0;
    uint64_t ne1;
    uint64_t offset;
    uint64_t bytes;
} tdesc;

static tdesc g_t[64];
static int   g_n;

static void add_tensor(const char *name, uint64_t ne0, uint64_t ne1)
{
    tdesc *t = &g_t[g_n];
    uint64_t elems = ne1 ? ne0 * ne1 : ne0;
    uint64_t prev_end = 0;

    snprintf(t->name, sizeof(t->name), "%s", name);
    t->ne0 = ne0;
    t->ne1 = ne1;
    t->bytes = elems * 4;

    if (g_n > 0) {
        tdesc *p = &g_t[g_n - 1];
        prev_end = p->offset + p->bytes;
    }
    t->offset = (prev_end + ALIGN - 1) & ~((uint64_t)ALIGN - 1);
    g_n++;
}

static void put32(FILE *f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
}

static void put64(FILE *f, uint64_t v)
{
    int i;
    for (i = 0; i < 8; i++) fputc((int)((v >> (8 * i)) & 0xFF), f);
}

static void put_str(FILE *f, const char *s)
{
    size_t n = strlen(s);
    put64(f, n);
    fwrite(s, 1, n, f);
}

static void put_f32(FILE *f, float v)
{
    union { float f; uint32_t u; } c;
    c.f = v;
    put32(f, c.u);
}

static void meta_u32(FILE *f, const char *key, uint32_t v)
{
    put_str(f, key);
    put32(f, FOX_GGUF_UINT32);
    put32(f, v);
}

static void meta_f32(FILE *f, const char *key, float v)
{
    put_str(f, key);
    put32(f, FOX_GGUF_FLOAT32);
    put_f32(f, v);
}

static float weight_value(int tensor, uint64_t i)
{
    int s = (int)((tensor * 31u + i * 17u + 7u) % 23u) - 11;
    return (float)s * 0.03f;
}

static void build_tensor_list(void)
{
    char name[64];
    int l;

    g_n = 0;
    add_tensor("token_embd.weight", N_EMBD, N_VOCAB);
    add_tensor("output_norm.weight", N_EMBD, 0);
    add_tensor("output.weight", N_EMBD, N_VOCAB);

    for (l = 0; l < N_LAYER; l++) {
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        add_tensor(name, N_EMBD, 0);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        add_tensor(name, N_EMBD, N_Q);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        add_tensor(name, N_EMBD, N_KVE);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        add_tensor(name, N_EMBD, N_KVE);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        add_tensor(name, N_Q, N_EMBD);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        add_tensor(name, N_EMBD, 0);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        add_tensor(name, N_EMBD, N_FF);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        add_tensor(name, N_EMBD, N_FF);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        add_tensor(name, N_FF, N_EMBD);
    }
}

static void write_model(const char *path)
{
    FILE *f = fopen(path, "wb");
    long at;
    int i;
    uint64_t written;

    build_tensor_list();

    put32(f, 0x46554747u);
    put32(f, 3);
    put64(f, (uint64_t)g_n);
    put64(f, 10);

    put_str(f, "general.alignment");
    put32(f, FOX_GGUF_UINT32);
    put32(f, ALIGN);

    put_str(f, "general.architecture");
    put32(f, FOX_GGUF_STRING);
    put_str(f, "llama");

    meta_u32(f, "llama.block_count", N_LAYER);
    meta_u32(f, "llama.embedding_length", N_EMBD);
    meta_u32(f, "llama.attention.head_count", N_HEAD);
    meta_u32(f, "llama.attention.head_count_kv", N_KV);
    meta_u32(f, "llama.feed_forward_length", N_FF);
    meta_u32(f, "llama.context_length", 64);
    meta_f32(f, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    meta_f32(f, "llama.rope.freq_base", 10000.0f);

    for (i = 0; i < g_n; i++) {
        put_str(f, g_t[i].name);
        if (g_t[i].ne1) {
            put32(f, 2);
            put64(f, g_t[i].ne0);
            put64(f, g_t[i].ne1);
        } else {
            put32(f, 1);
            put64(f, g_t[i].ne0);
        }
        put32(f, FOX_GGML_F32);
        put64(f, g_t[i].offset);
    }

    at = ftell(f);
    while (at % ALIGN) { fputc(0, f); at++; }

    written = 0;
    for (i = 0; i < g_n; i++) {
        uint64_t j;
        while (written < g_t[i].offset) { fputc(0, f); written++; }
        for (j = 0; j < g_t[i].bytes / 4; j++) put_f32(f, weight_value(i, j));
        written += g_t[i].bytes;
    }

    fclose(f);
}

static int finite_vector(const float *v, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (v[i] != v[i]) return 0;
        if (v[i] > 1e30f || v[i] < -1e30f) return 0;
    }
    return 1;
}

int main(void)
{
    const char *path = "rustfox-forward.gguf";
    fox_model *m = NULL;
    fox_context *ctx = NULL;
    fox_context_config ccfg;
    const fox_model_info *mi;
    uint32_t prompt[5] = { 1, 4, 9, 2, 7 };
    float batched[N_VOCAB], stepped[N_VOCAB], streamed[N_VOCAB];
    float *logits = NULL;
    fox_status st;
    size_t i;
    int ok;

    write_model(path);

    st = fox_model_open(path, FOX_WEIGHTS_RESIDENT, 0, &m);
    CHECK(st == FOX_OK && m != NULL, "a synthetic llama model loads");
    if (st != FOX_OK) { remove(path); CHECK_DONE(); }

    mi = fox_model_get_info(m);
    CHECK(mi->n_layer == N_LAYER && mi->n_embd == N_EMBD && mi->n_head == N_HEAD,
          "architecture metadata is read from the GGUF");
    CHECK(mi->n_head_kv == N_KV && mi->n_ff == N_FF,
          "grouped query attention and feed forward width are read");
    CHECK(mi->head_dim == HEAD_D, "head dimension is derived when not stated");
    CHECK(mi->n_vocab == N_VOCAB, "vocabulary size comes from the embedding rows");
    CHECK(mi->rms_eps > 0.0f && mi->rope_freq_base > 0.0f,
          "rms epsilon and rope base are read as floats");
    CHECK(mi->tied_output == 0, "a separate output head is detected");

    fox_context_config_default(&ccfg, m);
    ccfg.n_ctx = 32;
    ccfg.n_threads = 2;
    ccfg.n_batch = 8;

    st = fox_context_create(m, &ccfg, &ctx);
    CHECK(st == FOX_OK && ctx != NULL, "a context is created");
    if (st != FOX_OK) { fox_model_close(m); remove(path); CHECK_DONE(); }

    st = fox_eval(ctx, prompt, 5, &logits);
    CHECK(st == FOX_OK && logits != NULL, "the forward pass runs on a 5 token prompt");
    if (st != FOX_OK) {
        fprintf(stderr, "# %s\n", fox_last_error());
        fox_context_destroy(ctx);
        fox_model_close(m);
        remove(path);
        CHECK_DONE();
    }

    memcpy(batched, logits, sizeof(batched));
    CHECK(finite_vector(batched, N_VOCAB),
          "every logit is finite, so nothing blew up in rmsnorm or softmax");
    CHECK(fox_context_position(ctx) == 5, "the context advanced by five tokens");

    ok = 0;
    for (i = 1; i < N_VOCAB; i++) if (batched[i] != batched[0]) ok = 1;
    CHECK(ok, "the logits are not all the same value, so the weights actually "
              "reached the output head");

    fox_context_reset(ctx);
    for (i = 0; i < 5; i++) {
        st = fox_eval(ctx, &prompt[i], 1, &logits);
        if (st != FOX_OK) break;
    }
    CHECK(st == FOX_OK, "the same prompt runs one token at a time");
    memcpy(stepped, logits, sizeof(stepped));

    ok = 1;
    for (i = 0; i < N_VOCAB; i++) if (batched[i] != stepped[i]) ok = 0;
    CHECK(ok,
          "a five token batch produces bit-identical logits to five single "
          "token evals. This is what makes the weight-stationary loop safe: "
          "the prompt costs one pass over the weights instead of five, and the "
          "answer does not change");

    fox_context_destroy(ctx);
    fox_model_close(m);

    st = fox_model_open(path, FOX_WEIGHTS_STREAM, 4096, &m);
    CHECK(st == FOX_OK && m != NULL,
          "the same model opens in streaming mode with a budget far below its "
          "total size");
    if (st == FOX_OK) {
        st = fox_context_create(m, &ccfg, &ctx);
        CHECK(st == FOX_OK, "a streaming context is created");
        if (st == FOX_OK) {
            st = fox_eval(ctx, prompt, 5, &logits);
            CHECK(st == FOX_OK, "the forward pass runs with streamed weights");
            if (st == FOX_OK) {
                memcpy(streamed, logits, sizeof(streamed));
                ok = 1;
                for (i = 0; i < N_VOCAB; i++)
                    if (streamed[i] != batched[i]) ok = 0;
                CHECK(ok,
                      "streaming the weights off storage produces bit-identical "
                      "logits to holding them all in memory. This is the whole "
                      "bet of the project, checked end to end rather than at "
                      "the tensor boundary");
                CHECK(fox_weights_evictions(fox_model_weights(m)) > 0,
                      "the budget was tight enough that tensors really were "
                      "evicted and re-read during the pass");
            }
            fox_context_destroy(ctx);
        }
        fox_model_close(m);
    }

    {
        fox_sampler_config scfg;
        fox_sampler *s;
        uint32_t tok = 0;

        fox_sampler_config_default(&scfg);
        scfg.temperature = 0.0f;
        s = fox_sampler_create(&scfg, N_VOCAB);
        CHECK(s != NULL, "a sampler is created");
        CHECK(fox_sampler_pick(s, batched, &tok) == FOX_OK && tok < N_VOCAB,
              "greedy sampling returns a valid token");
        {
            size_t best = 0;
            for (i = 1; i < N_VOCAB; i++) if (batched[i] > batched[best]) best = i;
            CHECK(tok == (uint32_t)best, "greedy sampling picks the argmax");
        }
        fox_sampler_destroy(s);

        scfg.temperature = 0.8f;
        scfg.top_k = 5;
        scfg.seed = 12345;
        s = fox_sampler_create(&scfg, N_VOCAB);
        CHECK(s != NULL, "a stochastic sampler is created");
        ok = 1;
        for (i = 0; i < 200; i++) {
            if (fox_sampler_pick(s, batched, &tok) != FOX_OK || tok >= N_VOCAB) ok = 0;
        }
        CHECK(ok, "two hundred stochastic draws all land inside the vocabulary");
        fox_sampler_destroy(s);
        fox_sampler_destroy(NULL);
    }

    CHECK(fox_model_open("missing.gguf", FOX_WEIGHTS_RESIDENT, 0, &m) == FOX_ERR_IO,
          "a missing model reports IO");
    CHECK(fox_model_open(NULL, FOX_WEIGHTS_RESIDENT, 0, &m) == FOX_ERR_ARG,
          "a null path is rejected");
    fox_model_close(NULL);
    fox_context_destroy(NULL);

    remove(path);
    CHECK_DONE();
}
