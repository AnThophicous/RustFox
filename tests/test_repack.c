#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEMS       64
#define TENSOR_SIZE (ELEMS * 4)
#define TENSORS     8

static const char *NAMES[TENSORS] = {
    "blk.1.attn_k.weight",
    "output.weight",
    "blk.0.attn_v.weight",
    "token_embd.weight",
    "blk.1.attn_q.weight",
    "output_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight"
};

static void put32(FILE *f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8),
                     (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, sizeof(b), f);
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

static uint8_t byte_of(size_t tensor, size_t offset)
{
    return (uint8_t)((tensor * 37u + offset * 11u + 5u) & 0xFFu);
}

static void write_model(const char *path)
{
    FILE *f = fopen(path, "wb");
    long at;
    size_t t, i;

    put32(f, 0x46554747u);
    put32(f, 3);
    put64(f, TENSORS);
    put64(f, 1);
    put_str(f, "general.alignment");
    put32(f, FOX_GGUF_UINT32);
    put32(f, 32);

    for (t = 0; t < TENSORS; t++) {
        put_str(f, NAMES[t]);
        put32(f, 1);
        put64(f, ELEMS);
        put32(f, FOX_GGML_F32);
        put64(f, (uint64_t)t * TENSOR_SIZE);
    }

    at = ftell(f);
    while (at % 32) { fputc(0, f); at++; }
    for (t = 0; t < TENSORS; t++)
        for (i = 0; i < TENSOR_SIZE; i++) fputc(byte_of(t, i), f);
    fclose(f);
}

static int payload_matches(const fox_weights_lease *lease, size_t tensor)
{
    const uint8_t *p = (const uint8_t *)lease->data;
    size_t i;

    if (!p || lease->bytes != TENSOR_SIZE) return 0;
    for (i = 0; i < TENSOR_SIZE; i++)
        if (p[i] != byte_of(tensor, i)) return 0;
    return 1;
}

int main(void)
{
    const char *input = "rustfox-repack-input.gguf";
    const char *output = "rustfox-repack-output.foxpack";
    fox_gguf *original = NULL;
    fox_gguf *packed = NULL;
    fox_weights *stream = NULL;
    fox_weights_lease lease;
    fox_status st;
    size_t i;

    write_model(input);

    CHECK(fox_repack_gguf(input, output) == FOX_OK,
          "repack writes a valid output file");
    CHECK(fox_repack_gguf(input, input) == FOX_ERR_ARG,
          "repack refuses to overwrite its input in place");

    st = fox_gguf_open(input, &original);
    CHECK(st == FOX_OK, "the source fixture opens as GGUF");
    st = fox_gguf_open(output, &packed);
    CHECK(st == FOX_OK, "the .foxpack output still opens as GGUF");
    if (original && packed) {
        int same = fox_gguf_tensor_count(original) ==
                   fox_gguf_tensor_count(packed);
        for (i = 0; same && i < TENSORS; i++) {
            fox_gguf_tensor a, b;
            if (fox_gguf_tensor_at(original, i, &a) != FOX_OK ||
                fox_gguf_tensor_at(packed, i, &b) != FOX_OK ||
                a.type != b.type || a.n_dims != b.n_dims ||
                a.ne[0] != b.ne[0] || a.size_bytes != b.size_bytes)
                same = 0;
        }
        CHECK(same, "repack preserves tensor descriptors and metadata shape");
    } else {
        CHECK(0, "repack descriptors can be inspected");
    }

    st = fox_weights_open(output, FOX_WEIGHTS_STREAM, 3 * TENSOR_SIZE, &stream);
    CHECK(st == FOX_OK && stream != NULL,
          "the streaming loader accepts a layer-contiguous .foxpack");
    if (stream) {
        size_t q0 = fox_gguf_tensor_find(fox_weights_model(stream),
                                         "blk.0.attn_q.weight");
        size_t k0 = fox_gguf_tensor_find(fox_weights_model(stream),
                                         "blk.0.attn_k.weight");
        size_t q1 = fox_gguf_tensor_find(fox_weights_model(stream),
                                         "blk.1.attn_q.weight");

        st = fox_weights_acquire(stream, q0, &lease);
        CHECK(st == FOX_OK && payload_matches(&lease, 6),
              "the first layer tensor returns its original bytes");
        fox_weights_release(stream, &lease);
        CHECK(fox_weights_reads(stream) == 1,
              "the first tensor causes one contiguous layer read");

        st = fox_weights_acquire(stream, k0, &lease);
        CHECK(st == FOX_OK && payload_matches(&lease, 7),
              "a second tensor slices the cached layer span correctly");
        fox_weights_release(stream, &lease);
        CHECK(fox_weights_reads(stream) == 1,
              "tensors in one layer share their contiguous read");

        st = fox_weights_acquire(stream, q1, &lease);
        CHECK(st == FOX_OK && payload_matches(&lease, 4),
              "a different layer still returns its own bytes");
        fox_weights_release(stream, &lease);
        CHECK(fox_weights_reads(stream) == 2,
              "a different layer starts the next storage read");
        CHECK(fox_weights_resident_bytes(stream) <= 3 * TENSOR_SIZE,
              "layer spans remain within the streaming budget");

        CHECK(fox_weights_prefetch_async(stream, q0) == FOX_OK,
              "the next layer can be requested asynchronously");
        CHECK(fox_weights_prefetch_wait(stream, q0) == FOX_OK,
              "waiting observes completion and worker errors");
        CHECK(fox_weights_reads(stream) == 3,
              "the asynchronous layer read is accounted once");
        st = fox_weights_acquire(stream, q0, &lease);
        CHECK(st == FOX_OK && payload_matches(&lease, 6),
              "an asynchronously prefetched span is immediately usable");
        fox_weights_release(stream, &lease);
        CHECK(fox_weights_reads(stream) == 3,
              "acquire does not reread an asynchronously prefetched layer");
    }

    fox_weights_close(stream);
    fox_gguf_close(packed);
    fox_gguf_close(original);
    remove(output);
    remove(input);
    CHECK_DONE();
}
