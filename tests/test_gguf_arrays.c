#include "fox/fox.h"
#include "check.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

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

static void pad_and_fill(FILE *f, uint32_t align, size_t data_bytes)
{
    size_t i;
    long at = ftell(f);
    while (at % (long)align) { fputc(0, f); at++; }
    for (i = 0; i < data_bytes; i++) fputc((int)(i & 0xFF), f);
}

static void write_with_arrays(const char *path, int nest_the_array)
{
    static const char *vocab[] = { "<s>", "hello", "world", "</s>" };
    FILE *f = fopen(path, "wb");
    size_t i;

    put32(f, 0x46554747u);
    put32(f, 3);
    put64(f, 1);
    put64(f, 3);

    put_str(f, "general.alignment");
    put32(f, FOX_GGUF_UINT32);
    put32(f, 32);

    put_str(f, "tokenizer.ggml.tokens");
    put32(f, FOX_GGUF_ARRAY);
    put32(f, nest_the_array ? FOX_GGUF_ARRAY : FOX_GGUF_STRING);
    put64(f, 4);
    for (i = 0; i < 4; i++) put_str(f, vocab[i]);

    put_str(f, "tokenizer.ggml.token_type");
    put32(f, FOX_GGUF_ARRAY);
    put32(f, FOX_GGUF_INT32);
    put64(f, 4);
    for (i = 0; i < 4; i++) put32(f, (uint32_t)i);

    put_str(f, "token_embd.weight");
    put32(f, 1);
    put64(f, 256);
    put32(f, FOX_GGML_TQ2_0);
    put64(f, 0);

    pad_and_fill(f, 32, 66);
    fclose(f);
}

int main(void)
{
    const char *path = "rustfox-arrays.gguf";
    fox_gguf *g = NULL;
    fox_gguf_tensor t;
    fox_status st;
    size_t idx;

    write_with_arrays(path, 0);

    st = fox_gguf_open(path, &g);
    CHECK(st == FOX_OK,
          "a GGUF carrying a string array opens; every real model has a "
          "vocabulary stored exactly this way");

    if (st == FOX_OK) {
        CHECK(fox_gguf_metadata_count(g) == 3, "all three metadata keys survive");

        idx = fox_gguf_find(g, "tokenizer.ggml.tokens");
        CHECK(idx != SIZE_MAX, "the string array key is indexed");
        CHECK(fox_gguf_metadata_type(g, idx) == FOX_GGUF_ARRAY,
              "the string array reports as an array");

        idx = fox_gguf_find(g, "tokenizer.ggml.token_type");
        CHECK(idx != SIZE_MAX, "the numeric array key is indexed");

        CHECK(fox_gguf_tensor_count(g) == 1, "the tensor after the arrays is seen");

        idx = fox_gguf_tensor_find(g, "token_embd.weight");
        CHECK(idx == 0, "the tensor after the arrays is found by name");

        CHECK(fox_gguf_tensor_at(g, 0, &t) == FOX_OK &&
              t.type == FOX_GGML_TQ2_0 && t.n_dims == 1 &&
              t.ne[0] == 256 && t.size_bytes == 66,
              "the tensor descriptor is intact, which only holds if both "
              "arrays were skipped by exactly the right number of bytes");

        fox_gguf_close(g);
        g = NULL;
    }

    remove(path);

    write_with_arrays(path, 1);
    st = fox_gguf_open(path, &g);
    CHECK(st != FOX_OK, "an array of arrays is rejected rather than misread");
    if (st == FOX_OK) fox_gguf_close(g);
    remove(path);

    CHECK(fox_ggml_type_block_size(FOX_GGML_TQ1_0) == 256 &&
          fox_ggml_type_block_bytes(FOX_GGML_TQ1_0) == 54,
          "TQ1_0 is 256 elements in 54 bytes, which is 1.6875 bits per weight");
    CHECK(fox_ggml_type_block_size(FOX_GGML_TQ2_0) == 256 &&
          fox_ggml_type_block_bytes(FOX_GGML_TQ2_0) == 66,
          "TQ2_0 is 256 elements in 66 bytes, which is 2.0625 bits per weight");
    CHECK(fox_ggml_type_block_bytes(FOX_GGML_Q4_0) == 18 &&
          fox_ggml_type_block_bytes(FOX_GGML_Q8_0) == 34 &&
          fox_ggml_type_block_bytes(FOX_GGML_Q4_K) == 144 &&
          fox_ggml_type_block_bytes(FOX_GGML_Q6_K) == 210,
          "the above-1-bit block layouts match ggml");

    CHECK_DONE();
}
