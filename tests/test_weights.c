#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ELEMS      8
#define TENSOR_SZ  (ELEMS * 4)

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

static void write_model(const char *path)
{
    FILE *f = fopen(path, "wb");
    long at;
    int i;

    put32(f, 0x46554747u);
    put32(f, 3);
    put64(f, 2);
    put64(f, 1);

    put_str(f, "general.alignment");
    put32(f, FOX_GGUF_UINT32);
    put32(f, 32);

    put_str(f, "blk.0.attn_q.weight");
    put32(f, 1);
    put64(f, ELEMS);
    put32(f, FOX_GGML_F32);
    put64(f, 0);

    put_str(f, "blk.0.ffn_down.weight");
    put32(f, 1);
    put64(f, ELEMS);
    put32(f, FOX_GGML_F32);
    put64(f, TENSOR_SZ);

    at = ftell(f);
    while (at % 32) { fputc(0, f); at++; }

    for (i = 0; i < TENSOR_SZ; i++)  fputc(i & 0xFF, f);
    for (i = 0; i < TENSOR_SZ; i++)  fputc((0xA0 + i) & 0xFF, f);

    fclose(f);
}

int main(void)
{
    const char *path = "rustfox-weights.gguf";
    fox_weights *w = NULL;
    fox_weights_lease a, b, extra;
    fox_status st;
    int i, ok;

    write_model(path);

    st = fox_weights_open(path, FOX_WEIGHTS_STREAM, 0, &w);
    CHECK(st == FOX_ERR_UNSUPPORTED && w == NULL,
          "the streaming backend says so plainly instead of pretending");

    st = fox_weights_open(path, FOX_WEIGHTS_RESIDENT, 0, &w);
    CHECK(st == FOX_OK && w != NULL, "the resident backend opens the model");
    if (st != FOX_OK) { remove(path); CHECK_DONE(); }

    CHECK(fox_weights_get_mode(w) == FOX_WEIGHTS_RESIDENT, "mode is reported back");
    CHECK(fox_weights_model(w) != NULL, "the parsed model is reachable");
    CHECK(fox_gguf_tensor_count(fox_weights_model(w)) == 2,
          "both tensors are visible through the weights handle");
    CHECK(fox_weights_resident_bytes(w) == 0,
          "nothing is loaded before the first acquire");
    CHECK(fox_weights_live_leases(w) == 0, "no leases are outstanding at open");

    st = fox_weights_acquire(w, 0, &a);
    CHECK(st == FOX_OK, "acquiring the first tensor succeeds");
    CHECK(a.data != NULL && a.bytes == TENSOR_SZ, "the lease describes the tensor");
    CHECK(a.tensor_index == 0, "the lease remembers which tensor it is");
    CHECK(fox_weights_live_leases(w) == 1, "the lease is counted");
    CHECK(fox_weights_resident_bytes(w) == TENSOR_SZ,
          "only the tensor that was asked for got loaded");

    ok = 1;
    for (i = 0; i < TENSOR_SZ; i++)
        if (((const uint8_t *)a.data)[i] != (uint8_t)(i & 0xFF)) ok = 0;
    CHECK(ok, "the bytes read are the bytes that were written at that offset");

    st = fox_weights_acquire(w, 1, &b);
    CHECK(st == FOX_OK, "a second tensor can be held at the same time");
    ok = 1;
    for (i = 0; i < TENSOR_SZ; i++)
        if (((const uint8_t *)b.data)[i] != (uint8_t)((0xA0 + i) & 0xFF)) ok = 0;
    CHECK(ok, "the second tensor resolves to its own offset, not the first");
    CHECK(fox_weights_live_leases(w) == 2, "both leases are counted");

    fox_weights_release(w, &a);
    CHECK(fox_weights_live_leases(w) == 1, "releasing drops the count");
    CHECK(a.data == NULL,
          "a released lease is cleared, so a stale pointer fails loudly rather "
          "than reading a slot the streaming ring has already refilled");

    fox_weights_release(w, &b);
    CHECK(fox_weights_live_leases(w) == 0, "all leases released");

    for (i = 0; i < FOX_WEIGHTS_MAX_LIVE_LEASES; i++) {
        if (fox_weights_acquire(w, (size_t)(i % 2), &extra) != FOX_OK) break;
    }
    CHECK(i == FOX_WEIGHTS_MAX_LIVE_LEASES,
          "leases can be held up to the documented limit");
    CHECK(fox_weights_acquire(w, 0, &extra) == FOX_ERR_INTERNAL,
          "holding more leases than the limit is refused in resident mode too, "
          "so forward-pass code that leaks leases is caught by the reference "
          "backend instead of surviving until streaming exists");

    while (fox_weights_live_leases(w) > 0) fox_weights_release(w, &extra);
    CHECK(fox_weights_live_leases(w) == 0, "the pool drains again");

    fox_weights_close(w);
    w = NULL;

    st = fox_weights_open(path, FOX_WEIGHTS_RESIDENT, 16, &w);
    CHECK(st == FOX_OK, "a tight budget still opens; nothing is loaded yet");
    if (st == FOX_OK) {
        CHECK(fox_weights_acquire(w, 0, &a) == FOX_ERR_NOMEM,
              "a tensor larger than the whole budget is refused rather than "
              "quietly blowing past the governor's limit");
        fox_weights_close(w);
    }

    CHECK(fox_weights_open(NULL, FOX_WEIGHTS_RESIDENT, 0, &w) == FOX_ERR_ARG,
          "a null path is rejected");
    CHECK(fox_weights_open("missing.gguf", FOX_WEIGHTS_RESIDENT, 0, &w) == FOX_ERR_IO,
          "a missing model reports IO");
    CHECK(fox_weights_live_leases(NULL) == 0, "a null handle reports no leases");
    fox_weights_close(NULL);

    remove(path);
    CHECK_DONE();
}
