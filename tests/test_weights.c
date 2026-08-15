#include "fox/fox.h"
#include "check.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEMS     64
#define TENSOR_SZ (ELEMS * 4)
#define TENSORS   6

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

static uint8_t byte_of(int tensor, int offset)
{
    return (uint8_t)((tensor * 37 + offset * 11 + 5) & 0xFF);
}

static void write_model(const char *path)
{
    FILE *f = fopen(path, "wb");
    char name[64];
    long at;
    int t, i;

    put32(f, 0x46554747u);
    put32(f, 3);
    put64(f, TENSORS);
    put64(f, 1);

    put_str(f, "general.alignment");
    put32(f, FOX_GGUF_UINT32);
    put32(f, 32);

    for (t = 0; t < TENSORS; t++) {
        snprintf(name, sizeof(name), "blk.%d.weight", t);
        put_str(f, name);
        put32(f, 1);
        put64(f, ELEMS);
        put32(f, FOX_GGML_F32);
        put64(f, (uint64_t)t * TENSOR_SZ);
    }

    at = ftell(f);
    while (at % 32) { fputc(0, f); at++; }

    for (t = 0; t < TENSORS; t++)
        for (i = 0; i < TENSOR_SZ; i++)
            fputc(byte_of(t, i), f);

    fclose(f);
}

static int bytes_match(const fox_weights_lease *lease, int tensor)
{
    const uint8_t *p = (const uint8_t *)lease->data;
    int i;
    if (!p || lease->bytes != TENSOR_SZ) return 0;
    for (i = 0; i < TENSOR_SZ; i++)
        if (p[i] != byte_of(tensor, i)) return 0;
    return 1;
}

int main(void)
{
    const char *path = "rustfox-weights.gguf";
    fox_weights *res = NULL, *str = NULL;
    fox_weights_lease a, b, extra;
    fox_status st;
    uint64_t budget = 3 * TENSOR_SZ;
    int t, pass, i, ok;

    write_model(path);

    st = fox_weights_open(path, FOX_WEIGHTS_RESIDENT, 0, &res);
    CHECK(st == FOX_OK && res != NULL, "the resident backend opens the model");
    if (st != FOX_OK) { remove(path); CHECK_DONE(); }

    CHECK(fox_weights_get_mode(res) == FOX_WEIGHTS_RESIDENT, "mode is reported back");
    CHECK(fox_gguf_tensor_count(fox_weights_model(res)) == TENSORS,
          "every tensor is visible through the weights handle");
    CHECK(fox_weights_resident_bytes(res) == 0,
          "nothing is loaded before the first acquire");

    st = fox_weights_acquire(res, 0, &a);
    CHECK(st == FOX_OK && bytes_match(&a, 0),
          "the resident backend returns the bytes at that tensor's offset");
    CHECK(fox_weights_live_leases(res) == 1, "the lease is counted");
    CHECK(fox_weights_resident_bytes(res) == TENSOR_SZ,
          "only the tensor that was asked for got loaded");

    st = fox_weights_acquire(res, 3, &b);
    CHECK(st == FOX_OK && bytes_match(&b, 3),
          "a second tensor resolves to its own offset");

    fox_weights_release(res, &a);
    CHECK(a.data == NULL,
          "a released lease is cleared, so a stale pointer fails loudly rather "
          "than reading a slot the streaming ring has already refilled");
    fox_weights_release(res, &b);
    CHECK(fox_weights_live_leases(res) == 0, "all leases released");

    for (i = 0; i < FOX_WEIGHTS_MAX_LIVE_LEASES; i++)
        if (fox_weights_acquire(res, (size_t)(i % TENSORS), &extra) != FOX_OK) break;
    CHECK(i == FOX_WEIGHTS_MAX_LIVE_LEASES,
          "leases can be held up to the documented limit");
    CHECK(fox_weights_acquire(res, 0, &extra) == FOX_ERR_INTERNAL,
          "holding more leases than the limit is refused in resident mode too, "
          "so forward-pass code that leaks leases is caught by the reference "
          "backend instead of surviving until streaming exists");
    while (fox_weights_live_leases(res) > 0) fox_weights_release(res, &extra);

    st = fox_weights_open(path, FOX_WEIGHTS_STREAM, budget, &str);
    CHECK(st == FOX_OK && str != NULL,
          "the streaming backend opens with a budget that holds three of six "
          "tensors");
    if (st != FOX_OK) {
        fox_weights_close(res);
        remove(path);
        CHECK_DONE();
    }
    CHECK(fox_weights_get_mode(str) == FOX_WEIGHTS_STREAM, "streaming mode is reported");

    ok = 1;
    for (pass = 0; pass < 3; pass++) {
        for (t = 0; t < TENSORS; t++) {
            fox_weights_lease sl, rl;

            if (fox_weights_acquire(str, (size_t)t, &sl) != FOX_OK) { ok = 0; break; }
            if (fox_weights_acquire(res, (size_t)t, &rl) != FOX_OK) { ok = 0; break; }

            if (sl.bytes != rl.bytes ||
                memcmp(sl.data, rl.data, (size_t)sl.bytes) != 0) ok = 0;
            if (!bytes_match(&sl, t)) ok = 0;
            if (fox_weights_resident_bytes(str) > budget) ok = 0;

            fox_weights_release(str, &sl);
            fox_weights_release(res, &rl);
            if (!ok) break;
        }
        if (!ok) break;
    }
    CHECK(ok,
          "streaming returns byte-for-byte what the resident backend returns, "
          "for every tensor, over three full passes, while never holding more "
          "than the budget allows");

    CHECK(fox_weights_evictions(str) > 0,
          "the budget was actually tight enough to force eviction, so the "
          "comparison above exercised the refill path rather than a warm cache");
    CHECK(fox_weights_reads(str) > TENSORS,
          "evicted tensors were re-read from disk rather than served stale");
    CHECK(fox_weights_resident_bytes(str) <= budget,
          "the streaming backend never exceeds its budget");
    CHECK(fox_weights_reads(res) == TENSORS,
          "the resident backend reads each tensor exactly once no matter how "
          "many times it is acquired");

    CHECK(fox_weights_prefetch(str, 5) == FOX_OK, "prefetch loads a tensor");
    st = fox_weights_acquire(str, 5, &a);
    CHECK(st == FOX_OK && bytes_match(&a, 5), "a prefetched tensor acquires correctly");
    fox_weights_release(str, &a);

    CHECK(fox_weights_acquire(str, TENSORS + 9, &a) == FOX_ERR_NOTFOUND,
          "an out of range tensor index is rejected");

    fox_weights_close(str);
    str = NULL;

    st = fox_weights_open(path, FOX_WEIGHTS_STREAM, 16, &str);
    CHECK(st == FOX_ERR_NOMEM && str == NULL,
          "a budget smaller than the largest single tensor is refused at open, "
          "because streaming cannot help when one tensor does not fit");

    fox_weights_close(res);

    CHECK(fox_weights_open(NULL, FOX_WEIGHTS_RESIDENT, 0, &res) == FOX_ERR_ARG,
          "a null path is rejected");
    CHECK(fox_weights_open("missing.gguf", FOX_WEIGHTS_RESIDENT, 0, &res) == FOX_ERR_IO,
          "a missing model reports IO");
    CHECK(fox_weights_live_leases(NULL) == 0, "a null handle reports no leases");
    fox_weights_close(NULL);

    remove(path);
    CHECK_DONE();
}
