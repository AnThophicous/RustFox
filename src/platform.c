#include "platform.h"
#include "fox_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t fox_read_small_file(const char *path, char *buf, size_t cap)
{
    FILE *fp;
    size_t n;

    if (!path || !buf || cap == 0) return -1;
    buf[0] = '\0';

    fp = fopen(path, "rb");
    if (!fp) return -1;
    n = fread(buf, 1, cap - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    return (int64_t)n;
}

fox_status fox_read_u64_file(const char *path, uint64_t *out)
{
    char buf[128];
    char *p = buf;
    char *end = NULL;
    unsigned long long v;

    if (!out) return FOX_ERR_ARG;
    if (fox_read_small_file(path, buf, sizeof(buf)) <= 0)
        return FOX_ERR_NOTFOUND;

    while (*p == ' ' || *p == '\t' || *p == '\n') p++;

    if (strncmp(p, "max", 3) == 0) { *out = UINT64_MAX; return FOX_OK; }

    v = strtoull(p, &end, 10);
    if (end == p) return FOX_ERR_FORMAT;
    *out = (uint64_t)v;
    return FOX_OK;
}

char *fox_cache_dir(char *buf, size_t cap)
{
    const char *e;

    e = getenv("FOX_CACHE");
    if (e && *e) { fox_strlcpy(buf, e, cap); return buf; }

    e = getenv("XDG_CACHE_HOME");
    if (e && *e) { snprintf(buf, cap, "%s/rustfox", e); return buf; }

    e = getenv("HOME");
    if (e && *e) { snprintf(buf, cap, "%s/.cache/rustfox", e); return buf; }

#if defined(FOX_OS_WINDOWS)
    e = getenv("LOCALAPPDATA");
    if (e && *e) { snprintf(buf, cap, "%s\\rustfox\\cache", e); return buf; }
#endif

    {
        char tmp[FOX_PATH_MAX];
        fox_temp_dir(tmp, sizeof(tmp));
        snprintf(buf, cap, "%s/rustfox", tmp);
    }
    return buf;
}

#define FOX_BENCH_BLOCK      (1u << 20)
#define FOX_BENCH_RAND_BLOCK 4096u
#define FOX_BENCH_RAND_OPS   512

static uint64_t bench_rand(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

fox_status fox_bench_storage(const char *dir, uint64_t probe_bytes,
                             fox_storage_info *io)
{
    char path[FOX_PATH_MAX];
    char scratch[FOX_PATH_MAX];
    uint64_t cap = 0, avail = 0;
    fox_status st;
    FILE *fp = NULL;
    fox_file *f = NULL;
    void *block = NULL;
    int direct = 0;
    uint64_t i, nblocks;
    uint64_t t0, t1;
    double secs;
    double lat[FOX_BENCH_RAND_OPS];
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    fox_status rc = FOX_OK;

    if (!io) return FOX_ERR_ARG;

    if (dir && *dir) fox_strlcpy(scratch, dir, sizeof(scratch));
    else             fox_cache_dir(scratch, sizeof(scratch));

    st = fox_mkdir_p(scratch);
    if (st != FOX_OK) return st;

    if (fox_fs_space(scratch, &cap, &avail) != FOX_OK) avail = 0;

    if (probe_bytes == 0) probe_bytes = 64ull << 20;
    if (avail > 0 && probe_bytes > avail / 8) probe_bytes = avail / 8;
    if (probe_bytes < (8ull << 20)) {
        return fox_fail(FOX_ERR_IO,
                        "not enough free space under %s to benchmark storage",
                        scratch);
    }
    probe_bytes &= ~(uint64_t)(FOX_BENCH_BLOCK - 1);
    nblocks = probe_bytes / FOX_BENCH_BLOCK;

    snprintf(path, sizeof(path), "%s/.fox-iobench.tmp", scratch);

    block = fox_aligned_alloc(4096, FOX_BENCH_BLOCK);
    if (!block) return fox_fail(FOX_ERR_NOMEM, "iobench buffer");

    {
        unsigned char *b = (unsigned char *)block;
        for (i = 0; i < FOX_BENCH_BLOCK; i++) b[i] = (unsigned char)(i * 7u + 13u);
    }
    fp = fopen(path, "wb");
    if (!fp) { rc = fox_fail(FOX_ERR_IO, "create %s", path); goto done; }
    for (i = 0; i < nblocks; i++) {
        if (fwrite(block, 1, FOX_BENCH_BLOCK, fp) != FOX_BENCH_BLOCK) {
            fclose(fp); fp = NULL;
            rc = fox_fail(FOX_ERR_IO, "write %s (out of space?)", path);
            goto done;
        }
    }
    fflush(fp);
    fclose(fp);
    fp = NULL;
    fox_sync_path(path);

    f = fox_file_open_read(path, FOX_OPEN_DIRECT | FOX_OPEN_SEQ, &direct);
    if (!f) { rc = FOX_ERR_IO; goto done; }
    if (!direct) {
        if (!fox_file_drop_cache(f)) {
            FOX_WARN("storage bench: no O_DIRECT and no cache drop on %s; "
                     "sequential number will be optimistic", io->fstype);
        }
    }

    t0 = fox_now_ns();
    for (i = 0; i < nblocks; i++) {
        if (fox_file_pread(f, block, FOX_BENCH_BLOCK,
                           i * (uint64_t)FOX_BENCH_BLOCK) != (int64_t)FOX_BENCH_BLOCK) {
            rc = fox_fail(FOX_ERR_IO, "sequential read failed at block %llu",
                          (unsigned long long)i);
            goto done;
        }
    }
    t1 = fox_now_ns();
    secs = (double)(t1 - t0) / 1e9;
    if (secs > 0.0)
        io->seq_read_mbps = ((double)probe_bytes / (1024.0 * 1024.0)) / secs;

    fox_file_close(f);
    f = fox_file_open_read(path, FOX_OPEN_DIRECT | FOX_OPEN_RANDOM, &direct);
    if (!f) { rc = FOX_ERR_IO; goto done; }
    if (!direct) (void)fox_file_drop_cache(f);

    {
        uint64_t span = probe_bytes - FOX_BENCH_RAND_BLOCK;
        int ops = 0;
        uint64_t rt0 = fox_now_ns();
        for (i = 0; i < FOX_BENCH_RAND_OPS; i++) {
            uint64_t off = (bench_rand(&rng) % span) & ~(uint64_t)(FOX_BENCH_RAND_BLOCK - 1);
            uint64_t a = fox_now_ns();
            int64_t got = fox_file_pread(f, block, FOX_BENCH_RAND_BLOCK, off);
            uint64_t b = fox_now_ns();
            if (got != (int64_t)FOX_BENCH_RAND_BLOCK) break;
            lat[ops++] = (double)(b - a) / 1000.0;
        }
        {
            uint64_t rt1 = fox_now_ns();
            double rsecs = (double)(rt1 - rt0) / 1e9;
            if (ops > 0 && rsecs > 0.0) io->rand_read_4k_iops = (double)ops / rsecs;
        }
        if (ops > 0) {
            qsort(lat, (size_t)ops, sizeof(lat[0]), cmp_double);
            io->rand_read_lat_us = lat[ops / 2];
        }
    }

    io->supports_odirect = direct;
    io->measured = 1;

done:
    if (f)  fox_file_close(f);
    if (fp) fclose(fp);
    if (block) fox_aligned_free(block);
    remove(path);
    return rc;
}
