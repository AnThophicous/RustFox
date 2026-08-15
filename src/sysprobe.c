#include "fox_internal.h"
#include "platform.h"

#include <stdio.h>
#include <string.h>

fox_status fox_probe(fox_sysinfo *out, const char *scratch_dir, int measure_storage)
{
    char scratch[FOX_PATH_MAX];
    fox_status rc = FOX_OK;

    if (!out) return FOX_ERR_ARG;
    fox_log_init_from_env();
    memset(out, 0, sizeof(*out));

    fox_probe_cpu(&out->cpu);
    fox_probe_mem(&out->mem);
    fox_probe_gpu(&out->gpu);
    fox_probe_os(out->os, sizeof(out->os), out->kernel, sizeof(out->kernel));

    if (scratch_dir && *scratch_dir) fox_strlcpy(scratch, scratch_dir, sizeof(scratch));
    else                             fox_cache_dir(scratch, sizeof(scratch));

    fox_probe_storage_static(scratch, &out->storage);

    if (measure_storage) {
        fox_status st = fox_bench_storage(scratch, 0, &out->storage);
        if (st != FOX_OK) {
            FOX_WARN("storage benchmark skipped: %s", fox_last_error());
            rc = FOX_OK;
        }
    }

    return rc;
}

static const char *yesno(int v) { return v ? "yes" : "no"; }

void fox_sysinfo_print(const fox_sysinfo *si, void *vfp)
{
    FILE *fp = (FILE *)vfp;
    char b1[64], b2[64], b3[64];
    char feats[256];

    if (!si) return;
    if (!fp) fp = stdout;

    fprintf(fp, "rustfox %s — system probe\n", fox_version());
    fprintf(fp, "\n");

    fprintf(fp, "  host\n");
    fprintf(fp, "    os               %s %s\n", si->os, si->kernel);
    fprintf(fp, "    container        %s%s\n",
            yesno(si->mem.in_container),
            si->mem.cgroup_version ? (si->mem.cgroup_version == 2
                                      ? "  (cgroup v2)" : "  (cgroup v1)") : "");
    fprintf(fp, "\n");

    fprintf(fp, "  cpu\n");
    fprintf(fp, "    model            %s\n", si->cpu.brand);
    fprintf(fp, "    vendor / uarch   %s / %s\n", si->cpu.vendor, si->cpu.uarch);
    fprintf(fp, "    cores            %d physical, %d logical",
            si->cpu.physical_cores, si->cpu.logical_cores);
    if (si->cpu.cpu_quota > 0.0)
        fprintf(fp, "  (cgroup quota %.2f cores)", si->cpu.cpu_quota);
    fprintf(fp, "\n");
    fprintf(fp, "    numa nodes       %d\n", si->cpu.numa_nodes);
    fprintf(fp, "    caches           L1d %s  L2 %s  L3 %s\n",
            fox_fmt_bytes(si->cpu.l1d_bytes, b1, sizeof(b1)),
            fox_fmt_bytes(si->cpu.l2_bytes,  b2, sizeof(b2)),
            fox_fmt_bytes(si->cpu.l3_bytes,  b3, sizeof(b3)));
    fprintf(fp, "    features         %s\n",
            fox_cpu_features_str(si->cpu.features, feats, sizeof(feats)));
    fprintf(fp, "\n");

    fprintf(fp, "  memory\n");
    fprintf(fp, "    usable total     %s",
            fox_fmt_bytes(si->mem.total_bytes, b1, sizeof(b1)));
    if (si->mem.total_bytes != si->mem.host_total_bytes)
        fprintf(fp, "  (host has %s)",
                fox_fmt_bytes(si->mem.host_total_bytes, b2, sizeof(b2)));
    fprintf(fp, "\n");
    fprintf(fp, "    available now    %s\n",
            fox_fmt_bytes(si->mem.available_bytes, b1, sizeof(b1)));
    fprintf(fp, "    page cache       %s\n",
            fox_fmt_bytes(si->mem.cached_bytes, b1, sizeof(b1)));
    if (si->mem.cgroup_version) {
        if (si->mem.cgroup_limit == UINT64_MAX)
            fprintf(fp, "    cgroup limit     unlimited\n");
        else
            fprintf(fp, "    cgroup limit     %s  (using %s)\n",
                    fox_fmt_bytes(si->mem.cgroup_limit,   b1, sizeof(b1)),
                    fox_fmt_bytes(si->mem.cgroup_current, b2, sizeof(b2)));
        if (si->mem.cgroup_high != UINT64_MAX)
            fprintf(fp, "    cgroup high      %s  (throttle point)\n",
                    fox_fmt_bytes(si->mem.cgroup_high, b1, sizeof(b1)));
    }
    fprintf(fp, "    swap             %s free of %s\n",
            fox_fmt_bytes(si->mem.swap_free_bytes,  b1, sizeof(b1)),
            fox_fmt_bytes(si->mem.swap_total_bytes, b2, sizeof(b2)));
    if (si->mem.swappiness >= 0)
        fprintf(fp, "    vm.swappiness    %d\n", si->mem.swappiness);
    fprintf(fp, "    page size        %llu\n",
            (unsigned long long)si->mem.page_size);
    fprintf(fp, "\n");

    fprintf(fp, "  storage  (%s)\n", si->storage.path);
    fprintf(fp, "    device / fs      %s / %s\n",
            si->storage.device, si->storage.fstype);
    fprintf(fp, "    kind             %s\n",
            si->storage.rotational < 0 ? "unknown"
            : (si->storage.rotational ? "rotational (HDD)" : "solid state"));
    fprintf(fp, "    space            %s free of %s\n",
            fox_fmt_bytes(si->storage.free_bytes,     b1, sizeof(b1)),
            fox_fmt_bytes(si->storage.capacity_bytes, b2, sizeof(b2)));
    fprintf(fp, "    O_DIRECT         %s\n", yesno(si->storage.supports_odirect));
    if (si->storage.measured) {
        fprintf(fp, "    seq read         %.0f MiB/s\n", si->storage.seq_read_mbps);
        fprintf(fp, "    4K random read   %.0f IOPS, %.0f us median\n",
                si->storage.rand_read_4k_iops, si->storage.rand_read_lat_us);
    } else {
        fprintf(fp, "    throughput       not measured (pass --bench)\n");
    }
    fprintf(fp, "\n");

    fprintf(fp, "  gpu\n");
    if (!si->gpu.present) {
        fprintf(fp, "    none detected — CPU and storage carry the whole load\n");
    } else {
        fprintf(fp, "    device           %s\n", si->gpu.name);
        fprintf(fp, "    api              %s\n", si->gpu.api);
        fprintf(fp, "    memory           %s\n",
                si->gpu.unified_memory ? "unified with system RAM"
                                       : fox_fmt_bytes(si->gpu.vram_bytes, b1, sizeof(b1)));
        if (si->gpu.unified_memory)
            fprintf(fp, "    note             shares the memory bus with the CPU;\n"
                        "                     splitting work across both can be slower\n"
                        "                     than either alone. The scheduler measures\n"
                        "                     the split rather than assuming it.\n");
    }
    fprintf(fp, "\n");
}

struct jbuf {
    char  *p;
    size_t cap;
    size_t len;
    int    overflow;
};

static void jput(struct jbuf *j, const char *fmt, ...) FOX_PRINTF(2, 3);

static void jput(struct jbuf *j, const char *fmt, ...)
{
    va_list ap;
    int n;

    if (j->overflow) return;
    va_start(ap, fmt);
    n = vsnprintf(j->p + j->len, j->cap - j->len, fmt, ap);
    va_end(ap);

    if (n < 0 || (size_t)n >= j->cap - j->len) { j->overflow = 1; return; }
    j->len += (size_t)n;
}

static const char *jesc(const char *s, char *buf, size_t cap)
{
    size_t o = 0;
    if (!s) s = "";
    for (; *s && o + 7 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  buf[o++] = '\\'; buf[o++] = '"';  break;
        case '\\': buf[o++] = '\\'; buf[o++] = '\\'; break;
        case '\n': buf[o++] = '\\'; buf[o++] = 'n';  break;
        case '\r': buf[o++] = '\\'; buf[o++] = 'r';  break;
        case '\t': buf[o++] = '\\'; buf[o++] = 't';  break;
        default:
            if (c < 0x20) { o += (size_t)snprintf(buf + o, cap - o, "\\u%04x", c); }
            else          { buf[o++] = (char)c; }
        }
    }
    buf[o] = '\0';
    return buf;
}

static void jput_u64_or_null(struct jbuf *j, const char *key, uint64_t v)
{
    if (v == UINT64_MAX) jput(j, "\"%s\":null", key);
    else                 jput(j, "\"%s\":%llu", key, (unsigned long long)v);
}

fox_status fox_sysinfo_to_json(const fox_sysinfo *si, char *buf, size_t buflen,
                               size_t *written)
{
    struct jbuf j;
    char esc[FOX_PATH_MAX * 2 + 8];
    char feats[256];

    if (!si || !buf || buflen == 0) return FOX_ERR_ARG;
    j.p = buf; j.cap = buflen; j.len = 0; j.overflow = 0;
    buf[0] = '\0';

    jput(&j, "{");
    jput(&j, "\"version\":\"%s\",", jesc(fox_version(), esc, sizeof(esc)));
    jput(&j, "\"os\":\"%s\",", jesc(si->os, esc, sizeof(esc)));
    jput(&j, "\"kernel\":\"%s\",", jesc(si->kernel, esc, sizeof(esc)));

    jput(&j, "\"cpu\":{");
    jput(&j, "\"vendor\":\"%s\",", jesc(si->cpu.vendor, esc, sizeof(esc)));
    jput(&j, "\"brand\":\"%s\",", jesc(si->cpu.brand, esc, sizeof(esc)));
    jput(&j, "\"uarch\":\"%s\",", jesc(si->cpu.uarch, esc, sizeof(esc)));
    jput(&j, "\"physical_cores\":%d,", si->cpu.physical_cores);
    jput(&j, "\"logical_cores\":%d,", si->cpu.logical_cores);
    jput(&j, "\"numa_nodes\":%d,", si->cpu.numa_nodes);
    jput(&j, "\"cpu_quota\":%.4f,", si->cpu.cpu_quota);
    jput(&j, "\"l1d_bytes\":%llu,", (unsigned long long)si->cpu.l1d_bytes);
    jput(&j, "\"l2_bytes\":%llu,", (unsigned long long)si->cpu.l2_bytes);
    jput(&j, "\"l3_bytes\":%llu,", (unsigned long long)si->cpu.l3_bytes);
    jput(&j, "\"features\":\"%s\"",
         jesc(fox_cpu_features_str(si->cpu.features, feats, sizeof(feats)),
              esc, sizeof(esc)));
    jput(&j, "},");

    jput(&j, "\"memory\":{");
    jput(&j, "\"total_bytes\":%llu,", (unsigned long long)si->mem.total_bytes);
    jput(&j, "\"host_total_bytes\":%llu,", (unsigned long long)si->mem.host_total_bytes);
    jput(&j, "\"available_bytes\":%llu,", (unsigned long long)si->mem.available_bytes);
    jput(&j, "\"free_bytes\":%llu,", (unsigned long long)si->mem.free_bytes);
    jput(&j, "\"cached_bytes\":%llu,", (unsigned long long)si->mem.cached_bytes);
    jput(&j, "\"swap_total_bytes\":%llu,", (unsigned long long)si->mem.swap_total_bytes);
    jput(&j, "\"swap_free_bytes\":%llu,", (unsigned long long)si->mem.swap_free_bytes);
    jput(&j, "\"page_size\":%llu,", (unsigned long long)si->mem.page_size);
    jput(&j, "\"in_container\":%s,", si->mem.in_container ? "true" : "false");
    jput(&j, "\"cgroup_version\":%d,", si->mem.cgroup_version);
    jput_u64_or_null(&j, "cgroup_limit", si->mem.cgroup_limit); jput(&j, ",");
    jput_u64_or_null(&j, "cgroup_high", si->mem.cgroup_high);   jput(&j, ",");
    jput(&j, "\"cgroup_current\":%llu,", (unsigned long long)si->mem.cgroup_current);
    jput(&j, "\"swappiness\":%d", si->mem.swappiness);
    jput(&j, "},");

    jput(&j, "\"storage\":{");
    jput(&j, "\"path\":\"%s\",", jesc(si->storage.path, esc, sizeof(esc)));
    jput(&j, "\"device\":\"%s\",", jesc(si->storage.device, esc, sizeof(esc)));
    jput(&j, "\"fstype\":\"%s\",", jesc(si->storage.fstype, esc, sizeof(esc)));
    jput(&j, "\"rotational\":%d,", si->storage.rotational);
    jput(&j, "\"supports_odirect\":%s,", si->storage.supports_odirect ? "true" : "false");
    jput(&j, "\"capacity_bytes\":%llu,", (unsigned long long)si->storage.capacity_bytes);
    jput(&j, "\"free_bytes\":%llu,", (unsigned long long)si->storage.free_bytes);
    jput(&j, "\"measured\":%s,", si->storage.measured ? "true" : "false");
    jput(&j, "\"seq_read_mbps\":%.2f,", si->storage.seq_read_mbps);
    jput(&j, "\"rand_read_4k_iops\":%.2f,", si->storage.rand_read_4k_iops);
    jput(&j, "\"rand_read_lat_us\":%.2f", si->storage.rand_read_lat_us);
    jput(&j, "},");

    jput(&j, "\"gpu\":{");
    jput(&j, "\"present\":%s,", si->gpu.present ? "true" : "false");
    jput(&j, "\"name\":\"%s\",", jesc(si->gpu.name, esc, sizeof(esc)));
    jput(&j, "\"api\":\"%s\",", jesc(si->gpu.api, esc, sizeof(esc)));
    jput(&j, "\"vram_bytes\":%llu,", (unsigned long long)si->gpu.vram_bytes);
    jput(&j, "\"unified_memory\":%s", si->gpu.unified_memory ? "true" : "false");
    jput(&j, "}");

    jput(&j, "}");

    if (j.overflow)
        return fox_fail(FOX_ERR_ARG, "json buffer too small (need > %llu bytes)",
                        (unsigned long long)buflen);
    if (written) *written = j.len;
    return FOX_OK;
}
