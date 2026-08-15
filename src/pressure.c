#include "fox_internal.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_psi_line(const char *line, const char *tag,
                          double *a10, double *a60, double *a300, uint64_t *total)
{
    const char *p = strstr(line, tag);
    double v10 = 0, v60 = 0, v300 = 0;
    unsigned long long tot = 0;

    if (!p) return 0;
    if (sscanf(p, "%*s avg10=%lf avg60=%lf avg300=%lf total=%llu",
               &v10, &v60, &v300, &tot) < 3)
        return 0;

    if (a10)   *a10   = v10;
    if (a60)   *a60   = v60;
    if (a300)  *a300  = v300;
    if (total) *total = (uint64_t)tot;
    return 1;
}

static int read_psi(const char *resource, fox_psi *out)
{
    char path[FOX_PATH_MAX];
    char buf[512];

    memset(out, 0, sizeof(*out));

    snprintf(path, sizeof(path), "/sys/fs/cgroup/%s.pressure", resource);
    if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) {
        snprintf(path, sizeof(path), "/proc/pressure/%s", resource);
        if (fox_read_small_file(path, buf, sizeof(buf)) <= 0) return 0;
    }

    if (!parse_psi_line(buf, "some", &out->some_avg10, &out->some_avg60,
                        &out->some_avg300, &out->some_total_us))
        return 0;

    parse_psi_line(buf, "full", &out->full_avg10, &out->full_avg60,
                   &out->full_avg300, &out->full_total_us);

    out->available = 1;
    return 1;
}

static int flatfile_field(const char *text, const char *key, uint64_t *out)
{
    size_t klen = strlen(key);
    const char *p = text;

    while (p && *p) {
        if (strncmp(p, key, klen) == 0 && (p[klen] == ' ' || p[klen] == '\t')) {
            *out = (uint64_t)strtoull(p + klen + 1, NULL, 10);
            return 1;
        }
        p = strchr(p, '\n');
        if (p) p++;
    }
    return 0;
}

fox_status fox_pressure_read(fox_pressure *out)
{
    char buf[8192];
    int any = 0;

    if (!out) return FOX_ERR_ARG;
    memset(out, 0, sizeof(*out));

    any |= read_psi("cpu",    &out->cpu);
    any |= read_psi("memory", &out->mem);
    any |= read_psi("io",     &out->io);

    if (fox_read_small_file("/sys/fs/cgroup/memory.stat", buf, sizeof(buf)) > 0) {
        if (flatfile_field(buf, "pgmajfault", &out->pgmajfault)) any = 1;
    }
    if (out->pgmajfault == 0 &&
        fox_read_small_file("/proc/vmstat", buf, sizeof(buf)) > 0) {
        if (flatfile_field(buf, "pgmajfault", &out->pgmajfault)) any = 1;
        flatfile_field(buf, "pswpin", &out->pswpin);
    }

    if (fox_read_small_file("/sys/fs/cgroup/memory.events", buf, sizeof(buf)) > 0) {
        flatfile_field(buf, "high", &out->cgroup_high_events);
        flatfile_field(buf, "max",  &out->cgroup_max_events);
        any = 1;
    }

    out->available = any;
    return any ? FOX_OK
               : fox_fail(FOX_ERR_UNSUPPORTED,
                          "no pressure telemetry on this platform "
                          "(need Linux with CONFIG_PSI or cgroup v2)");
}
