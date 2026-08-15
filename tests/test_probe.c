#include "fox/fox.h"
#include "check.h"

#include <stdlib.h>
#include <string.h>

static int braces_balance(const char *s)
{
    int depth = 0;
    int in_string = 0;

    for (; *s; s++) {
        if (in_string) {
            if (*s == '\\' && s[1]) { s++; continue; }
            if (*s == '"') in_string = 0;
            continue;
        }
        if (*s == '"') { in_string = 1; continue; }
        if (*s == '{') depth++;
        if (*s == '}') { depth--; if (depth < 0) return 0; }
    }
    return depth == 0 && !in_string;
}

int main(void)
{
    fox_sysinfo si;
    char *json;
    size_t written = 0;
    char feats[256];
    fox_status st;

    st = fox_probe(&si, NULL, 0);
    CHECK(st == FOX_OK, "fox_probe succeeds without benchmarking");

    CHECK(si.cpu.logical_cores >= 1, "at least one logical core");
    CHECK(si.cpu.physical_cores >= 1, "at least one physical core");
    CHECK(si.cpu.physical_cores <= si.cpu.logical_cores,
          "physical cores never exceed logical cores");
    CHECK(si.cpu.numa_nodes >= 1, "at least one numa node");
    CHECK(si.cpu.cacheline_bytes > 0, "cache line size is known");

    CHECK(si.mem.page_size >= 512, "page size is plausible");
    CHECK((si.mem.page_size & (si.mem.page_size - 1)) == 0,
          "page size is a power of two");
    CHECK(si.mem.host_total_bytes > 0, "host reports some physical memory");
    CHECK(si.mem.total_bytes > 0, "usable total is non-zero");
    CHECK(si.mem.total_bytes <= si.mem.host_total_bytes,
          "usable total never exceeds host total");
    CHECK(si.mem.cgroup_version >= 0 && si.mem.cgroup_version <= 2,
          "cgroup version is 0, 1 or 2");

    if (si.mem.cgroup_limit != UINT64_MAX)
        CHECK(si.mem.cgroup_limit > 0, "a cgroup limit, if set, is non-zero");

    CHECK(si.storage.path[0] != '\0', "storage probe recorded a path");
    CHECK(si.storage.rotational >= -1 && si.storage.rotational <= 1,
          "rotational is -1, 0 or 1");
    CHECK(si.storage.measured == 0,
          "throughput stays unmeasured when benchmarking is off");
    CHECK(si.storage.seq_read_mbps == 0.0,
          "no bandwidth is invented when unmeasured");

    CHECK(si.os[0] != '\0', "os name is populated");

    fox_cpu_features_str(si.cpu.features, feats, sizeof(feats));
    CHECK(feats[0] != '\0', "feature string is never empty");

    fox_cpu_features_str(0, feats, sizeof(feats));
    CHECK(strcmp(feats, "none") == 0, "no features prints as none");

    json = (char *)malloc(16384);
    CHECK(json != NULL, "test buffer allocated");
    if (json) {
        st = fox_sysinfo_to_json(&si, json, 16384, &written);
        CHECK(st == FOX_OK, "json serialisation succeeds");
        CHECK(written > 0 && written < 16384, "json length is sane");
        CHECK(json[0] == '{', "json starts with an object");
        CHECK(json[written ? written - 1 : 0] == '}', "json ends with an object");
        CHECK(braces_balance(json), "json braces balance");
        CHECK(strstr(json, "\"logical_cores\"") != NULL,
              "json carries the core count");
        CHECK(strstr(json, "\"cgroup_limit\"") != NULL,
              "json carries the cgroup limit");

        st = fox_sysinfo_to_json(&si, json, 8, &written);
        CHECK(st != FOX_OK, "json serialisation refuses a too-small buffer");

        free(json);
    }

    CHECK(strcmp(fox_status_str(FOX_OK), "ok") == 0, "status strings resolve");
    CHECK(fox_version()[0] != '\0', "version string is present");

    CHECK_DONE();
}
