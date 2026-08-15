#include "fox_internal.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(FILE *fp, int rc)
{
    fprintf(fp,
        "rustfox %s\n"
        "\n"
        "usage: fox <command> [options]\n"
        "\n"
        "commands:\n"
        "  probe            report cpu, memory, cgroup limits, storage and gpu\n"
        "  watch            run the residency governor live and print each tick\n"
        "  version          print the version and exit\n"
        "\n"
        "probe options:\n"
        "  --bench          measure storage throughput and latency for real\n"
        "  --json           emit machine-readable output\n"
        "  --dir PATH       probe this directory instead of the cache dir\n"
        "\n"
        "watch options:\n"
        "  --seconds N      how long to run (default 10, 0 means forever)\n"
        "  --interval MS    tick period (default 250)\n"
        "  --budget BYTES   starting residency budget\n"
        "\n"
        "environment:\n"
        "  FOX_LOG          error|warn|info|debug|trace\n"
        "  FOX_CACHE        where repacked models and scratch files live\n",
        fox_version());
    return rc;
}

static int arg_matches(const char *a, const char *name)
{
    return strcmp(a, name) == 0;
}

static int need_value(int argc, char **argv, int i, const char *name)
{
    if (i + 1 >= argc) {
        fprintf(stderr, "fox: %s needs a value\n", name);
        return 0;
    }
    return 1;
}

static int cmd_probe(int argc, char **argv)
{
    fox_sysinfo si;
    const char *dir = NULL;
    int bench = 0, json = 0;
    int i;
    fox_status st;

    for (i = 0; i < argc; i++) {
        if (arg_matches(argv[i], "--bench")) bench = 1;
        else if (arg_matches(argv[i], "--json")) json = 1;
        else if (arg_matches(argv[i], "--dir")) {
            if (!need_value(argc, argv, i, "--dir")) return 2;
            dir = argv[++i];
        } else {
            fprintf(stderr, "fox probe: unknown option %s\n", argv[i]);
            return 2;
        }
    }

    st = fox_probe(&si, dir, bench);
    if (st != FOX_OK) {
        fprintf(stderr, "fox: probe failed: %s: %s\n",
                fox_status_str(st), fox_last_error());
        return 1;
    }

    if (json) {
        size_t n = 0;
        char *buf = (char *)malloc(16384);
        if (!buf) { fprintf(stderr, "fox: out of memory\n"); return 1; }
        st = fox_sysinfo_to_json(&si, buf, 16384, &n);
        if (st != FOX_OK) {
            fprintf(stderr, "fox: %s: %s\n", fox_status_str(st), fox_last_error());
            free(buf);
            return 1;
        }
        printf("%s\n", buf);
        free(buf);
    } else {
        fox_sysinfo_print(&si, stdout);
    }
    return 0;
}

static const char *action_str(fox_gov_action a)
{
    switch (a) {
    case FOX_GOV_GROW: return "grow";
    case FOX_GOV_CUT:  return "CUT ";
    case FOX_GOV_HOLD: return "hold";
    }
    return "????";
}

static int cmd_watch(int argc, char **argv)
{
    fox_sysinfo si;
    fox_governor_config cfg;
    fox_governor *g;
    unsigned long seconds = 10;
    unsigned long interval = 250;
    unsigned long long budget = 0;
    uint64_t start, deadline_ns;
    char b1[64], b2[64];
    int i;

    for (i = 0; i < argc; i++) {
        if (arg_matches(argv[i], "--seconds")) {
            if (!need_value(argc, argv, i, "--seconds")) return 2;
            seconds = strtoul(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--interval")) {
            if (!need_value(argc, argv, i, "--interval")) return 2;
            interval = strtoul(argv[++i], NULL, 10);
            if (interval < 20) interval = 20;
        } else if (arg_matches(argv[i], "--budget")) {
            if (!need_value(argc, argv, i, "--budget")) return 2;
            budget = strtoull(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "fox watch: unknown option %s\n", argv[i]);
            return 2;
        }
    }

    if (fox_probe(&si, NULL, 0) != FOX_OK) {
        fprintf(stderr, "fox: probe failed: %s\n", fox_last_error());
        return 1;
    }

    fox_governor_config_default(&cfg, &si);
    cfg.tick_ms = (uint32_t)interval;

    g = fox_governor_create(&cfg, (uint64_t)budget);
    if (!g) {
        fprintf(stderr, "fox: %s\n", fox_last_error());
        return 1;
    }

    printf("residency governor: floor %s, ceiling %s, tick %lu ms\n",
           fox_fmt_bytes(cfg.floor_bytes, b1, sizeof(b1)),
           fox_fmt_bytes(cfg.ceil_bytes,  b2, sizeof(b2)),
           interval);
    printf("%-8s %-5s %10s  %7s %7s %9s  %s\n",
           "time", "act", "budget", "mem%", "io%", "majflt/s", "why");
    fflush(stdout);

    start = fox_now_ns();
    deadline_ns = seconds ? start + (uint64_t)seconds * 1000000000ull : 0;

    for (;;) {
        fox_gov_tick tick;
        uint64_t now;

        if (fox_governor_tick(g, &tick) != FOX_OK) {
            fprintf(stderr, "fox: governor tick failed: %s\n", fox_last_error());
            break;
        }

        now = fox_now_ns();
        printf("%7.2fs %-5s %10s  %7.2f %7.2f %9.1f  %s\n",
               (double)(now - start) / 1e9,
               action_str(tick.action),
               fox_fmt_bytes(tick.budget_bytes, b1, sizeof(b1)),
               tick.mem_stall, tick.io_stall, tick.majflt_per_s,
               tick.reason);
        fflush(stdout);

        if (deadline_ns && now >= deadline_ns) break;
        fox_sleep_ms((uint32_t)interval);
    }

    fox_governor_destroy(g);
    return 0;
}

int main(int argc, char **argv)
{
    fox_log_init_from_env();

    if (argc < 2) return usage(stderr, 2);

    if (arg_matches(argv[1], "-h") || arg_matches(argv[1], "--help") ||
        arg_matches(argv[1], "help"))
        return usage(stdout, 0);

    if (arg_matches(argv[1], "version") || arg_matches(argv[1], "--version")) {
        printf("%s\n", fox_version());
        return 0;
    }

    if (arg_matches(argv[1], "probe")) return cmd_probe(argc - 2, argv + 2);
    if (arg_matches(argv[1], "watch")) return cmd_watch(argc - 2, argv + 2);

    fprintf(stderr, "fox: unknown command '%s'\n\n", argv[1]);
    return usage(stderr, 2);
}
