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
        "  run              generate text from a GGUF model\n"
        "  repack           arrange a GGUF for contiguous streamed layers\n"
        "  probe            report cpu, memory, cgroup limits, storage and gpu\n"
        "  watch            run the residency governor live and print each tick\n"
        "  version          print the version and exit\n"
        "\n"
        "run options:\n"
        "  <model.gguf>     the model to load\n"
        "  -p, --prompt S   prompt text (default \"Hello\")\n"
        "  -n, --predict N  tokens to generate (default 64)\n"
        "  --ctx N          context length (default: whatever the model trained on)\n"
        "  --batch N        tokens per weight pass (default 32)\n"
        "  -t, --threads N  worker threads (default: one per core)\n"
        "  --stream         stream weights from storage instead of loading them\n"
        "  --budget BYTES   cap on resident weight bytes\n"
        "  --temp F         sampling temperature, 0 for greedy (default 0.8)\n"
        "  --top-k N        (default 40)\n"
        "  --top-p F        (default 0.95)\n"
        "  --seed N         reproducible sampling\n"
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

static int cmd_repack(int argc, char **argv)
{
    fox_status st;

    if (argc != 2) {
        fprintf(stderr, "usage: fox repack <input.gguf> <output.foxpack>\n");
        return 2;
    }

    st = fox_repack_gguf(argv[0], argv[1]);
    if (st != FOX_OK) {
        fprintf(stderr, "fox repack: %s: %s\n",
                fox_status_str(st), fox_last_error());
        return 1;
    }
    printf("repacked %s -> %s\n", argv[0], argv[1]);
    return 0;
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

static int cmd_run(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = "Hello";
    fox_model *model = NULL;
    fox_context *ctx = NULL;
    fox_sampler *sampler = NULL;
    fox_context_config ccfg;
    fox_sampler_config scfg;
    const fox_model_info *mi;
    const fox_tokenizer *tok;
    fox_weights_mode mode = FOX_WEIGHTS_RESIDENT;
    unsigned long n_predict = 64;
    unsigned long long budget = 0;
    uint32_t *ids = NULL;
    size_t n_ids = 0, cap;
    char b1[64];
    float *logits = NULL;
    double prompt_seconds = 0.0, gen_seconds = 0.0;
    unsigned long generated = 0;
    int rc = 1, i;

    fox_sampler_config_default(&scfg);
    memset(&ccfg, 0, sizeof(ccfg));

    for (i = 0; i < argc; i++) {
        if (argv[i][0] != '-' && !path) { path = argv[i]; continue; }
        if (arg_matches(argv[i], "--prompt") || arg_matches(argv[i], "-p")) {
            if (!need_value(argc, argv, i, "--prompt")) return 2;
            prompt = argv[++i];
        } else if (arg_matches(argv[i], "-n") || arg_matches(argv[i], "--predict")) {
            if (!need_value(argc, argv, i, "-n")) return 2;
            n_predict = strtoul(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--ctx")) {
            if (!need_value(argc, argv, i, "--ctx")) return 2;
            ccfg.n_ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--batch")) {
            if (!need_value(argc, argv, i, "--batch")) return 2;
            ccfg.n_batch = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--threads") || arg_matches(argv[i], "-t")) {
            if (!need_value(argc, argv, i, "--threads")) return 2;
            ccfg.n_threads = (int)strtol(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--stream")) {
            mode = FOX_WEIGHTS_STREAM;
        } else if (arg_matches(argv[i], "--budget")) {
            if (!need_value(argc, argv, i, "--budget")) return 2;
            budget = strtoull(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--temp")) {
            if (!need_value(argc, argv, i, "--temp")) return 2;
            scfg.temperature = (float)atof(argv[++i]);
        } else if (arg_matches(argv[i], "--top-k")) {
            if (!need_value(argc, argv, i, "--top-k")) return 2;
            scfg.top_k = (int)strtol(argv[++i], NULL, 10);
        } else if (arg_matches(argv[i], "--top-p")) {
            if (!need_value(argc, argv, i, "--top-p")) return 2;
            scfg.top_p = (float)atof(argv[++i]);
        } else if (arg_matches(argv[i], "--seed")) {
            if (!need_value(argc, argv, i, "--seed")) return 2;
            scfg.seed = strtoull(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "fox run: unknown option %s\n", argv[i]);
            return 2;
        }
    }

    if (!path) {
        fprintf(stderr, "fox run: give me a model path\n");
        return 2;
    }

    if (fox_model_open(path, mode, budget, &model) != FOX_OK) {
        fprintf(stderr, "fox: %s\n", fox_last_error());
        return 1;
    }
    mi = fox_model_get_info(model);
    tok = fox_model_tokenizer(model);

    fprintf(stderr, "model   %s (%s)\n", mi->name, mi->arch);
    fprintf(stderr, "shape   %u layers, %u embd, %u heads (%u kv), %u ff\n",
            mi->n_layer, mi->n_embd, mi->n_head, mi->n_head_kv, mi->n_ff);
    fprintf(stderr, "vocab   %u%s\n", mi->n_vocab,
            mi->tied_output ? ", output tied to embedding" : "");
    fprintf(stderr, "weights %s, %s\n",
            fox_fmt_bytes(mi->weight_bytes, b1, sizeof(b1)),
            mode == FOX_WEIGHTS_STREAM ? "streamed" : "resident");

    if (!tok) {
        fprintf(stderr, "fox: this model has no tokenizer metadata we can use\n");
        goto done;
    }

    if (fox_context_create(model, &ccfg, &ctx) != FOX_OK) {
        fprintf(stderr, "fox: %s\n", fox_last_error());
        goto done;
    }

    cap = strlen(prompt) + 8;
    ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!ids) { fprintf(stderr, "fox: out of memory\n"); goto done; }

    if (fox_tokenizer_encode(tok, prompt, ids, cap, &n_ids) != FOX_OK || n_ids == 0) {
        fprintf(stderr, "fox: cannot tokenise the prompt: %s\n", fox_last_error());
        goto done;
    }

    sampler = fox_sampler_create(&scfg, mi->n_vocab);
    if (!sampler) { fprintf(stderr, "fox: %s\n", fox_last_error()); goto done; }

    fprintf(stderr, "prompt  %llu tokens\n\n", (unsigned long long)n_ids);
    fputs(prompt, stdout);
    fflush(stdout);

    if (fox_eval(ctx, ids, n_ids, &logits) != FOX_OK) {
        fprintf(stderr, "\nfox: %s\n", fox_last_error());
        goto done;
    }
    prompt_seconds = fox_context_last_eval_seconds(ctx);
    for (i = 0; i < (int)n_ids; i++) fox_sampler_accept(sampler, ids[i]);

    while (generated < n_predict) {
        uint32_t next = 0;
        char piece[256];
        size_t written = 0;

        if (fox_sampler_pick(sampler, logits, &next) != FOX_OK) break;
        if (next == fox_tokenizer_eos(tok)) break;

        if (fox_tokenizer_decode(tok, &next, 1, piece, sizeof(piece), &written) == FOX_OK) {
            fwrite(piece, 1, written, stdout);
            fflush(stdout);
        }

        fox_sampler_accept(sampler, next);
        generated++;

        if (fox_eval(ctx, &next, 1, &logits) != FOX_OK) {
            fprintf(stderr, "\nfox: %s\n", fox_last_error());
            break;
        }
        gen_seconds += fox_context_last_eval_seconds(ctx);
    }

    fprintf(stdout, "\n\n");
    fprintf(stderr, "prompt  %llu tokens in %.2fs  (%.2f tok/s)\n",
            (unsigned long long)n_ids, prompt_seconds,
            prompt_seconds > 0.0 ? (double)n_ids / prompt_seconds : 0.0);
    fprintf(stderr, "gen     %lu tokens in %.2fs  (%.2f tok/s)\n",
            generated, gen_seconds,
            gen_seconds > 0.0 ? (double)generated / gen_seconds : 0.0);
    fprintf(stderr, "reads   %llu tensor reads from storage\n",
            (unsigned long long)fox_context_weight_reads(ctx));
    rc = 0;

done:
    free(ids);
    if (sampler) fox_sampler_destroy(sampler);
    if (ctx) fox_context_destroy(ctx);
    if (model) fox_model_close(model);
    return rc;
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
    if (arg_matches(argv[1], "repack")) return cmd_repack(argc - 2, argv + 2);
    if (arg_matches(argv[1], "run"))   return cmd_run(argc - 2, argv + 2);

    fprintf(stderr, "fox: unknown command '%s'\n\n", argv[1]);
    return usage(stderr, 2);
}
