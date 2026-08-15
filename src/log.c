#include "fox_internal.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  define FOX_TLS __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#  define FOX_TLS __thread
#else
#  define FOX_TLS
#endif

const char *fox_version(void)
{
#ifdef FOX_GIT_DESCRIBE
    return FOX_GIT_DESCRIBE;
#else
    return "0.1.0";
#endif
}

const char *fox_status_str(fox_status s)
{
    switch (s) {
    case FOX_OK:              return "ok";
    case FOX_ERR_ARG:         return "invalid argument";
    case FOX_ERR_IO:          return "io error";
    case FOX_ERR_NOMEM:       return "out of memory";
    case FOX_ERR_FORMAT:      return "bad format";
    case FOX_ERR_UNSUPPORTED: return "unsupported";
    case FOX_ERR_NOTFOUND:    return "not found";
    case FOX_ERR_INTERNAL:    return "internal error";
    }
    return "unknown status";
}

static FOX_TLS char g_err[512];

const char *fox_last_error(void)
{
    return g_err[0] ? g_err : "no error recorded";
}

fox_status fox_fail(fox_status st, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);

    if (fox_log_get_level() >= FOX_LOG_DEBUG)
        fprintf(stderr, "fox: error: %s\n", g_err);
    return st;
}

static fox_log_level g_level = FOX_LOG_WARN;
static int           g_level_from_env = 0;

void fox_log_set_level(fox_log_level lv) { g_level = lv; }
fox_log_level fox_log_get_level(void)    { return g_level; }

void fox_log_init_from_env(void)
{
    const char *e;
    if (g_level_from_env) return;
    g_level_from_env = 1;

    e = getenv("FOX_LOG");
    if (!e || !*e) return;

    if      (strcmp(e, "error") == 0) g_level = FOX_LOG_ERROR;
    else if (strcmp(e, "warn")  == 0) g_level = FOX_LOG_WARN;
    else if (strcmp(e, "info")  == 0) g_level = FOX_LOG_INFO;
    else if (strcmp(e, "debug") == 0) g_level = FOX_LOG_DEBUG;
    else if (strcmp(e, "trace") == 0) g_level = FOX_LOG_TRACE;
    else fprintf(stderr, "fox: ignoring FOX_LOG=%s "
                         "(want error|warn|info|debug|trace)\n", e);
}

static const char *level_tag(fox_log_level lv)
{
    switch (lv) {
    case FOX_LOG_ERROR: return "error";
    case FOX_LOG_WARN:  return "warn ";
    case FOX_LOG_INFO:  return "info ";
    case FOX_LOG_DEBUG: return "debug";
    case FOX_LOG_TRACE: return "trace";
    }
    return "?????";
}

void fox_logv(fox_log_level lv, const char *fmt, ...)
{
    va_list ap;
    if (lv > g_level) return;

    fprintf(stderr, "fox %s ", level_tag(lv));
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

char *fox_fmt_bytes(uint64_t bytes, char *buf, size_t cap)
{
    static const char *unit[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
    double v = (double)bytes;
    size_t u = 0;

    if (!buf || cap == 0) return buf;

    while (v >= 1024.0 && u + 1 < FOX_ARRAY_LEN(unit)) {
        v /= 1024.0;
        u++;
    }
    if (u == 0) snprintf(buf, cap, "%llu B", (unsigned long long)bytes);
    else        snprintf(buf, cap, "%.2f %s", v, unit[u]);
    return buf;
}
