#ifndef FOX_INTERNAL_H
#define FOX_INTERNAL_H

#include "fox/fox.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#  define FOX_PRINTF(f, a) __attribute__((format(printf, f, a)))
#  define FOX_UNUSED       __attribute__((unused))
#else
#  define FOX_PRINTF(f, a)
#  define FOX_UNUSED
#endif

fox_status fox_fail(fox_status st, const char *fmt, ...) FOX_PRINTF(2, 3);

void fox_logv(fox_log_level lv, const char *fmt, ...) FOX_PRINTF(2, 3);

#define FOX_ERROR(...) fox_logv(FOX_LOG_ERROR, __VA_ARGS__)
#define FOX_WARN(...)  fox_logv(FOX_LOG_WARN,  __VA_ARGS__)
#define FOX_INFO(...)  fox_logv(FOX_LOG_INFO,  __VA_ARGS__)
#define FOX_DEBUG(...) fox_logv(FOX_LOG_DEBUG, __VA_ARGS__)
#define FOX_TRACE(...) fox_logv(FOX_LOG_TRACE, __VA_ARGS__)

#define FOX_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static FOX_UNUSED uint64_t fox_min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }
static FOX_UNUSED uint64_t fox_max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }

static FOX_UNUSED uint64_t fox_clamp_u64(uint64_t v, uint64_t lo, uint64_t hi)
{
    if (hi < lo) hi = lo;
    if (v < lo)  return lo;
    if (v > hi)  return hi;
    return v;
}

static FOX_UNUSED double fox_clamp_f64(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static FOX_UNUSED void fox_strlcpy(char *dst, const char *src, size_t cap)
{
    size_t n;
    if (cap == 0) return;
    if (src == NULL) { dst[0] = '\0'; return; }
    n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

char *fox_fmt_bytes(uint64_t bytes, char *buf, size_t cap);

#endif
