/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * common.c — logging, fatal errors, checked allocation, string helpers.
 */
#include "pdl/common.h"

#include <stdarg.h>
#include <string.h>

static pdl_log_level g_level = PDL_LOG_INFO;
static int g_rank = -1;
static pdl_abort_hook g_abort_hook = NULL;

void pdl_log_set_level(pdl_log_level level) { g_level = level; }
void pdl_log_set_rank(int rank) { g_rank = rank; }
void pdl_set_abort_hook(pdl_abort_hook hook) { g_abort_hook = hook; }

static const char *level_tag(pdl_log_level l) {
    switch (l) {
    case PDL_LOG_ERROR: return "ERROR";
    case PDL_LOG_WARN: return "WARN ";
    case PDL_LOG_INFO: return "INFO ";
    default: return "DEBUG";
    }
}

static void vlog(pdl_log_level level, const char *fmt, va_list ap) {
    FILE *out = level <= PDL_LOG_WARN ? stderr : stdout;
    if (g_rank >= 0)
        fprintf(out, "[%s][rank %d] ", level_tag(level), g_rank);
    else
        fprintf(out, "[%s] ", level_tag(level));
    vfprintf(out, fmt, ap);
    fputc('\n', out);
    fflush(out);
}

void pdl_log(pdl_log_level level, const char *fmt, ...) {
    if (level > g_level) return;
    va_list ap;
    va_start(ap, fmt);
    vlog(level, fmt, ap);
    va_end(ap);
}

void pdl_fatal(const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[FATAL]%s%d%s %s:%d: ", g_rank >= 0 ? "[rank " : "", g_rank >= 0 ? g_rank : 0,
            g_rank >= 0 ? "]" : "", file, line);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
    if (g_abort_hook) g_abort_hook(1);
    exit(1);
}

void *pdl_xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) PDL_FATAL("out of memory allocating %zu bytes", bytes);
    return p;
}

void *pdl_xcalloc(size_t count, size_t size) {
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) PDL_FATAL("out of memory allocating %zu x %zu bytes", count, size);
    return p;
}

void *pdl_xrealloc(void *ptr, size_t bytes) {
    void *p = realloc(ptr, bytes ? bytes : 1);
    if (!p) PDL_FATAL("out of memory reallocating %zu bytes", bytes);
    return p;
}

char *pdl_xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)pdl_xmalloc(n);
    memcpy(p, s, n);
    return p;
}

int pdl_str_starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

int pdl_str_split(const char *s, char sep, char **parts, int max_parts) {
    int n = 0;
    const char *start = s;
    for (const char *p = s;; ++p) {
        if (*p == sep || *p == '\0') {
            if (n < max_parts) {
                size_t len = (size_t)(p - start);
                parts[n] = (char *)pdl_xmalloc(len + 1);
                memcpy(parts[n], start, len);
                parts[n][len] = '\0';
                n++;
            }
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    return n;
}

void pdl_str_free_parts(char **parts, int n) {
    for (int i = 0; i < n; ++i) free(parts[i]);
}
