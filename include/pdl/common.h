/*
 * pdl-bench — Parallel & Distributed Deep Learning Benchmark
 * SPDX-License-Identifier: MIT
 *
 * common.h — shared macros, error handling and logging.
 */
#ifndef PDL_COMMON_H
#define PDL_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PDL_VERSION_STRING "1.0.0"

#define PDL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PDL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PDL_UNUSED(x) ((void)(x))
#define PDL_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Log levels. */
typedef enum { PDL_LOG_ERROR = 0, PDL_LOG_WARN = 1, PDL_LOG_INFO = 2, PDL_LOG_DEBUG = 3 } pdl_log_level;

void pdl_log_set_level(pdl_log_level level);
void pdl_log_set_rank(int rank); /* prefixes messages with [rank N] */
void pdl_log(pdl_log_level level, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define PDL_ERROR(...) pdl_log(PDL_LOG_ERROR, __VA_ARGS__)
#define PDL_WARN(...) pdl_log(PDL_LOG_WARN, __VA_ARGS__)
#define PDL_INFO(...) pdl_log(PDL_LOG_INFO, __VA_ARGS__)
#define PDL_DEBUG(...) pdl_log(PDL_LOG_DEBUG, __VA_ARGS__)

/* Fatal error: logs, then aborts the whole job (MPI_Abort when MPI is up). */
void pdl_fatal(const char *file, int line, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((noreturn, format(printf, 3, 4)))
#endif
    ;

#define PDL_FATAL(...) pdl_fatal(__FILE__, __LINE__, __VA_ARGS__)
#define PDL_CHECK(cond, ...)                                                                       \
    do {                                                                                           \
        if (!(cond)) pdl_fatal(__FILE__, __LINE__, __VA_ARGS__);                                   \
    } while (0)

/* Hook used by pdl_fatal to tear down MPI; installed by comm module. */
typedef void (*pdl_abort_hook)(int code);
void pdl_set_abort_hook(pdl_abort_hook hook);

/* Checked host allocation helpers. */
void *pdl_xmalloc(size_t bytes);
void *pdl_xcalloc(size_t count, size_t size);
void *pdl_xrealloc(void *ptr, size_t bytes);
char *pdl_xstrdup(const char *s);

/* Small string helpers. */
int pdl_str_starts_with(const char *s, const char *prefix);
/* Splits `s` by `sep` into at most `max_parts` newly allocated strings; returns count. */
int pdl_str_split(const char *s, char sep, char **parts, int max_parts);
void pdl_str_free_parts(char **parts, int n);

#ifdef __cplusplus
}
#endif

#endif /* PDL_COMMON_H */
