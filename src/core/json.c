/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * json.c — minimal streaming JSON writer with automatic comma handling.
 */
#include "pdl/json.h"

#include <math.h>
#include <string.h>

#include "pdl/common.h"

#define PDL_JSON_MAX_DEPTH 64

struct pdl_json {
    FILE *f;
    int owns;
    int depth;
    int first[PDL_JSON_MAX_DEPTH]; /* 1 if no element written yet at this depth */
};

pdl_json *pdl_json_from_file(FILE *f) {
    pdl_json *j = (pdl_json *)pdl_xcalloc(1, sizeof(*j));
    j->f = f;
    j->owns = 0;
    j->first[0] = 1;
    return j;
}

pdl_json *pdl_json_open(const char *path) {
    if (strcmp(path, "-") == 0) return pdl_json_from_file(stdout);
    FILE *f = fopen(path, "w");
    if (!f) {
        PDL_ERROR("cannot open '%s' for writing", path);
        return NULL;
    }
    pdl_json *j = pdl_json_from_file(f);
    j->owns = 1;
    return j;
}

void pdl_json_close(pdl_json *j) {
    if (!j) return;
    fputc('\n', j->f);
    if (j->owns)
        fclose(j->f);
    else
        fflush(j->f);
    free(j);
}

static void indent(pdl_json *j) {
    fputc('\n', j->f);
    for (int i = 0; i < j->depth; ++i) fputs("  ", j->f);
}

static void write_escaped(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; p && *p; ++p) {
        switch (*p) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20)
                fprintf(f, "\\u%04x", *p);
            else
                fputc(*p, f);
        }
    }
    fputc('"', f);
}

/* Emits separator + key for the next element. */
static void prefix(pdl_json *j, const char *key) {
    if (j->depth > 0) {
        if (!j->first[j->depth]) fputc(',', j->f);
        j->first[j->depth] = 0;
        indent(j);
    }
    if (key) {
        write_escaped(j->f, key);
        fputs(": ", j->f);
    }
}

void pdl_json_begin_object(pdl_json *j, const char *key) {
    prefix(j, key);
    fputc('{', j->f);
    PDL_CHECK(j->depth + 1 < PDL_JSON_MAX_DEPTH, "JSON nesting too deep");
    j->first[++j->depth] = 1;
}

void pdl_json_end_object(pdl_json *j) {
    int empty = j->first[j->depth];
    j->depth--;
    if (!empty) indent(j);
    fputc('}', j->f);
}

void pdl_json_begin_array(pdl_json *j, const char *key) {
    prefix(j, key);
    fputc('[', j->f);
    PDL_CHECK(j->depth + 1 < PDL_JSON_MAX_DEPTH, "JSON nesting too deep");
    j->first[++j->depth] = 1;
}

void pdl_json_end_array(pdl_json *j) {
    int empty = j->first[j->depth];
    j->depth--;
    if (!empty) indent(j);
    fputc(']', j->f);
}

static void write_num(FILE *f, double v) {
    if (isnan(v) || isinf(v))
        fputs("null", f); /* JSON has no NaN/Inf */
    else if (v == (double)(long long)v && fabs(v) < 1e15)
        fprintf(f, "%lld", (long long)v);
    else
        fprintf(f, "%.9g", v);
}

void pdl_json_kv_str(pdl_json *j, const char *key, const char *v) {
    prefix(j, key);
    if (v)
        write_escaped(j->f, v);
    else
        fputs("null", j->f);
}
void pdl_json_kv_num(pdl_json *j, const char *key, double v) {
    prefix(j, key);
    write_num(j->f, v);
}
void pdl_json_kv_int(pdl_json *j, const char *key, long long v) {
    prefix(j, key);
    fprintf(j->f, "%lld", v);
}
void pdl_json_kv_bool(pdl_json *j, const char *key, int v) {
    prefix(j, key);
    fputs(v ? "true" : "false", j->f);
}
void pdl_json_kv_null(pdl_json *j, const char *key) {
    prefix(j, key);
    fputs("null", j->f);
}
void pdl_json_str(pdl_json *j, const char *v) { pdl_json_kv_str(j, NULL, v); }
void pdl_json_num(pdl_json *j, double v) { pdl_json_kv_num(j, NULL, v); }
void pdl_json_int(pdl_json *j, long long v) { pdl_json_kv_int(j, NULL, v); }
