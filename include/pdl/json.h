/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * json.h — minimal streaming JSON writer for result files.
 *
 *   pdl_json *j = pdl_json_open("out.json");
 *   pdl_json_begin_object(j, NULL);
 *     pdl_json_kv_str(j, "name", "x");
 *     pdl_json_begin_array(j, "values"); pdl_json_num(j, 1.0); pdl_json_end_array(j);
 *   pdl_json_end_object(j);
 *   pdl_json_close(j);
 */
#ifndef PDL_JSON_H
#define PDL_JSON_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pdl_json pdl_json;

pdl_json *pdl_json_open(const char *path); /* "-" → stdout */
pdl_json *pdl_json_from_file(FILE *f);
void pdl_json_close(pdl_json *j);

/* `key` is used when inside an object, must be NULL inside arrays / at top level. */
void pdl_json_begin_object(pdl_json *j, const char *key);
void pdl_json_end_object(pdl_json *j);
void pdl_json_begin_array(pdl_json *j, const char *key);
void pdl_json_end_array(pdl_json *j);

void pdl_json_kv_str(pdl_json *j, const char *key, const char *v);
void pdl_json_kv_num(pdl_json *j, const char *key, double v);
void pdl_json_kv_int(pdl_json *j, const char *key, long long v);
void pdl_json_kv_bool(pdl_json *j, const char *key, int v);
void pdl_json_kv_null(pdl_json *j, const char *key);
void pdl_json_str(pdl_json *j, const char *v);
void pdl_json_num(pdl_json *j, double v);
void pdl_json_int(pdl_json *j, long long v);

#ifdef __cplusplus
}
#endif

#endif /* PDL_JSON_H */
