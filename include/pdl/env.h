/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * env.h — capture of the execution environment for reproducibility.
 * Every result file embeds an "environment" object written by pdl_env_write_json.
 */
#ifndef PDL_ENV_H
#define PDL_ENV_H

#include "pdl/backend.h"
#include "pdl/json.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Writes {"environment": {...}} key into the currently open JSON object.
 * Not collective: `hosts` must come from a prior collective pdl_count_hosts().
 * Collects: timestamp, hostname, OS, CPU model, logical cores, compiler,
 * build type/flags, git commit, MPI library + world size, OpenMP threads,
 * CUDA runtime/driver, GPU name/memory/SMs, command line. */
void pdl_env_write_json(pdl_json *j, const pdl_ctx *ctx, int argc, char **argv, int world,
                        int hosts);

/* Short human-readable banner on stdout. */
void pdl_env_print_banner(const pdl_ctx *ctx, int world);

/* CUDA introspection (implemented in backend; return 0 / "" when unavailable). */
int pdl_cuda_device_info(int gpu_id, char *name, size_t name_len, size_t *total_mem,
                         int *sm_count, int *cc_major, int *cc_minor);
int pdl_cuda_versions(int *runtime, int *driver);

#ifdef __cplusplus
}
#endif

#endif /* PDL_ENV_H */
