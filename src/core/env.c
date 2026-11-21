/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * env.c — environment capture for reproducibility (embedded in every result).
 */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include "pdl/env.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "pdl/common.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifndef PDL_BUILD_TYPE
#define PDL_BUILD_TYPE "unknown"
#endif
#ifndef PDL_BUILD_FLAGS
#define PDL_BUILD_FLAGS ""
#endif
#ifndef PDL_GIT_COMMIT
#define PDL_GIT_COMMIT "unknown"
#endif
#ifndef PDL_GIT_DIRTY
#define PDL_GIT_DIRTY 0
#endif

static void cpu_model(char *buf, size_t len) {
    snprintf(buf, len, "unknown");
#if defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ') colon++;
                snprintf(buf, len, "%s", colon);
                size_t n = strlen(buf);
                if (n && buf[n - 1] == '\n') buf[n - 1] = '\0';
            }
            break;
        }
    }
    fclose(f);
#elif defined(__APPLE__)
    size_t n = len;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, NULL, 0) != 0) snprintf(buf, len, "unknown");
#endif
}

static void compiler_string(char *buf, size_t len) {
#if defined(__clang__)
    snprintf(buf, len, "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    snprintf(buf, len, "GNU %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    snprintf(buf, len, "unknown");
#endif
}

static void iso_timestamp(char *buf, size_t len) {
    time_t now = time(NULL);
    struct tm tmv;
    gmtime_r(&now, &tmv);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", &tmv);
}

void pdl_env_write_json(pdl_json *j, const pdl_ctx *ctx, int argc, char **argv, int world,
                        int hosts) {
    char buf[1024];
    pdl_json_begin_object(j, "environment");

    iso_timestamp(buf, sizeof(buf));
    pdl_json_kv_str(j, "timestamp", buf);
    if (gethostname(buf, sizeof(buf)) != 0) snprintf(buf, sizeof(buf), "unknown");
    pdl_json_kv_str(j, "hostname", buf);
    struct utsname u;
    if (uname(&u) == 0) {
        snprintf(buf, sizeof(buf), "%s %s %s", u.sysname, u.release, u.machine);
        pdl_json_kv_str(j, "os", buf);
    }
    cpu_model(buf, sizeof(buf));
    pdl_json_kv_str(j, "cpu_model", buf);
    pdl_json_kv_int(j, "logical_cpus", sysconf(_SC_NPROCESSORS_ONLN));
    compiler_string(buf, sizeof(buf));
    pdl_json_kv_str(j, "compiler", buf);
    pdl_json_kv_str(j, "build_type", PDL_BUILD_TYPE);
    pdl_json_kv_str(j, "build_flags", PDL_BUILD_FLAGS);
    pdl_json_kv_str(j, "git_commit", PDL_GIT_COMMIT);
    pdl_json_kv_bool(j, "git_dirty", PDL_GIT_DIRTY);
    pdl_json_kv_str(j, "pdl_version", PDL_VERSION_STRING);

    char mpiver[MPI_MAX_LIBRARY_VERSION_STRING];
    int mlen = 0;
    MPI_Get_library_version(mpiver, &mlen);
    /* keep the first line only */
    char *nl = strchr(mpiver, '\n');
    if (nl) *nl = '\0';
    pdl_json_kv_str(j, "mpi_library", mpiver);
    pdl_json_kv_int(j, "mpi_world_size", world);
    pdl_json_kv_int(j, "mpi_hosts", hosts);

    pdl_json_kv_bool(j, "openmp", pdl_has_openmp());
    pdl_json_kv_int(j, "omp_max_threads", pdl_openmp_max_threads());

    int rt = 0, drv = 0;
    int has_cuda = pdl_cuda_versions(&rt, &drv);
    pdl_json_kv_bool(j, "cuda", has_cuda);
    pdl_json_kv_int(j, "cuda_runtime", rt);
    pdl_json_kv_int(j, "cuda_driver", drv);
    pdl_json_kv_bool(j, "cublas", pdl_has_cublas());
    char gname[256] = "";
    size_t gmem = 0;
    int sms = 0, ccmaj = 0, ccmin = 0;
    int gid = (ctx && ctx->device == PDL_DEV_CUDA) ? ctx->gpu_id : 0;
    if (has_cuda && pdl_cuda_device_info(gid, gname, sizeof(gname), &gmem, &sms, &ccmaj, &ccmin)) {
        pdl_json_kv_str(j, "gpu_name", gname);
        pdl_json_kv_num(j, "gpu_memory_bytes", (double)gmem);
        pdl_json_kv_int(j, "gpu_sm_count", sms);
        snprintf(buf, sizeof(buf), "%d.%d", ccmaj, ccmin);
        pdl_json_kv_str(j, "gpu_cc", buf);
    } else {
        pdl_json_kv_null(j, "gpu_name");
    }
    if (ctx) {
        pdl_json_kv_str(j, "device", pdl_device_name(ctx->device));
        pdl_json_kv_str(j, "impl", pdl_impl_name(ctx->impl));
        pdl_json_kv_int(j, "threads", ctx->num_threads > 0 ? ctx->num_threads : pdl_openmp_max_threads());
    }
    /* command line */
    size_t off = 0;
    buf[0] = '\0';
    for (int i = 0; i < argc && off < sizeof(buf) - 1; ++i) {
        int w = snprintf(buf + off, sizeof(buf) - off, "%s%s", i ? " " : "", argv[i]);
        if (w < 0) break;
        off += (size_t)w;
    }
    pdl_json_kv_str(j, "command_line", buf);
    const char *omp_env = getenv("OMP_NUM_THREADS");
    pdl_json_kv_str(j, "OMP_NUM_THREADS", omp_env ? omp_env : "");
    pdl_json_end_object(j);
}

void pdl_env_print_banner(const pdl_ctx *ctx, int world) {
    char cpu[256];
    cpu_model(cpu, sizeof(cpu));
    char gname[256] = "-";
    size_t gmem = 0;
    int sms = 0, a = 0, b = 0;
    if (ctx && ctx->device == PDL_DEV_CUDA)
        pdl_cuda_device_info(ctx->gpu_id, gname, sizeof(gname), &gmem, &sms, &a, &b);
    PDL_INFO("pdl-bench %s | ranks=%d | device=%s impl=%s threads=%d | CPU: %s | GPU: %s",
             PDL_VERSION_STRING, world, ctx ? pdl_device_name(ctx->device) : "-",
             ctx ? pdl_impl_name(ctx->impl) : "-",
             ctx && ctx->num_threads > 0 ? ctx->num_threads : pdl_openmp_max_threads(), cpu, gname);
}
