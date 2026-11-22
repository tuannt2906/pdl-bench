/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * backend.c — device dispatch, context lifecycle, memory management.
 */
#include <string.h>

#include "internal.h"
#include "pdl/common.h"
#include "pdl/env.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PDL_WITH_CUDA
#define IS_CUDA(ctx) ((ctx)->device == PDL_DEV_CUDA)
#define DISPATCH(ctx, fn, ...)                                                                     \
    do {                                                                                           \
        if (IS_CUDA(ctx))                                                                          \
            pdl_cuda_##fn(__VA_ARGS__);                                                            \
        else                                                                                       \
            pdl_cpu_##fn(__VA_ARGS__);                                                             \
    } while (0)
#define DISPATCH_RET(ctx, fn, ...) (IS_CUDA(ctx) ? pdl_cuda_##fn(__VA_ARGS__) : pdl_cpu_##fn(__VA_ARGS__))
#else
#define IS_CUDA(ctx) 0
#define DISPATCH(ctx, fn, ...) pdl_cpu_##fn(__VA_ARGS__)
#define DISPATCH_RET(ctx, fn, ...) pdl_cpu_##fn(__VA_ARGS__)
#endif

/* ---- capabilities ------------------------------------------------------- */

int pdl_has_cuda(void) {
#ifdef PDL_WITH_CUDA
    return pdl_cuda_device_count() > 0;
#else
    return 0;
#endif
}

int pdl_has_cublas(void) {
#if defined(PDL_WITH_CUDA) && defined(PDL_WITH_CUBLAS)
    return 1;
#else
    return 0;
#endif
}

int pdl_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int pdl_openmp_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static const char *k_impl_names[PDL_IMPL_COUNT] = {
    "cpu-naive", "cpu-blocked", "cpu-omp", "cuda-naive", "cuda-tiled", "cublas",
};

const char *pdl_impl_name(pdl_impl impl) {
    return (impl >= 0 && impl < PDL_IMPL_COUNT) ? k_impl_names[impl] : "?";
}

const char *pdl_device_name(pdl_device dev) { return dev == PDL_DEV_CUDA ? "cuda" : "cpu"; }

int pdl_impl_parse(const char *s, pdl_device dev) {
    if (!s || !*s || strcmp(s, "auto") == 0) {
        if (dev == PDL_DEV_CUDA) return pdl_has_cublas() ? PDL_IMPL_CUDA_CUBLAS : PDL_IMPL_CUDA_TILED;
        return pdl_has_openmp() ? PDL_IMPL_CPU_OMP : PDL_IMPL_CPU_BLOCKED;
    }
    for (int i = 0; i < PDL_IMPL_COUNT; ++i)
        if (strcmp(s, k_impl_names[i]) == 0) return i;
    /* short names relative to the device */
    if (dev == PDL_DEV_CUDA) {
        if (strcmp(s, "naive") == 0) return PDL_IMPL_CUDA_NAIVE;
        if (strcmp(s, "tiled") == 0) return PDL_IMPL_CUDA_TILED;
        if (strcmp(s, "cuda-cublas") == 0) return PDL_IMPL_CUDA_CUBLAS;
    } else {
        if (strcmp(s, "naive") == 0) return PDL_IMPL_CPU_NAIVE;
        if (strcmp(s, "blocked") == 0) return PDL_IMPL_CPU_BLOCKED;
        if (strcmp(s, "omp") == 0) return PDL_IMPL_CPU_OMP;
    }
    return -1;
}

/* ---- context ------------------------------------------------------------- */

void pdl_ctx_init(pdl_ctx *ctx, pdl_device dev, pdl_impl impl, int num_threads, int gpu_id) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->device = dev;
    ctx->impl = impl;
    ctx->num_threads = num_threads;
    ctx->gpu_id = gpu_id < 0 ? 0 : gpu_id;
    ctx->timing_sync = 1;
    if (dev == PDL_DEV_CUDA) {
#ifdef PDL_WITH_CUDA
        PDL_CHECK(impl >= PDL_IMPL_CUDA_NAIVE, "CPU implementation '%s' selected for CUDA device",
                  pdl_impl_name(impl));
        PDL_CHECK(impl != PDL_IMPL_CUDA_CUBLAS || pdl_has_cublas(), "built without cuBLAS");
        pdl_cuda_ctx_init(ctx);
#else
        PDL_FATAL("this build has no CUDA support (reconfigure with -DPDL_ENABLE_CUDA=ON)");
#endif
    } else {
        PDL_CHECK(impl <= PDL_IMPL_CPU_OMP, "CUDA implementation '%s' selected for CPU device",
                  pdl_impl_name(impl));
        if (impl == PDL_IMPL_CPU_OMP && !pdl_has_openmp())
            PDL_WARN("built without OpenMP: cpu-omp runs single-threaded");
        /* The thread count lives in ctx->num_threads and every OpenMP region carries
         * num_threads(nt). Do NOT call omp_set_num_threads() here: it is process-global, so a
         * 1-thread reference context (used by --verify) would silently make every later
         * cpu-omp context in the same process run single-threaded. */
    }
}

void pdl_ctx_destroy(pdl_ctx *ctx) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) pdl_cuda_ctx_destroy(ctx);
#endif
    memset(ctx, 0, sizeof(*ctx));
}

void pdl_ctx_sync(const pdl_ctx *ctx) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) pdl_cuda_sync(ctx);
#else
    PDL_UNUSED(ctx);
#endif
}

void pdl_ctx_timing_sync(const pdl_ctx *ctx) {
    if (ctx->timing_sync) pdl_ctx_sync(ctx);
}

/* ---- memory --------------------------------------------------------------- */

void *pdl_dev_malloc(const pdl_ctx *ctx, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        void *p = pdl_cuda_malloc(bytes);
        pdl_cuda_memset0(ctx, p, bytes);
        return p;
    }
#endif
    PDL_UNUSED(ctx);
    return pdl_xcalloc(1, bytes);
}

void pdl_dev_free(const pdl_ctx *ctx, void *p) {
    if (!p) return;
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_free(p);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    free(p);
}

void pdl_dev_memset0(const pdl_ctx *ctx, void *p, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_memset0(ctx, p, bytes);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    memset(p, 0, bytes);
}

void pdl_memcpy_h2d(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_memcpy(ctx, dst, src, bytes, 1);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    if (dst != src) memcpy(dst, src, bytes);
}

void pdl_memcpy_d2h(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_memcpy(ctx, dst, src, bytes, 2);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    if (dst != src) memcpy(dst, src, bytes);
}

void pdl_memcpy_d2d(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_memcpy(ctx, dst, src, bytes, 3);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    if (dst != src) memmove(dst, src, bytes);
}

void *pdl_host_malloc(const pdl_ctx *ctx, size_t bytes) {
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) return pdl_cuda_host_malloc(bytes);
#endif
    PDL_UNUSED(ctx);
    return pdl_xcalloc(1, bytes);
}

void pdl_host_free(const pdl_ctx *ctx, void *p) {
    if (!p) return;
#ifdef PDL_WITH_CUDA
    if (IS_CUDA(ctx)) {
        pdl_cuda_host_free(p);
        return;
    }
#endif
    PDL_UNUSED(ctx);
    free(p);
}

/* ---- kernels -------------------------------------------------------------- */

void pdl_gemm(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha, const float *A,
              int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if (M == 0 || N == 0) return;
    DISPATCH(ctx, gemm, ctx, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void pdl_gemm_strided_batched(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,
                              const float *A, int lda, long long sa, const float *B, int ldb,
                              long long sb, float beta, float *C, int ldc, long long sc, int batch) {
    if (M == 0 || N == 0 || batch == 0) return;
    DISPATCH(ctx, gemm_strided_batched, ctx, ta, tb, M, N, K, alpha, A, lda, sa, B, ldb, sb, beta, C,
             ldc, sc, batch);
}

void pdl_axpy(const pdl_ctx *ctx, size_t n, float alpha, const float *x, float *y) {
    DISPATCH(ctx, axpby, ctx, n, alpha, x, 1.0f, y);
}

void pdl_axpby(const pdl_ctx *ctx, size_t n, float a, const float *x, float b, float *y) {
    DISPATCH(ctx, axpby, ctx, n, a, x, b, y);
}

void pdl_scal(const pdl_ctx *ctx, size_t n, float alpha, float *x) {
    if (alpha == 1.0f) return;
    DISPATCH(ctx, scal, ctx, n, alpha, x);
}

void pdl_copy(const pdl_ctx *ctx, size_t n, const float *x, float *y) {
    pdl_memcpy_d2d(ctx, y, x, n * sizeof(float));
}

double pdl_sumsq(const pdl_ctx *ctx, size_t n, const float *x) {
    return DISPATCH_RET(ctx, sumsq, ctx, n, x);
}

size_t pdl_count_nonfinite(const pdl_ctx *ctx, size_t n, const float *x) {
    return DISPATCH_RET(ctx, count_nonfinite, ctx, n, x);
}

void pdl_im2col(const pdl_ctx *ctx, const float *im, int N, int C, int H, int W, int kh, int kw,
                int stride, int pad, int OH, int OW, float *col) {
    DISPATCH(ctx, im2col, ctx, im, N, C, H, W, kh, kw, stride, pad, OH, OW, col);
}

void pdl_col2im(const pdl_ctx *ctx, const float *col, int N, int C, int H, int W, int kh, int kw,
                int stride, int pad, int OH, int OW, float *im) {
    DISPATCH(ctx, col2im, ctx, col, N, C, H, W, kh, kw, stride, pad, OH, OW, im);
}

void pdl_conv2d_direct(const pdl_ctx *ctx, const float *in, const float *w, const float *b, int N,
                       int C, int H, int W, int OC, int kh, int kw, int stride, int pad, int OH,
                       int OW, float *out) {
    DISPATCH(ctx, conv2d_direct, ctx, in, w, b, N, C, H, W, OC, kh, kw, stride, pad, OH, OW, out);
}

void pdl_add_bias_nchw(const pdl_ctx *ctx, float *out, const float *bias, int N, int C, int HW) {
    DISPATCH(ctx, add_bias_nchw, ctx, out, bias, N, C, HW);
}

void pdl_bias_grad_nchw(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int C, int HW) {
    DISPATCH(ctx, bias_grad_nchw, ctx, dout, dbias, N, C, HW);
}

/* Dense bias = NCHW bias with HW = 1 and C = D. */
void pdl_add_bias_rows(const pdl_ctx *ctx, float *out, const float *bias, int N, int D) {
    DISPATCH(ctx, add_bias_nchw, ctx, out, bias, N, D, 1);
}

void pdl_bias_grad_rows(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int D) {
    DISPATCH(ctx, bias_grad_nchw, ctx, dout, dbias, N, D, 1);
}

void pdl_relu_forward(const pdl_ctx *ctx, size_t n, const float *in, float *out) {
    DISPATCH(ctx, relu_forward, ctx, n, in, out);
}

void pdl_relu_backward(const pdl_ctx *ctx, size_t n, const float *in, const float *dout, float *din) {
    DISPATCH(ctx, relu_backward, ctx, n, in, dout, din);
}

void pdl_maxpool_forward(const pdl_ctx *ctx, const float *in, int N, int C, int H, int W, int k,
                         int s, int OH, int OW, float *out, int32_t *argmax) {
    DISPATCH(ctx, maxpool_forward, ctx, in, N, C, H, W, k, s, OH, OW, out, argmax);
}

void pdl_maxpool_backward(const pdl_ctx *ctx, const float *dout, const int32_t *argmax, int N,
                          int C, int H, int W, int k, int s, int OH, int OW, float *din) {
    DISPATCH(ctx, maxpool_backward, ctx, dout, argmax, N, C, H, W, k, s, OH, OW, din);
}

double pdl_softmax_xent(const pdl_ctx *ctx, const float *logits, const int32_t *labels, int N,
                        int K, float grad_scale, float *dlogits, int *correct) {
    return DISPATCH_RET(ctx, softmax_xent, ctx, logits, labels, N, K, grad_scale, dlogits, correct);
}

void pdl_sgd_update(const pdl_ctx *ctx, size_t n, float lr, float momentum, float weight_decay,
                    float *w, const float *g, float *v) {
    DISPATCH(ctx, sgd_update, ctx, n, lr, momentum, weight_decay, w, g, v);
}

/* ---- CUDA introspection stubs for CPU-only builds ------------------------- */
#ifndef PDL_WITH_CUDA
int pdl_cuda_device_info(int gpu_id, char *name, size_t name_len, size_t *total_mem, int *sm_count,
                         int *cc_major, int *cc_minor) {
    PDL_UNUSED(gpu_id);
    if (name && name_len) name[0] = '\0';
    if (total_mem) *total_mem = 0;
    if (sm_count) *sm_count = 0;
    if (cc_major) *cc_major = 0;
    if (cc_minor) *cc_minor = 0;
    return 0;
}

int pdl_cuda_versions(int *runtime, int *driver) {
    if (runtime) *runtime = 0;
    if (driver) *driver = 0;
    return 0;
}
#endif
