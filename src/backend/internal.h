/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * internal.h — per-device kernel entry points behind the backend.h dispatch.
 * pdl_cpu_* live in cpu_kernels.c, pdl_cuda_* in cuda_kernels.cu.
 */
#ifndef PDL_BACKEND_INTERNAL_H
#define PDL_BACKEND_INTERNAL_H

#include "pdl/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PDL_DECLARE_KERNELS(P)                                                                     \
    void P##gemm(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,             \
                 const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc); \
    void P##gemm_strided_batched(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K,          \
                                 float alpha, const float *A, int lda, long long sa,               \
                                 const float *B, int ldb, long long sb, float beta, float *C,      \
                                 int ldc, long long sc, int batch);                                \
    void P##axpby(const pdl_ctx *ctx, size_t n, float a, const float *x, float b, float *y);       \
    void P##scal(const pdl_ctx *ctx, size_t n, float a, float *x);                                \
    double P##sumsq(const pdl_ctx *ctx, size_t n, const float *x);                                 \
    size_t P##count_nonfinite(const pdl_ctx *ctx, size_t n, const float *x);                       \
    void P##im2col(const pdl_ctx *ctx, const float *im, int N, int C, int H, int W, int kh,        \
                   int kw, int stride, int pad, int OH, int OW, float *col);                       \
    void P##col2im(const pdl_ctx *ctx, const float *col, int N, int C, int H, int W, int kh,       \
                   int kw, int stride, int pad, int OH, int OW, float *im);                        \
    void P##conv2d_direct(const pdl_ctx *ctx, const float *in, const float *w, const float *b,     \
                          int N, int C, int H, int W, int OC, int kh, int kw, int stride, int pad, \
                          int OH, int OW, float *out);                                             \
    void P##add_bias_nchw(const pdl_ctx *ctx, float *out, const float *bias, int N, int C,         \
                          int HW);                                                                 \
    void P##bias_grad_nchw(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int C,      \
                           int HW);                                                                \
    void P##relu_forward(const pdl_ctx *ctx, size_t n, const float *in, float *out);               \
    void P##relu_backward(const pdl_ctx *ctx, size_t n, const float *in, const float *dout,        \
                          float *din);                                                             \
    void P##maxpool_forward(const pdl_ctx *ctx, const float *in, int N, int C, int H, int W,       \
                            int k, int s, int OH, int OW, float *out, int32_t *argmax);            \
    void P##maxpool_backward(const pdl_ctx *ctx, const float *dout, const int32_t *argmax, int N,  \
                             int C, int H, int W, int k, int s, int OH, int OW, float *din);       \
    double P##softmax_xent(const pdl_ctx *ctx, const float *logits, const int32_t *labels, int N,  \
                           int K, float grad_scale, float *dlogits, int *correct);                 \
    void P##sgd_update(const pdl_ctx *ctx, size_t n, float lr, float momentum, float wd, float *w, \
                       const float *g, float *v);

PDL_DECLARE_KERNELS(pdl_cpu_)

#ifdef PDL_WITH_CUDA
PDL_DECLARE_KERNELS(pdl_cuda_)

/* CUDA runtime glue (cuda_kernels.cu) */
void pdl_cuda_ctx_init(pdl_ctx *ctx);
void pdl_cuda_ctx_destroy(pdl_ctx *ctx);
void pdl_cuda_sync(const pdl_ctx *ctx);
void *pdl_cuda_malloc(size_t bytes);
void pdl_cuda_free(void *p);
void pdl_cuda_memset0(const pdl_ctx *ctx, void *p, size_t bytes);
void pdl_cuda_memcpy(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes, int kind);
void *pdl_cuda_host_malloc(size_t bytes);
void pdl_cuda_host_free(void *p);
int pdl_cuda_device_count(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* PDL_BACKEND_INTERNAL_H */
