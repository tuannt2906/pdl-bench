/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * backend.h — compute backend abstraction (CPU / CUDA).
 *
 * Every numerical kernel takes a pdl_ctx that selects the device and the
 * implementation variant. Buffers passed to kernels must live on ctx->device
 * (host memory for CPU, device memory for CUDA). All matrices are row-major.
 *
 * Implementation variants (benchmarked against each other, paper §3):
 *   CPU : naive (triple loop), blocked (cache tiled), omp (blocked + OpenMP)
 *   CUDA: naive (1 thread / output), tiled (shared-memory tiles), cublas
 */
#ifndef PDL_BACKEND_H
#define PDL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PDL_DEV_CPU = 0, PDL_DEV_CUDA = 1 } pdl_device;

typedef enum {
    PDL_IMPL_CPU_NAIVE = 0,
    PDL_IMPL_CPU_BLOCKED,
    PDL_IMPL_CPU_OMP,
    PDL_IMPL_CUDA_NAIVE,
    PDL_IMPL_CUDA_TILED,
    PDL_IMPL_CUDA_CUBLAS,
    PDL_IMPL_COUNT
} pdl_impl;

typedef struct pdl_ctx {
    pdl_device device;
    pdl_impl impl;       /* GEMM / kernel variant */
    int num_threads;     /* OpenMP threads (CPU); <=0 → runtime default */
    int gpu_id;          /* CUDA device ordinal */
    int timing_sync;     /* if non-zero, synchronize device at phase boundaries */
    void *cuda;          /* opaque CUDA state (stream, cuBLAS handle) */
    float *scratch;      /* device scratch for reductions (CUDA) */
    size_t scratch_len;
} pdl_ctx;

/* ---- capability queries ---------------------------------------------- */
int pdl_has_cuda(void);   /* compiled with CUDA support and a device is present */
int pdl_has_cublas(void); /* compiled with cuBLAS */
int pdl_has_openmp(void);
int pdl_openmp_max_threads(void);

const char *pdl_impl_name(pdl_impl impl);
const char *pdl_device_name(pdl_device dev);
/* Parses "cpu-naive", "naive", "cublas", ... ; returns -1 on failure. */
int pdl_impl_parse(const char *s, pdl_device dev);

/* ---- context lifecycle ----------------------------------------------- */
void pdl_ctx_init(pdl_ctx *ctx, pdl_device dev, pdl_impl impl, int num_threads, int gpu_id);
void pdl_ctx_destroy(pdl_ctx *ctx);
void pdl_ctx_sync(const pdl_ctx *ctx);        /* wait for all queued device work */
void pdl_ctx_timing_sync(const pdl_ctx *ctx); /* sync only when ctx->timing_sync */

/* ---- memory ------------------------------------------------------------ */
void *pdl_dev_malloc(const pdl_ctx *ctx, size_t bytes); /* zero-initialised */
void pdl_dev_free(const pdl_ctx *ctx, void *p);
void pdl_dev_memset0(const pdl_ctx *ctx, void *p, size_t bytes);
void pdl_memcpy_h2d(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes);
void pdl_memcpy_d2h(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes);
void pdl_memcpy_d2d(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes);
/* Page-locked host memory when CUDA is active (faster H2D/D2H), malloc otherwise. */
void *pdl_host_malloc(const pdl_ctx *ctx, size_t bytes);
void pdl_host_free(const pdl_ctx *ctx, void *p);

/* ---- BLAS-like kernels -------------------------------------------------- */
/* C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C ; row-major. */
void pdl_gemm(const pdl_ctx *ctx, int trans_a, int trans_b, int M, int N, int K, float alpha,
              const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
/* Batched GEMM with element strides between consecutive matrices (stride 0 = shared). */
void pdl_gemm_strided_batched(const pdl_ctx *ctx, int trans_a, int trans_b, int M, int N, int K,
                              float alpha, const float *A, int lda, long long stride_a,
                              const float *B, int ldb, long long stride_b, float beta, float *C,
                              int ldc, long long stride_c, int batch);
void pdl_axpy(const pdl_ctx *ctx, size_t n, float alpha, const float *x, float *y); /* y += a x */
void pdl_scal(const pdl_ctx *ctx, size_t n, float alpha, float *x);                 /* x *= a   */
void pdl_copy(const pdl_ctx *ctx, size_t n, const float *x, float *y);              /* y = x    */
/* y = a*x + b*y */
void pdl_axpby(const pdl_ctx *ctx, size_t n, float a, const float *x, float b, float *y);
/* Returns sum of squares (double accumulation). */
double pdl_sumsq(const pdl_ctx *ctx, size_t n, const float *x);
/* Returns number of NaN/Inf entries. */
size_t pdl_count_nonfinite(const pdl_ctx *ctx, size_t n, const float *x);

/* ---- CNN kernels -------------------------------------------------------- */
/* im2col over a batch: im [N][C][H][W] -> col [N][C*kh*kw][OH*OW]. */
void pdl_im2col(const pdl_ctx *ctx, const float *im, int N, int C, int H, int W, int kh, int kw,
                int stride, int pad, int OH, int OW, float *col);
/* col2im (accumulating, im must be zeroed by caller): inverse scatter of im2col. */
void pdl_col2im(const pdl_ctx *ctx, const float *col, int N, int C, int H, int W, int kh, int kw,
                int stride, int pad, int OH, int OW, float *im);
/* Direct (non-GEMM) convolution forward; reference / benchmark baseline. */
void pdl_conv2d_direct(const pdl_ctx *ctx, const float *in, const float *w, const float *b, int N,
                       int C, int H, int W, int OC, int kh, int kw, int stride, int pad, int OH,
                       int OW, float *out);
/* out[n][c][hw] += bias[c] */
void pdl_add_bias_nchw(const pdl_ctx *ctx, float *out, const float *bias, int N, int C, int HW);
/* dbias[c] += sum_{n,hw} dout[n][c][hw] */
void pdl_bias_grad_nchw(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int C, int HW);
/* Dense: out[n][j] += bias[j] ; dbias[j] += sum_n dout[n][j] */
void pdl_add_bias_rows(const pdl_ctx *ctx, float *out, const float *bias, int N, int D);
void pdl_bias_grad_rows(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int D);
void pdl_relu_forward(const pdl_ctx *ctx, size_t n, const float *in, float *out);
/* din = dout * (in > 0) */
void pdl_relu_backward(const pdl_ctx *ctx, size_t n, const float *in, const float *dout, float *din);
/* Max pooling (kernel k, stride s, no padding). argmax stores flat input index per output. */
void pdl_maxpool_forward(const pdl_ctx *ctx, const float *in, int N, int C, int H, int W, int k,
                         int s, int OH, int OW, float *out, int32_t *argmax);
/* din must be zeroed by caller. */
void pdl_maxpool_backward(const pdl_ctx *ctx, const float *dout, const int32_t *argmax, int N,
                          int C, int H, int W, int k, int s, int OH, int OW, float *din);
/*
 * Softmax + cross-entropy (paper eq. 2) over logits [N][K] with int32 labels.
 * Writes dlogits = (softmax - onehot) * grad_scale. Returns summed loss over the
 * batch; *correct receives the number of argmax hits. Labels live on ctx->device.
 */
double pdl_softmax_xent(const pdl_ctx *ctx, const float *logits, const int32_t *labels, int N,
                        int K, float grad_scale, float *dlogits, int *correct);

/* ---- optimizer kernels -------------------------------------------------- */
/* v = mu*v + (g + wd*w) ; w -= lr * v      (mu = 0 → plain SGD) */
void pdl_sgd_update(const pdl_ctx *ctx, size_t n, float lr, float momentum, float weight_decay,
                    float *w, const float *g, float *v);

#ifdef __cplusplus
}
#endif

#endif /* PDL_BACKEND_H */
