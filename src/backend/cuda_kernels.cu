/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * cuda_kernels.cu — CUDA implementations of all numerical kernels.
 *
 * GEMM variants (compared in pdl_bench_compute, paper §3):
 *   cuda-naive : one thread per output element, operands read from global memory
 *   cuda-tiled : 32x32 shared-memory tiles, each thread computes 4 outputs
 *   cublas     : cublasSgemm / cublasSgemmStridedBatched
 * Every other kernel is shared by all variants. All work is issued on one
 * stream per context; host-visible results (losses, reductions) synchronize it.
 *
 * Target: Turing sm_75 (RTX 2060) and newer; requires double atomicAdd (sm_60+).
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef PDL_WITH_CUBLAS
#include <cublas_v2.h>
#endif

#include "internal.h"
#include "pdl/common.h"
#include "pdl/env.h"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess)                                                                  \
            pdl_fatal(__FILE__, __LINE__, "CUDA error %s: %s", #call, cudaGetErrorString(err__));  \
    } while (0)

#define CUDA_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

#ifdef PDL_WITH_CUBLAS
#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t st__ = (call);                                                              \
        if (st__ != CUBLAS_STATUS_SUCCESS)                                                         \
            pdl_fatal(__FILE__, __LINE__, "cuBLAS error %d in %s", (int)st__, #call);             \
    } while (0)
#endif

typedef struct {
    cudaStream_t stream;
#ifdef PDL_WITH_CUBLAS
    cublasHandle_t blas;
#endif
    double *d_acc;            /* device accumulator for reductions */
    unsigned long long *d_cnt;
    double *h_acc;            /* pinned host mirrors */
    unsigned long long *h_cnt;
    double *d_loss;           /* per-sample loss buffer for softmax_xent */
    int *d_hit;
    double *h_loss;
    int *h_hit;
    int loss_cap;
} cuda_state;

static inline cuda_state *ST(const pdl_ctx *ctx) { return (cuda_state *)ctx->cuda; }
static inline cudaStream_t STREAM(const pdl_ctx *ctx) { return ST(ctx)->stream; }

static inline unsigned int grid_for(size_t n, int block) {
    size_t g = (n + (size_t)block - 1) / (size_t)block;
    if (g > 65535u * 32u) g = 65535u * 32u; /* grid-stride loops cover the rest */
    if (g == 0) g = 1;
    return (unsigned int)g;
}

#define GRID_STRIDE(i, n)                                                                          \
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n);                        \
         i += (size_t)blockDim.x * gridDim.x)

/* ======================================================================== */
/* Runtime glue                                                              */
/* ======================================================================== */

extern "C" int pdl_cuda_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError(); /* clear sticky error */
        return 0;
    }
    return n;
}

extern "C" int pdl_cuda_device_info(int gpu_id, char *name, size_t name_len, size_t *total_mem,
                                    int *sm_count, int *cc_major, int *cc_minor) {
    if (pdl_cuda_device_count() <= gpu_id) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, gpu_id) != cudaSuccess) return 0;
    if (name && name_len) snprintf(name, name_len, "%s", p.name);
    if (total_mem) *total_mem = p.totalGlobalMem;
    if (sm_count) *sm_count = p.multiProcessorCount;
    if (cc_major) *cc_major = p.major;
    if (cc_minor) *cc_minor = p.minor;
    return 1;
}

extern "C" int pdl_cuda_versions(int *runtime, int *driver) {
    int r = 0, d = 0;
    cudaRuntimeGetVersion(&r);
    cudaDriverGetVersion(&d);
    if (runtime) *runtime = r;
    if (driver) *driver = d;
    return pdl_cuda_device_count() > 0;
}

extern "C" void pdl_cuda_ctx_init(pdl_ctx *ctx) {
    int count = pdl_cuda_device_count();
    PDL_CHECK(count > 0, "no CUDA device found (driver installed? nvidia-smi works?)");
    PDL_CHECK(ctx->gpu_id < count, "gpu id %d out of range (%d devices)", ctx->gpu_id, count);
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
    cuda_state *s = (cuda_state *)pdl_xcalloc(1, sizeof(cuda_state));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
#ifdef PDL_WITH_CUBLAS
    CUBLAS_CHECK(cublasCreate(&s->blas));
    CUBLAS_CHECK(cublasSetStream(s->blas, s->stream));
    /* Keep FP32 math (no TF32 / tensor-core down-conversion) for comparability. */
    CUBLAS_CHECK(cublasSetMathMode(s->blas, CUBLAS_DEFAULT_MATH));
#endif
    CUDA_CHECK(cudaMalloc(&s->d_acc, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&s->d_cnt, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&s->h_acc, sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&s->h_cnt, sizeof(unsigned long long)));
    ctx->cuda = s;
}

extern "C" void pdl_cuda_ctx_destroy(pdl_ctx *ctx) {
    cuda_state *s = ST(ctx);
    if (!s) return;
    cudaStreamSynchronize(s->stream);
#ifdef PDL_WITH_CUBLAS
    cublasDestroy(s->blas);
#endif
    cudaFree(s->d_acc);
    cudaFree(s->d_cnt);
    cudaFreeHost(s->h_acc);
    cudaFreeHost(s->h_cnt);
    cudaFree(s->d_loss);
    cudaFree(s->d_hit);
    cudaFreeHost(s->h_loss);
    cudaFreeHost(s->h_hit);
    cudaStreamDestroy(s->stream);
    free(s);
    ctx->cuda = NULL;
}

extern "C" void pdl_cuda_sync(const pdl_ctx *ctx) { CUDA_CHECK(cudaStreamSynchronize(STREAM(ctx))); }

extern "C" void *pdl_cuda_malloc(size_t bytes) {
    void *p = NULL;
    cudaError_t e = cudaMalloc(&p, bytes ? bytes : 1);
    if (e != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        PDL_FATAL("cudaMalloc(%zu bytes) failed: %s (free %.1f MiB of %.1f MiB) — reduce batch "
                  "size / width or the number of ranks sharing the GPU",
                  bytes, cudaGetErrorString(e), free_b / 1048576.0, total_b / 1048576.0);
    }
    return p;
}

extern "C" void pdl_cuda_free(void *p) { CUDA_CHECK(cudaFree(p)); }

extern "C" void pdl_cuda_memset0(const pdl_ctx *ctx, void *p, size_t bytes) {
    CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, STREAM(ctx)));
}

/* kind: 1 = H2D, 2 = D2H, 3 = D2D. H2D/D2H block until complete so callers may
 * reuse / read host buffers immediately. */
extern "C" void pdl_cuda_memcpy(const pdl_ctx *ctx, void *dst, const void *src, size_t bytes,
                                int kind) {
    if (bytes == 0) return;
    cudaMemcpyKind k = kind == 1 ? cudaMemcpyHostToDevice
                                 : (kind == 2 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, k, STREAM(ctx)));
    if (kind != 3) CUDA_CHECK(cudaStreamSynchronize(STREAM(ctx)));
}

extern "C" void *pdl_cuda_host_malloc(size_t bytes) {
    void *p = NULL;
    CUDA_CHECK(cudaMallocHost(&p, bytes ? bytes : 1));
    memset(p, 0, bytes);
    return p;
}

extern "C" void pdl_cuda_host_free(void *p) { CUDA_CHECK(cudaFreeHost(p)); }

/* ======================================================================== */
/* GEMM                                                                      */
/* ======================================================================== */

#define LA(i, k) (ta ? A[(size_t)(k) * lda + (i)] : A[(size_t)(i) * lda + (k)])
#define LB(k, j) (tb ? B[(size_t)(j) * ldb + (k)] : B[(size_t)(k) * ldb + (j)])

__global__ void k_gemm_naive(int ta, int tb, int M, int N, int K, float alpha, const float *A,
                             int lda, long long sa, const float *B, int ldb, long long sb,
                             float beta, float *C, int ldc, long long sc) {
    A += blockIdx.z * sa;
    B += blockIdx.z * sb;
    C += blockIdx.z * sc;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += LA(i, k) * LB(k, j);
    float *c = &C[(size_t)i * ldc + j];
    *c = alpha * acc + (beta == 0.0f ? 0.0f : beta * *c);
}

#define TILE 32
#define TROWS 8 /* blockDim = (TILE, TROWS); each thread computes TILE/TROWS outputs */

__global__ void k_gemm_tiled(int ta, int tb, int M, int N, int K, float alpha, const float *A,
                             int lda, long long sa, const float *B, int ldb, long long sb,
                             float beta, float *C, int ldc, long long sc) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    A += blockIdx.z * sa;
    B += blockIdx.z * sb;
    C += blockIdx.z * sc;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row0 = blockIdx.y * TILE, col0 = blockIdx.x * TILE;
    float acc[TILE / TROWS];
#pragma unroll
    for (int r = 0; r < TILE / TROWS; ++r) acc[r] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
#pragma unroll
        for (int r = 0; r < TILE / TROWS; ++r) {
            const int rr = ty + r * TROWS;
            const int i = row0 + rr, ka = k0 + tx;
            As[rr][tx] = (i < M && ka < K) ? LA(i, ka) : 0.0f;
            const int kb = k0 + rr, j = col0 + tx;
            Bs[rr][tx] = (kb < K && j < N) ? LB(kb, j) : 0.0f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TILE; ++kk) {
            const float b = Bs[kk][tx];
#pragma unroll
            for (int r = 0; r < TILE / TROWS; ++r) acc[r] += As[ty + r * TROWS][kk] * b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < TILE / TROWS; ++r) {
        const int i = row0 + ty + r * TROWS, j = col0 + tx;
        if (i < M && j < N) {
            float *c = &C[(size_t)i * ldc + j];
            *c = alpha * acc[r] + (beta == 0.0f ? 0.0f : beta * *c);
        }
    }
}

static void launch_gemm(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,
                        const float *A, int lda, long long sa, const float *B, int ldb,
                        long long sb, float beta, float *C, int ldc, long long sc, int batch) {
    cudaStream_t st = STREAM(ctx);
    switch (ctx->impl) {
    case PDL_IMPL_CUDA_NAIVE: {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16, batch);
        k_gemm_naive<<<grid, block, 0, st>>>(ta, tb, M, N, K, alpha, A, lda, sa, B, ldb, sb, beta,
                                             C, ldc, sc);
        CUDA_LAUNCH_CHECK();
        break;
    }
    case PDL_IMPL_CUDA_TILED: {
        dim3 block(TILE, TROWS);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
        k_gemm_tiled<<<grid, block, 0, st>>>(ta, tb, M, N, K, alpha, A, lda, sa, B, ldb, sb, beta,
                                             C, ldc, sc);
        CUDA_LAUNCH_CHECK();
        break;
    }
    case PDL_IMPL_CUDA_CUBLAS: {
#ifdef PDL_WITH_CUBLAS
        /* Row-major C = op(A) op(B)  <=>  column-major C^T = op(B)^T op(A)^T. */
        cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
        if (batch == 1)
            CUBLAS_CHECK(cublasSgemm(ST(ctx)->blas, opB, opA, N, M, K, &alpha, B, ldb, A, lda,
                                     &beta, C, ldc));
        else
            CUBLAS_CHECK(cublasSgemmStridedBatched(ST(ctx)->blas, opB, opA, N, M, K, &alpha, B, ldb,
                                                   sb, A, lda, sa, &beta, C, ldc, sc, batch));
#else
        PDL_FATAL("built without cuBLAS");
#endif
        break;
    }
    default: PDL_FATAL("invalid CUDA GEMM implementation %d", (int)ctx->impl);
    }
}

extern "C" void pdl_cuda_gemm(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,
                              const float *A, int lda, const float *B, int ldb, float beta,
                              float *C, int ldc) {
    launch_gemm(ctx, ta, tb, M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}

extern "C" void pdl_cuda_gemm_strided_batched(const pdl_ctx *ctx, int ta, int tb, int M, int N,
                                              int K, float alpha, const float *A, int lda,
                                              long long sa, const float *B, int ldb, long long sb,
                                              float beta, float *C, int ldc, long long sc,
                                              int batch) {
    if (sc == 0) {
        /* Accumulating into a shared output: serialize to avoid write races. */
        for (int b = 0; b < batch; ++b)
            launch_gemm(ctx, ta, tb, M, N, K, alpha, A + b * sa, lda, 0, B + b * sb, ldb, 0,
                        b == 0 ? beta : 1.0f, C, ldc, 0, 1);
        return;
    }
    launch_gemm(ctx, ta, tb, M, N, K, alpha, A, lda, sa, B, ldb, sb, beta, C, ldc, sc, batch);
}

/* ======================================================================== */
/* BLAS-1 and reductions                                                     */
/* ======================================================================== */

__global__ void k_axpby(size_t n, float a, const float *x, float b, float *y) {
    GRID_STRIDE(i, n) y[i] = a * x[i] + (b == 0.0f ? 0.0f : b * y[i]);
}

__global__ void k_scal(size_t n, float a, float *x) {
    GRID_STRIDE(i, n) x[i] *= a;
}

__global__ void k_sumsq(size_t n, const float *x, double *acc) {
    __shared__ double sh[256];
    double s = 0.0;
    GRID_STRIDE(i, n) s += (double)x[i] * (double)x[i];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
}

__global__ void k_count_nonfinite(size_t n, const float *x, unsigned long long *cnt) {
    unsigned long long c = 0;
    GRID_STRIDE(i, n) c += !isfinite(x[i]);
    if (c) atomicAdd(cnt, c);
}

extern "C" void pdl_cuda_axpby(const pdl_ctx *ctx, size_t n, float a, const float *x, float b,
                               float *y) {
    if (!n) return;
    k_axpby<<<grid_for(n, 256), 256, 0, STREAM(ctx)>>>(n, a, x, b, y);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_scal(const pdl_ctx *ctx, size_t n, float a, float *x) {
    if (!n) return;
    k_scal<<<grid_for(n, 256), 256, 0, STREAM(ctx)>>>(n, a, x);
    CUDA_LAUNCH_CHECK();
}

extern "C" double pdl_cuda_sumsq(const pdl_ctx *ctx, size_t n, const float *x) {
    cuda_state *s = ST(ctx);
    CUDA_CHECK(cudaMemsetAsync(s->d_acc, 0, sizeof(double), s->stream));
    if (n) {
        unsigned int g = grid_for(n, 256);
        if (g > 1024) g = 1024;
        k_sumsq<<<g, 256, 0, s->stream>>>(n, x, s->d_acc);
        CUDA_LAUNCH_CHECK();
    }
    CUDA_CHECK(cudaMemcpyAsync(s->h_acc, s->d_acc, sizeof(double), cudaMemcpyDeviceToHost, s->stream));
    CUDA_CHECK(cudaStreamSynchronize(s->stream));
    return *s->h_acc;
}

extern "C" size_t pdl_cuda_count_nonfinite(const pdl_ctx *ctx, size_t n, const float *x) {
    cuda_state *s = ST(ctx);
    CUDA_CHECK(cudaMemsetAsync(s->d_cnt, 0, sizeof(unsigned long long), s->stream));
    if (n) {
        unsigned int g = grid_for(n, 256);
        if (g > 1024) g = 1024;
        k_count_nonfinite<<<g, 256, 0, s->stream>>>(n, x, s->d_cnt);
        CUDA_LAUNCH_CHECK();
    }
    CUDA_CHECK(cudaMemcpyAsync(s->h_cnt, s->d_cnt, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, s->stream));
    CUDA_CHECK(cudaStreamSynchronize(s->stream));
    return (size_t)*s->h_cnt;
}

/* ======================================================================== */
/* Convolution helpers                                                       */
/* ======================================================================== */

/* One thread per (n, c, oh, ow): writes kh*kw column entries. */
__global__ void k_im2col(const float *im, int N, int C, int H, int W, int kh, int kw, int stride,
                         int pad, int OH, int OW, float *col) {
    const size_t total = (size_t)N * C * OH * OW;
    const size_t ohw = (size_t)OH * OW;
    GRID_STRIDE(idx, total) {
        int ow = (int)(idx % OW);
        int oh = (int)((idx / OW) % OH);
        int c = (int)((idx / ohw) % C);
        int n = (int)(idx / (ohw * C));
        const float *plane = im + ((size_t)n * C + c) * H * W;
        float *dst = col + ((size_t)n * C * kh * kw + (size_t)c * kh * kw) * ohw + (size_t)oh * OW + ow;
        for (int ki = 0; ki < kh; ++ki) {
            int ih = oh * stride - pad + ki;
            for (int kj = 0; kj < kw; ++kj) {
                int iw = ow * stride - pad + kj;
                float v = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? plane[(size_t)ih * W + iw] : 0.0f;
                dst[(size_t)(ki * kw + kj) * ohw] = v;
            }
        }
    }
}

/* One thread per input pixel: gathers all column entries that map to it (no atomics). */
__global__ void k_col2im(const float *col, int N, int C, int H, int W, int kh, int kw, int stride,
                         int pad, int OH, int OW, float *im) {
    const size_t total = (size_t)N * C * H * W;
    const size_t ohw = (size_t)OH * OW;
    GRID_STRIDE(idx, total) {
        int w = (int)(idx % W);
        int h = (int)((idx / W) % H);
        int c = (int)((idx / ((size_t)H * W)) % C);
        int n = (int)(idx / ((size_t)H * W * C));
        const float *cn = col + (size_t)n * C * kh * kw * ohw;
        float acc = 0.0f;
        for (int ki = 0; ki < kh; ++ki) {
            int t = h + pad - ki;
            if (t < 0 || t % stride) continue;
            int oh = t / stride;
            if (oh >= OH) continue;
            for (int kj = 0; kj < kw; ++kj) {
                int u = w + pad - kj;
                if (u < 0 || u % stride) continue;
                int ow = u / stride;
                if (ow >= OW) continue;
                acc += cn[((size_t)(c * kh + ki) * kw + kj) * ohw + (size_t)oh * OW + ow];
            }
        }
        im[idx] += acc;
    }
}

__global__ void k_conv_direct(const float *in, const float *w, const float *b, int N, int C, int H,
                              int W, int OC, int kh, int kw, int stride, int pad, int OH, int OW,
                              float *out) {
    const size_t total = (size_t)N * OC * OH * OW;
    GRID_STRIDE(idx, total) {
        int ow = (int)(idx % OW);
        int oh = (int)((idx / OW) % OH);
        int oc = (int)((idx / ((size_t)OH * OW)) % OC);
        int n = (int)(idx / ((size_t)OH * OW * OC));
        float acc = b ? b[oc] : 0.0f;
        const float *wk = w + (size_t)oc * C * kh * kw;
        for (int c = 0; c < C; ++c) {
            const float *plane = in + ((size_t)n * C + c) * H * W;
            for (int ki = 0; ki < kh; ++ki) {
                int ih = oh * stride - pad + ki;
                if (ih < 0 || ih >= H) continue;
                for (int kj = 0; kj < kw; ++kj) {
                    int iw = ow * stride - pad + kj;
                    if (iw < 0 || iw >= W) continue;
                    acc += plane[(size_t)ih * W + iw] * wk[(c * kh + ki) * kw + kj];
                }
            }
        }
        out[idx] = acc;
    }
}

extern "C" void pdl_cuda_im2col(const pdl_ctx *ctx, const float *im, int N, int C, int H, int W,
                                int kh, int kw, int stride, int pad, int OH, int OW, float *col) {
    size_t total = (size_t)N * C * OH * OW;
    k_im2col<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(im, N, C, H, W, kh, kw, stride, pad, OH,
                                                            OW, col);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_col2im(const pdl_ctx *ctx, const float *col, int N, int C, int H, int W,
                                int kh, int kw, int stride, int pad, int OH, int OW, float *im) {
    size_t total = (size_t)N * C * H * W;
    k_col2im<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(col, N, C, H, W, kh, kw, stride, pad,
                                                            OH, OW, im);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_conv2d_direct(const pdl_ctx *ctx, const float *in, const float *w,
                                       const float *b, int N, int C, int H, int W, int OC, int kh,
                                       int kw, int stride, int pad, int OH, int OW, float *out) {
    size_t total = (size_t)N * OC * OH * OW;
    k_conv_direct<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(in, w, b, N, C, H, W, OC, kh, kw,
                                                                 stride, pad, OH, OW, out);
    CUDA_LAUNCH_CHECK();
}

/* ======================================================================== */
/* Bias, activation, pooling                                                 */
/* ======================================================================== */

__global__ void k_add_bias(float *out, const float *bias, size_t total, int C, int HW) {
    GRID_STRIDE(i, total) out[i] += bias[(i / HW) % C];
}

/* One block per channel; deterministic tree reduction. */
__global__ void k_bias_grad(const float *dout, float *dbias, int N, int C, int HW) {
    __shared__ float sh[256];
    const int c = blockIdx.x;
    float s = 0.0f;
    for (size_t t = threadIdx.x; t < (size_t)N * HW; t += blockDim.x) {
        size_t n = t / HW, i = t % HW;
        s += dout[(n * C + c) * HW + i];
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) dbias[c] += sh[0];
}

__global__ void k_relu_fwd(size_t n, const float *in, float *out) {
    GRID_STRIDE(i, n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void k_relu_bwd(size_t n, const float *in, const float *dout, float *din) {
    GRID_STRIDE(i, n) din[i] = in[i] > 0.0f ? dout[i] : 0.0f;
}

__global__ void k_maxpool_fwd(const float *in, int N, int C, int H, int W, int k, int s, int OH,
                              int OW, float *out, int32_t *argmax) {
    const size_t total = (size_t)N * C * OH * OW;
    GRID_STRIDE(idx, total) {
        int ow = (int)(idx % OW);
        int oh = (int)((idx / OW) % OH);
        size_t p = idx / ((size_t)OH * OW);
        size_t base = p * H * W;
        int h0 = oh * s, w0 = ow * s;
        int best = h0 * W + w0;
        float bv = in[base + best];
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                int h = h0 + i, w = w0 + j;
                if (h >= H || w >= W) continue;
                float v = in[base + h * W + w];
                if (v > bv) {
                    bv = v;
                    best = h * W + w;
                }
            }
        out[idx] = bv;
        argmax[idx] = (int32_t)(base + best);
    }
}

/* Gather formulation: one thread per input pixel, deterministic even when
 * pooling windows overlap (k > s). */
__global__ void k_maxpool_bwd(const float *dout, const int32_t *argmax, int N, int C, int H, int W,
                              int k, int s, int OH, int OW, float *din) {
    const size_t total = (size_t)N * C * H * W;
    GRID_STRIDE(idx, total) {
        int w = (int)(idx % W);
        int h = (int)((idx / W) % H);
        size_t p = idx / ((size_t)H * W);
        int oh0 = h - k + 1 > 0 ? (h - k + s) / s : 0;
        int ow0 = w - k + 1 > 0 ? (w - k + s) / s : 0;
        int oh1 = min(h / s, OH - 1), ow1 = min(w / s, OW - 1);
        float acc = 0.0f;
        for (int oh = oh0; oh <= oh1; ++oh)
            for (int ow = ow0; ow <= ow1; ++ow) {
                size_t o = p * OH * OW + (size_t)oh * OW + ow;
                if ((size_t)argmax[o] == idx) acc += dout[o];
            }
        din[idx] += acc;
    }
}

extern "C" void pdl_cuda_add_bias_nchw(const pdl_ctx *ctx, float *out, const float *bias, int N,
                                       int C, int HW) {
    size_t total = (size_t)N * C * HW;
    k_add_bias<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(out, bias, total, C, HW);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_bias_grad_nchw(const pdl_ctx *ctx, const float *dout, float *dbias, int N,
                                        int C, int HW) {
    k_bias_grad<<<C, 256, 0, STREAM(ctx)>>>(dout, dbias, N, C, HW);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_relu_forward(const pdl_ctx *ctx, size_t n, const float *in, float *out) {
    k_relu_fwd<<<grid_for(n, 256), 256, 0, STREAM(ctx)>>>(n, in, out);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_relu_backward(const pdl_ctx *ctx, size_t n, const float *in,
                                       const float *dout, float *din) {
    k_relu_bwd<<<grid_for(n, 256), 256, 0, STREAM(ctx)>>>(n, in, dout, din);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_maxpool_forward(const pdl_ctx *ctx, const float *in, int N, int C, int H,
                                         int W, int k, int s, int OH, int OW, float *out,
                                         int32_t *argmax) {
    size_t total = (size_t)N * C * OH * OW;
    k_maxpool_fwd<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(in, N, C, H, W, k, s, OH, OW, out,
                                                                 argmax);
    CUDA_LAUNCH_CHECK();
}

extern "C" void pdl_cuda_maxpool_backward(const pdl_ctx *ctx, const float *dout,
                                          const int32_t *argmax, int N, int C, int H, int W, int k,
                                          int s, int OH, int OW, float *din) {
    size_t total = (size_t)N * C * H * W;
    k_maxpool_bwd<<<grid_for(total, 256), 256, 0, STREAM(ctx)>>>(dout, argmax, N, C, H, W, k, s, OH,
                                                                 OW, din);
    CUDA_LAUNCH_CHECK();
}

/* ======================================================================== */
/* Loss and optimizer                                                        */
/* ======================================================================== */

/* One thread per sample (K is small: number of classes). */
__global__ void k_softmax_xent(const float *logits, const int32_t *labels, int N, int K,
                               float grad_scale, float *dlogits, double *loss, int *hit) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *f = logits + (size_t)n * K;
    int y = labels[n];
    float mx = f[0];
    int am = 0;
    for (int k = 1; k < K; ++k)
        if (f[k] > mx) {
            mx = f[k];
            am = k;
        }
    double sum = 0.0;
    for (int k = 0; k < K; ++k) sum += exp((double)(f[k] - mx));
    loss[n] = log(sum) - (double)(f[y] - mx);
    hit[n] = (am == y);
    if (dlogits) {
        float *d = dlogits + (size_t)n * K;
        for (int k = 0; k < K; ++k) {
            float p = (float)(exp((double)(f[k] - mx)) / sum);
            d[k] = (p - (k == y ? 1.0f : 0.0f)) * grad_scale;
        }
    }
}

extern "C" double pdl_cuda_softmax_xent(const pdl_ctx *ctx, const float *logits,
                                        const int32_t *labels, int N, int K, float grad_scale,
                                        float *dlogits, int *correct) {
    cuda_state *s = ST(ctx);
    if (N > s->loss_cap) {
        cudaFree(s->d_loss);
        cudaFree(s->d_hit);
        cudaFreeHost(s->h_loss);
        cudaFreeHost(s->h_hit);
        CUDA_CHECK(cudaMalloc(&s->d_loss, (size_t)N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s->d_hit, (size_t)N * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&s->h_loss, (size_t)N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&s->h_hit, (size_t)N * sizeof(int)));
        s->loss_cap = N;
    }
    k_softmax_xent<<<(N + 127) / 128, 128, 0, s->stream>>>(logits, labels, N, K, grad_scale,
                                                           dlogits, s->d_loss, s->d_hit);
    CUDA_LAUNCH_CHECK();
    CUDA_CHECK(cudaMemcpyAsync(s->h_loss, s->d_loss, (size_t)N * sizeof(double),
                               cudaMemcpyDeviceToHost, s->stream));
    CUDA_CHECK(cudaMemcpyAsync(s->h_hit, s->d_hit, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost,
                               s->stream));
    CUDA_CHECK(cudaStreamSynchronize(s->stream));
    double loss = 0.0;
    int hits = 0;
    for (int n = 0; n < N; ++n) {
        loss += s->h_loss[n];
        hits += s->h_hit[n];
    }
    if (correct) *correct = hits;
    return loss;
}

__global__ void k_sgd(size_t n, float lr, float mu, float wd, float *w, const float *g, float *v) {
    GRID_STRIDE(i, n) {
        float grad = g[i] + wd * w[i];
        if (v) {
            v[i] = mu * v[i] + grad;
            grad = v[i];
        }
        w[i] -= lr * grad;
    }
}

extern "C" void pdl_cuda_sgd_update(const pdl_ctx *ctx, size_t n, float lr, float momentum,
                                    float wd, float *w, const float *g, float *v) {
    if (!n) return;
    k_sgd<<<grid_for(n, 256), 256, 0, STREAM(ctx)>>>(n, lr, momentum, wd, w, g,
                                                     momentum == 0.0f ? NULL : v);
    CUDA_LAUNCH_CHECK();
}
