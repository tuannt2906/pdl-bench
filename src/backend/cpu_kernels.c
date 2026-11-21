/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * cpu_kernels.c — CPU implementations of all numerical kernels.
 *
 * Three GEMM variants are provided so their performance can be compared
 * (paper §3, "use multiple cores to process multiple images at once"):
 *   naive   : textbook triple loop, no blocking
 *   blocked : cache-blocked loops ordered for unit-stride inner loops
 *   omp     : blocked + OpenMP over output rows / batch images
 * Element-wise kernels use OpenMP only in the `omp` variant.
 */
#include <math.h>
#include <string.h>

#include "internal.h"
#include "pdl/common.h"

#ifdef _OPENMP
#include <omp.h>
#define PDL_OMP(...) _Pragma(#__VA_ARGS__)
#else
#define PDL_OMP(...)
#endif

/* Number of threads to use for a kernel: 1 unless the OpenMP variant is selected. */
static int cpu_threads(const pdl_ctx *ctx) {
#ifdef _OPENMP
    if (ctx->impl != PDL_IMPL_CPU_OMP) return 1;
    return ctx->num_threads > 0 ? ctx->num_threads : omp_get_max_threads();
#else
    PDL_UNUSED(ctx);
    return 1;
#endif
}

/* Element-wise loops only go parallel above this size (thread start-up cost). */
#define PDL_PAR_MIN 32768

/* ------------------------------------------------------------------------ */
/* GEMM                                                                      */
/* ------------------------------------------------------------------------ */

#define OPA(i, k) (ta ? A[(size_t)(k) * lda + (i)] : A[(size_t)(i) * lda + (k)])
#define OPB(k, j) (tb ? B[(size_t)(j) * ldb + (k)] : B[(size_t)(k) * ldb + (j)])

static void scale_rows(int M0, int M1, int N, float beta, float *C, int ldc) {
    for (int i = M0; i < M1; ++i) {
        float *c = C + (size_t)i * ldc;
        if (beta == 0.0f)
            memset(c, 0, (size_t)N * sizeof(float));
        else if (beta != 1.0f)
            for (int j = 0; j < N; ++j) c[j] *= beta;
    }
}

static void gemm_naive(int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda,
                       const float *B, int ldb, float beta, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += OPA(i, k) * OPB(k, j);
            float *c = &C[(size_t)i * ldc + j];
            *c = alpha * acc + (beta == 0.0f ? 0.0f : beta * *c);
        }
    }
}

/* 8-way unrolled dot product (lets the compiler vectorize without -ffast-math). */
static inline float dot8(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        s0 += a[k] * b[k];
        s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2];
        s3 += a[k + 3] * b[k + 3];
        s4 += a[k + 4] * b[k + 4];
        s5 += a[k + 5] * b[k + 5];
        s6 += a[k + 6] * b[k + 6];
        s7 += a[k + 7] * b[k + 7];
    }
    float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
}

#define MB 32
#define KB 256
#define NB 1024

/* Blocked GEMM restricted to output rows [M0, M1). C must already be scaled by beta. */
static void gemm_blocked_rows(int ta, int tb, int M0, int M1, int N, int K, float alpha,
                              const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    if (!tb) {
        /* op(B) rows are contiguous: C[i, j0:j1] += a_ik * B[k, j0:j1] */
        for (int i0 = M0; i0 < M1; i0 += MB) {
            int i1 = PDL_MIN(i0 + MB, M1);
            for (int k0 = 0; k0 < K; k0 += KB) {
                int k1 = PDL_MIN(k0 + KB, K);
                for (int j0 = 0; j0 < N; j0 += NB) {
                    int j1 = PDL_MIN(j0 + NB, N);
                    for (int i = i0; i < i1; ++i) {
                        float *c = C + (size_t)i * ldc;
                        for (int k = k0; k < k1; ++k) {
                            const float a = alpha * OPA(i, k);
                            const float *b = B + (size_t)k * ldb;
                            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
                        }
                    }
                }
            }
        }
    } else if (!ta) {
        /* A row i and B row j are both contiguous over k: dot products. */
        const int JB = 64;
        for (int i0 = M0; i0 < M1; i0 += MB) {
            int i1 = PDL_MIN(i0 + MB, M1);
            for (int j0 = 0; j0 < N; j0 += JB) {
                int j1 = PDL_MIN(j0 + JB, N);
                for (int i = i0; i < i1; ++i) {
                    const float *a = A + (size_t)i * lda;
                    float *c = C + (size_t)i * ldc;
                    for (int j = j0; j < j1; ++j) c[j] += alpha * dot8(a, B + (size_t)j * ldb, K);
                }
            }
        }
    } else {
        /* op(A) = A^T, op(B) = B^T: uncommon, generic loop. */
        for (int i = M0; i < M1; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += OPA(i, k) * OPB(k, j);
                C[(size_t)i * ldc + j] += alpha * acc;
            }
    }
}

static void gemm_serial(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,
                        const float *A, int lda, const float *B, int ldb, float beta, float *C,
                        int ldc) {
    if (ctx->impl == PDL_IMPL_CPU_NAIVE) {
        gemm_naive(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    scale_rows(0, M, N, beta, C, ldc);
    gemm_blocked_rows(ta, tb, 0, M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

void pdl_cpu_gemm(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K, float alpha,
                  const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    int nt = cpu_threads(ctx);
    if (nt <= 1 || (double)M * N * K < 1e5) {
        gemm_serial(ctx, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    /* Parallel over row blocks of C — each thread owns disjoint rows (no races). */
    int nblocks = (M + MB - 1) / MB;
    if (nblocks >= nt) {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static))
        for (int blk = 0; blk < nblocks; ++blk) {
            int i0 = blk * MB, i1 = PDL_MIN(i0 + MB, M);
            scale_rows(i0, i1, N, beta, C, ldc);
            gemm_blocked_rows(ta, tb, i0, i1, N, K, alpha, A, lda, B, ldb, C, ldc);
        }
    } else {
        /* Few rows (e.g. batch-1 FC layer): split the columns instead. */
        scale_rows(0, M, N, beta, C, ldc);
        int chunk = (N + nt - 1) / nt;
        PDL_OMP(omp parallel for num_threads(nt) schedule(static))
        for (int t = 0; t < nt; ++t) {
            int j0 = t * chunk, j1 = PDL_MIN(j0 + chunk, N);
            if (j0 >= j1) continue;
            /* Sub-matrix view: columns [j0, j1) of op(B) and C. */
            const float *Bs = tb ? B + (size_t)j0 * ldb : B + j0;
            gemm_blocked_rows(ta, tb, 0, M, j1 - j0, K, alpha, A, lda, Bs, ldb, C + j0, ldc);
        }
    }
}

void pdl_cpu_gemm_strided_batched(const pdl_ctx *ctx, int ta, int tb, int M, int N, int K,
                                  float alpha, const float *A, int lda, long long sa,
                                  const float *B, int ldb, long long sb, float beta, float *C,
                                  int ldc, long long sc, int batch) {
    int nt = cpu_threads(ctx);
    if (nt > 1 && batch > 1 && sc != 0) {
        /* Images of the mini-batch in parallel (paper §2: embarrassingly parallel). */
        PDL_OMP(omp parallel for num_threads(nt) schedule(dynamic, 1))
        for (int b = 0; b < batch; ++b)
            gemm_serial(ctx, ta, tb, M, N, K, alpha, A + b * sa, lda, B + b * sb, ldb, beta,
                        C + b * sc, ldc);
    } else {
        for (int b = 0; b < batch; ++b)
            pdl_cpu_gemm(ctx, ta, tb, M, N, K, alpha, A + b * sa, lda, B + b * sb, ldb,
                         (sc == 0 && b > 0) ? 1.0f : beta, C + b * sc, ldc);
    }
}

/* ------------------------------------------------------------------------ */
/* BLAS-1                                                                    */
/* ------------------------------------------------------------------------ */

void pdl_cpu_axpby(const pdl_ctx *ctx, size_t n, float a, const float *x, float b, float *y) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    if (b == 0.0f) {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
        for (size_t i = 0; i < n; ++i) y[i] = a * x[i];
    } else if (b == 1.0f) {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
        for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
    } else {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
        for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + b * y[i];
    }
}

void pdl_cpu_scal(const pdl_ctx *ctx, size_t n, float a, float *x) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (size_t i = 0; i < n; ++i) x[i] *= a;
}

double pdl_cpu_sumsq(const pdl_ctx *ctx, size_t n, const float *x) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    double s = 0.0;
    PDL_OMP(omp parallel for num_threads(nt) reduction(+ : s) schedule(static) if (nt > 1))
    for (size_t i = 0; i < n; ++i) s += (double)x[i] * (double)x[i];
    return s;
}

size_t pdl_cpu_count_nonfinite(const pdl_ctx *ctx, size_t n, const float *x) {
    PDL_UNUSED(ctx);
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += !isfinite(x[i]);
    return c;
}

/* ------------------------------------------------------------------------ */
/* Convolution helpers                                                       */
/* ------------------------------------------------------------------------ */

void pdl_cpu_im2col(const pdl_ctx *ctx, const float *im, int N, int C, int H, int W, int kh,
                    int kw, int stride, int pad, int OH, int OW, float *col) {
    const int KK = C * kh * kw;
    const size_t ohw = (size_t)OH * OW;
    int nt = cpu_threads(ctx);
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1 && N > 1))
    for (int n = 0; n < N; ++n) {
        const float *img = im + (size_t)n * C * H * W;
        float *cn = col + (size_t)n * KK * ohw;
        for (int r = 0; r < KK; ++r) {
            const int c = r / (kh * kw);
            const int ki = (r / kw) % kh;
            const int kj = r % kw;
            float *dst = cn + (size_t)r * ohw;
            const float *plane = img + (size_t)c * H * W;
            for (int oh = 0; oh < OH; ++oh) {
                const int ih = oh * stride - pad + ki;
                float *d = dst + (size_t)oh * OW;
                if (ih < 0 || ih >= H) {
                    memset(d, 0, (size_t)OW * sizeof(float));
                    continue;
                }
                const float *srow = plane + (size_t)ih * W;
                for (int ow = 0; ow < OW; ++ow) {
                    const int iw = ow * stride - pad + kj;
                    d[ow] = (iw >= 0 && iw < W) ? srow[iw] : 0.0f;
                }
            }
        }
    }
}

void pdl_cpu_col2im(const pdl_ctx *ctx, const float *col, int N, int C, int H, int W, int kh,
                    int kw, int stride, int pad, int OH, int OW, float *im) {
    const int KK = C * kh * kw;
    const size_t ohw = (size_t)OH * OW;
    int nt = cpu_threads(ctx);
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1 && N > 1))
    for (int n = 0; n < N; ++n) {
        float *img = im + (size_t)n * C * H * W;
        const float *cn = col + (size_t)n * KK * ohw;
        for (int r = 0; r < KK; ++r) {
            const int c = r / (kh * kw);
            const int ki = (r / kw) % kh;
            const int kj = r % kw;
            const float *src = cn + (size_t)r * ohw;
            float *plane = img + (size_t)c * H * W;
            for (int oh = 0; oh < OH; ++oh) {
                const int ih = oh * stride - pad + ki;
                if (ih < 0 || ih >= H) continue;
                float *drow = plane + (size_t)ih * W;
                const float *s = src + (size_t)oh * OW;
                for (int ow = 0; ow < OW; ++ow) {
                    const int iw = ow * stride - pad + kj;
                    if (iw >= 0 && iw < W) drow[iw] += s[ow];
                }
            }
        }
    }
}

void pdl_cpu_conv2d_direct(const pdl_ctx *ctx, const float *in, const float *w, const float *b,
                           int N, int C, int H, int W, int OC, int kh, int kw, int stride, int pad,
                           int OH, int OW, float *out) {
    int nt = cpu_threads(ctx);
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for collapse(2) num_threads(nt) schedule(static) if (nt > 1))
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            const float *wk = w + (size_t)oc * C * kh * kw;
            float *o = out + ((size_t)n * OC + oc) * OH * OW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float acc = b ? b[oc] : 0.0f;
                    for (int c = 0; c < C; ++c) {
                        const float *plane = in + ((size_t)n * C + c) * H * W;
                        const float *wc = wk + (size_t)c * kh * kw;
                        for (int ki = 0; ki < kh; ++ki) {
                            int ih = oh * stride - pad + ki;
                            if (ih < 0 || ih >= H) continue;
                            for (int kj = 0; kj < kw; ++kj) {
                                int iw = ow * stride - pad + kj;
                                if (iw < 0 || iw >= W) continue;
                                acc += plane[(size_t)ih * W + iw] * wc[ki * kw + kj];
                            }
                        }
                    }
                    o[(size_t)oh * OW + ow] = acc;
                }
            }
        }
    }
}

void pdl_cpu_add_bias_nchw(const pdl_ctx *ctx, float *out, const float *bias, int N, int C, int HW) {
    int nt = (size_t)N * C * HW >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for collapse(2) num_threads(nt) schedule(static) if (nt > 1))
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            float *o = out + ((size_t)n * C + c) * HW;
            const float bv = bias[c];
            for (int i = 0; i < HW; ++i) o[i] += bv;
        }
}

void pdl_cpu_bias_grad_nchw(const pdl_ctx *ctx, const float *dout, float *dbias, int N, int C,
                            int HW) {
    int nt = (size_t)N * C * HW >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (int c = 0; c < C; ++c) {
        double acc = 0.0;
        for (int n = 0; n < N; ++n) {
            const float *d = dout + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; ++i) acc += d[i];
        }
        dbias[c] += (float)acc;
    }
}

void pdl_cpu_relu_forward(const pdl_ctx *ctx, size_t n, const float *in, float *out) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (size_t i = 0; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void pdl_cpu_relu_backward(const pdl_ctx *ctx, size_t n, const float *in, const float *dout,
                           float *din) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (size_t i = 0; i < n; ++i) din[i] = in[i] > 0.0f ? dout[i] : 0.0f;
}

void pdl_cpu_maxpool_forward(const pdl_ctx *ctx, const float *in, int N, int C, int H, int W,
                             int k, int s, int OH, int OW, float *out, int32_t *argmax) {
    int nt = (size_t)N * C * H * W >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (int p = 0; p < N * C; ++p) {
        const size_t base = (size_t)p * H * W;
        const float *plane = in + base;
        float *o = out + (size_t)p * OH * OW;
        int32_t *am = argmax + (size_t)p * OH * OW;
        for (int oh = 0; oh < OH; ++oh)
            for (int ow = 0; ow < OW; ++ow) {
                int h0 = oh * s, w0 = ow * s;
                int best = h0 * W + w0;
                float bv = plane[best];
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j) {
                        int h = h0 + i, w = w0 + j;
                        if (h >= H || w >= W) continue;
                        float v = plane[h * W + w];
                        if (v > bv) {
                            bv = v;
                            best = h * W + w;
                        }
                    }
                o[oh * OW + ow] = bv;
                am[oh * OW + ow] = (int32_t)(base + (size_t)best);
            }
    }
}

void pdl_cpu_maxpool_backward(const pdl_ctx *ctx, const float *dout, const int32_t *argmax, int N,
                              int C, int H, int W, int k, int s, int OH, int OW, float *din) {
    PDL_UNUSED(H);
    PDL_UNUSED(W);
    PDL_UNUSED(k);
    PDL_UNUSED(s);
    /* Pooling windows may overlap (k > s) but argmax never leaves its (n, c)
     * plane, so parallelizing over planes is race-free. */
    int nt = (size_t)N * C * OH * OW >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
    for (int p = 0; p < N * C; ++p) {
        const size_t o0 = (size_t)p * OH * OW;
        for (size_t o = o0; o < o0 + (size_t)OH * OW; ++o) din[argmax[o]] += dout[o];
    }
}

double pdl_cpu_softmax_xent(const pdl_ctx *ctx, const float *logits, const int32_t *labels, int N,
                            int K, float grad_scale, float *dlogits, int *correct) {
    PDL_UNUSED(ctx);
    double loss = 0.0;
    int hits = 0;
    for (int n = 0; n < N; ++n) {
        const float *f = logits + (size_t)n * K;
        float *d = dlogits ? dlogits + (size_t)n * K : NULL;
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
        /* L_i = -f_y + log sum_j exp(f_j)   (paper eq. 2) */
        loss += log(sum) - (double)(f[y] - mx);
        hits += (am == y);
        if (d) {
            for (int k = 0; k < K; ++k) {
                float p = (float)(exp((double)(f[k] - mx)) / sum);
                d[k] = (p - (k == y ? 1.0f : 0.0f)) * grad_scale;
            }
        }
    }
    if (correct) *correct = hits;
    return loss;
}

void pdl_cpu_sgd_update(const pdl_ctx *ctx, size_t n, float lr, float momentum, float wd, float *w,
                        const float *g, float *v) {
    int nt = n >= PDL_PAR_MIN ? cpu_threads(ctx) : 1;
    PDL_UNUSED(nt);
    if (momentum == 0.0f || !v) {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
        for (size_t i = 0; i < n; ++i) w[i] -= lr * (g[i] + wd * w[i]);
    } else {
        PDL_OMP(omp parallel for num_threads(nt) schedule(static) if (nt > 1))
        for (size_t i = 0; i < n; ++i) {
            v[i] = momentum * v[i] + g[i] + wd * w[i];
            w[i] -= lr * v[i];
        }
    }
}
