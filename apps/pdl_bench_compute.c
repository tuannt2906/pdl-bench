/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * pdl_bench_compute — computation micro-benchmarks (paper §3).
 *
 * Reproduces the paper's CPU-vs-GPU study and extends it to several kernel
 * implementations:
 *   gemm       square GEMM N×N×N                       (GFLOP/s vs size)
 *   fc         fully-connected forward, n×n weights     (paper Fig. 3–4, batch 1 = mat-vec)
 *   conv       convolution forward vs batch size        (paper Fig. 1–2, 96 filters 11×11 /4 on 224×224)
 *   conv-bwd   convolution backward (dW + dX) vs batch size
 *   train-step full forward + backward + SGD step of a model vs batch size
 * Every result is verified against the cpu-blocked reference.
 */
#include <math.h>
#include <string.h>

#include "pdl/backend.h"
#include "pdl/cli.h"
#include "pdl/comm.h"
#include "pdl/common.h"
#include "pdl/env.h"
#include "pdl/json.h"
#include "pdl/model.h"
#include "pdl/rng.h"
#include "pdl/timer.h"

static const pdl_cli_opt k_opts[] = {
    {"op", "gemm", "list of gemm, fc, conv, conv-bwd, train-step"},
    {"impls", "all", "list of cpu-naive, cpu-blocked, cpu-omp, cpu-direct, cuda-naive, cuda-tiled, cublas, cuda-direct"},
    {"sizes", "64,128,256,512,1024", "gemm sizes N / fc matrix sizes n"},
    {"batches", "1,8,16,32,64,128,256", "batch sizes for conv, conv-bwd, train-step"},
    {"fc-batch", "1", "batch size(s) for fc (1 = matrix-vector, as in the paper)"},
    {"conv", "1:224:224:96:11:4:0", "conv geometry C:H:W:OC:K:S:P (paper's first layer)"},
    {"model", "lenet", "model preset/spec for train-step"},
    {"width", "1", "width multiplier for train-step presets"},
    {"classes", "10", "classes for train-step"},
    {"input", "1:28:28", "input C:H:W for train-step"},
    {"reps", "10", "timed repetitions"},
    {"warmup", "2", "untimed warm-up runs"},
    {"threads", "0", "OpenMP threads (0 = default)"},
    {"gpu-id", "0", "CUDA device"},
    {"include-transfer", "0", "include host<->device copies in GPU timings"},
    {"verify", "1", "verify results against cpu-blocked"},
    {"max-naive", "5", "skip cpu-naive / cpu-direct when one repetition exceeds this many GFLOP"},
    {"seed", "42", "random seed"},
    {"output", "", "result JSON path"},
    {"run-id", "", "run identifier"},
    {"tag", "", "free-form tag"},
};

/* ------------------------------------------------------------------------ */

typedef struct {
    const char *name;
    pdl_device dev;
    pdl_impl impl;
    int direct; /* direct convolution instead of im2col + GEMM */
} impl_spec;

static int resolve_impl(const char *s, impl_spec *out) {
    static const impl_spec table[] = {
        {"cpu-naive", PDL_DEV_CPU, PDL_IMPL_CPU_NAIVE, 0},    {"cpu-blocked", PDL_DEV_CPU, PDL_IMPL_CPU_BLOCKED, 0},
        {"cpu-omp", PDL_DEV_CPU, PDL_IMPL_CPU_OMP, 0},        {"cpu-direct", PDL_DEV_CPU, PDL_IMPL_CPU_OMP, 1},
        {"cuda-naive", PDL_DEV_CUDA, PDL_IMPL_CUDA_NAIVE, 0}, {"cuda-tiled", PDL_DEV_CUDA, PDL_IMPL_CUDA_TILED, 0},
        {"cublas", PDL_DEV_CUDA, PDL_IMPL_CUDA_CUBLAS, 0},    {"cuda-direct", PDL_DEV_CUDA, PDL_IMPL_CUDA_TILED, 1},
    };
    for (size_t i = 0; i < PDL_ARRAY_LEN(table); ++i)
        if (strcmp(s, table[i].name) == 0) {
            *out = table[i];
            return 0;
        }
    return -1;
}

static int impl_available(const impl_spec *s) {
    if (s->dev == PDL_DEV_CUDA && !pdl_has_cuda()) return 0;
    if (s->impl == PDL_IMPL_CUDA_CUBLAS && !pdl_has_cublas()) return 0;
    return 1;
}

typedef struct {
    double mean, std, min, median;
    int reps;
} stats;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef void (*bench_fn)(void *arg);

/* GPUs idle at a low clock and need a few hundred ms of sustained load to boost; a fixed
 * number of warm-up repetitions of a sub-millisecond kernel is far too short, so GPU timings
 * were up to 10x too slow at small sizes. Warm up for a minimum wall time on CUDA. */
#define GPU_WARMUP_SECONDS 0.5

static stats time_it(const pdl_ctx *ctx, bench_fn fn, void *arg, int warmup, int reps) {
    const double w0 = pdl_now();
    for (int i = 0; i < warmup || (ctx->device == PDL_DEV_CUDA && pdl_now() - w0 < GPU_WARMUP_SECONDS); ++i) {
        fn(arg);
        if (ctx->device == PDL_DEV_CUDA) pdl_ctx_sync(ctx);
    }
    pdl_ctx_sync(ctx);
    double *t = (double *)pdl_xmalloc((size_t)reps * sizeof(double));
    for (int i = 0; i < reps; ++i) {
        double t0 = pdl_now();
        fn(arg);
        pdl_ctx_sync(ctx);
        t[i] = pdl_now() - t0;
    }
    stats s = {0, 0, 1e300, 0, reps};
    for (int i = 0; i < reps; ++i) {
        s.mean += t[i];
        s.min = PDL_MIN(s.min, t[i]);
    }
    s.mean /= reps;
    for (int i = 0; i < reps; ++i) s.std += (t[i] - s.mean) * (t[i] - s.mean);
    s.std = reps > 1 ? sqrt(s.std / (reps - 1)) : 0.0;
    qsort(t, (size_t)reps, sizeof(double), cmp_double);
    s.median = reps % 2 ? t[reps / 2] : 0.5 * (t[reps / 2 - 1] + t[reps / 2]);
    free(t);
    return s;
}

static void fill_random(float *x, size_t n, uint64_t seed, uint64_t stream) {
    pdl_rng r;
    pdl_rng_seed(&r, seed, 0xBE4C4ULL, stream, 0);
    for (size_t i = 0; i < n; ++i) x[i] = pdl_rng_uniformf(&r, -1.0f, 1.0f);
}

/* Compares two host arrays: max |a-b| and max |a-b| / max |ref|. */
static void compare(const float *a, const float *ref, size_t n, double *max_abs, double *rel) {
    double m = 0.0, r = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)ref[i]);
        if (d > m || isnan(d)) m = isnan(d) ? INFINITY : d;
        r = PDL_MAX(r, fabs((double)ref[i]));
    }
    *max_abs = m;
    *rel = r > 0 ? m / r : m;
}

/* Copies host data to a new device buffer of ctx. */
static float *to_dev(const pdl_ctx *ctx, const float *h, size_t n) {
    float *d = (float *)pdl_dev_malloc(ctx, n * sizeof(float));
    pdl_memcpy_h2d(ctx, d, h, n * sizeof(float));
    return d;
}

/* ------------------------------------------------------------------------ */
/* Record output                                                             */
/* ------------------------------------------------------------------------ */

typedef struct {
    pdl_json *j;
    int verbose;
} out_ctx;

static void write_record(out_ctx *o, const char *op, const impl_spec *is, const pdl_ctx *ctx, long long M, long long N,
                         long long K, long long n, long long batch, const char *conv, const char *model, double flops,
                         stats st, int include_transfer, int verified, double max_abs, double rel) {
    pdl_json *j = o->j;
    double gflops = st.mean > 0 ? flops / st.mean / 1e9 : 0.0;
    if (o->verbose)
        PDL_INFO("%-10s %-12s M=%-6lld N=%-6lld K=%-6lld batch=%-4lld  %10.3f ms  %9.2f GFLOP/s%s", op, is->name, M, N,
                 K, batch, st.mean * 1e3, gflops, verified ? (rel < 1e-3 ? "  ok" : "  MISMATCH") : "");
    if (!j) return;
    pdl_json_begin_object(j, NULL);
    pdl_json_kv_str(j, "op", op);
    pdl_json_kv_str(j, "impl", is->name);
    pdl_json_kv_str(j, "device", pdl_device_name(is->dev));
    pdl_json_kv_int(j, "threads", ctx->num_threads > 0 ? ctx->num_threads : pdl_openmp_max_threads());
    pdl_json_kv_int(j, "M", M);
    pdl_json_kv_int(j, "N", N);
    pdl_json_kv_int(j, "K", K);
    pdl_json_kv_int(j, "n", n);
    pdl_json_kv_int(j, "batch", batch);
    if (conv)
        pdl_json_kv_str(j, "conv", conv);
    else
        pdl_json_kv_null(j, "conv");
    if (model)
        pdl_json_kv_str(j, "model", model);
    else
        pdl_json_kv_null(j, "model");
    pdl_json_kv_num(j, "flops", flops);
    pdl_json_kv_num(j, "time_mean_s", st.mean);
    pdl_json_kv_num(j, "time_std_s", st.std);
    pdl_json_kv_num(j, "time_min_s", st.min);
    pdl_json_kv_num(j, "time_median_s", st.median);
    pdl_json_kv_int(j, "reps", st.reps);
    pdl_json_kv_num(j, "gflops", gflops);
    if (batch > 0)
        pdl_json_kv_num(j, "throughput_samples_per_s", st.mean > 0 ? batch / st.mean : 0.0);
    else
        pdl_json_kv_null(j, "throughput_samples_per_s");
    pdl_json_kv_bool(j, "include_transfer", include_transfer && is->dev == PDL_DEV_CUDA);
    if (verified) {
        pdl_json_kv_num(j, "max_abs_err", max_abs);
        pdl_json_kv_num(j, "rel_err", rel);
        pdl_json_kv_bool(j, "verified", rel < 1e-3);
    } else {
        pdl_json_kv_null(j, "max_abs_err");
        pdl_json_kv_null(j, "rel_err");
        pdl_json_kv_null(j, "verified");
    }
    pdl_json_end_object(j);
}

/* ------------------------------------------------------------------------ */
/* GEMM / FC                                                                 */
/* ------------------------------------------------------------------------ */

typedef struct {
    const pdl_ctx *ctx;
    int tb, M, N, K;
    const float *hA, *hB; /* host sources (transfer mode) */
    float *A, *B, *C, *hC;
    int transfer;
} gemm_arg;

static void gemm_run(void *p) {
    gemm_arg *g = (gemm_arg *)p;
    if (g->transfer) {
        pdl_memcpy_h2d(g->ctx, g->A, g->hA, (size_t)g->M * g->K * sizeof(float));
        pdl_memcpy_h2d(g->ctx, g->B, g->hB, (size_t)g->K * g->N * sizeof(float));
    }
    pdl_gemm(g->ctx, 0, g->tb, g->M, g->N, g->K, 1.0f, g->A, g->K, g->B, g->tb ? g->K : g->N, 0.0f, g->C, g->N);
    if (g->transfer) pdl_memcpy_d2h(g->ctx, g->hC, g->C, (size_t)g->M * g->N * sizeof(float));
}

/* op "gemm": C[n×n] = A[n×n] B[n×n]; op "fc": Y[b×n] = X[b×n] W[n×n]^T. */
static void bench_gemm_like(out_ctx *o, const char *op, const impl_spec *is, const pdl_ctx *ctx, int M, int N, int K,
                            int tb, long long n_label, long long batch, int warmup, int reps, int transfer, int verify,
                            uint64_t seed) {
    size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *hA = (float *)pdl_xmalloc(na * sizeof(float)), *hB = (float *)pdl_xmalloc(nb * sizeof(float));
    float *hC = (float *)pdl_xcalloc(nc, sizeof(float));
    fill_random(hA, na, seed, 1);
    fill_random(hB, nb, seed, 2);
    gemm_arg g = {ctx, tb, M, N, K, hA, hB, to_dev(ctx, hA, na), to_dev(ctx, hB, nb),
                  (float *)pdl_dev_malloc(ctx, nc * sizeof(float)), hC, transfer && ctx->device == PDL_DEV_CUDA};
    stats st = time_it(ctx, gemm_run, &g, warmup, reps);
    double max_abs = 0, rel = 0;
    if (verify) {
        pdl_memcpy_d2h(ctx, hC, g.C, nc * sizeof(float));
        pdl_ctx ref;
        pdl_ctx_init(&ref, PDL_DEV_CPU, PDL_IMPL_CPU_BLOCKED, 1, 0);
        float *R = (float *)pdl_xcalloc(nc, sizeof(float));
        pdl_gemm(&ref, 0, tb, M, N, K, 1.0f, hA, K, hB, tb ? K : N, 0.0f, R, N);
        compare(hC, R, nc, &max_abs, &rel);
        free(R);
        pdl_ctx_destroy(&ref);
    }
    write_record(o, op, is, ctx, M, N, K, n_label, batch, NULL, NULL, 2.0 * M * N * K, st, transfer, verify, max_abs,
                 rel);
    pdl_dev_free(ctx, g.A);
    pdl_dev_free(ctx, g.B);
    pdl_dev_free(ctx, g.C);
    free(hA);
    free(hB);
    free(hC);
}

/* ------------------------------------------------------------------------ */
/* Convolution                                                               */
/* ------------------------------------------------------------------------ */

typedef struct {
    int C, H, W, OC, K, S, P, OH, OW;
} conv_geom;

typedef struct {
    const pdl_ctx *ctx;
    conv_geom g;
    int B, direct, transfer, backward;
    const float *hX;
    float *X, *Wt, *bias, *col, *Y, *hY;
    float *dY, *dW, *db, *dcol, *dX; /* backward */
} conv_arg;

static void conv_forward(conv_arg *a) {
    const conv_geom *g = &a->g;
    const int Kc = g->C * g->K * g->K, P = g->OH * g->OW;
    if (a->direct) {
        pdl_conv2d_direct(a->ctx, a->X, a->Wt, a->bias, a->B, g->C, g->H, g->W, g->OC, g->K, g->K, g->S, g->P, g->OH,
                          g->OW, a->Y);
        return;
    }
    pdl_im2col(a->ctx, a->X, a->B, g->C, g->H, g->W, g->K, g->K, g->S, g->P, g->OH, g->OW, a->col);
    pdl_gemm_strided_batched(a->ctx, 0, 0, g->OC, P, Kc, 1.0f, a->Wt, Kc, 0, a->col, P, (long long)Kc * P, 0.0f, a->Y,
                             P, (long long)g->OC * P, a->B);
    pdl_add_bias_nchw(a->ctx, a->Y, a->bias, a->B, g->OC, P);
}

static void conv_backward(conv_arg *a) {
    const conv_geom *g = &a->g;
    const int Kc = g->C * g->K * g->K, P = g->OH * g->OW;
    pdl_dev_memset0(a->ctx, a->db, (size_t)g->OC * sizeof(float));
    pdl_gemm_strided_batched(a->ctx, 0, 1, g->OC, Kc, P, 1.0f, a->dY, P, (long long)g->OC * P, a->col, P,
                             (long long)Kc * P, 0.0f, a->dW, Kc, 0, a->B);
    pdl_bias_grad_nchw(a->ctx, a->dY, a->db, a->B, g->OC, P);
    pdl_gemm_strided_batched(a->ctx, 1, 0, Kc, P, g->OC, 1.0f, a->Wt, Kc, 0, a->dY, P, (long long)g->OC * P, 0.0f,
                             a->dcol, P, (long long)Kc * P, a->B);
    pdl_dev_memset0(a->ctx, a->dX, (size_t)a->B * g->C * g->H * g->W * sizeof(float));
    pdl_col2im(a->ctx, a->dcol, a->B, g->C, g->H, g->W, g->K, g->K, g->S, g->P, g->OH, g->OW, a->dX);
}

static void conv_run(void *p) {
    conv_arg *a = (conv_arg *)p;
    size_t nx = (size_t)a->B * a->g.C * a->g.H * a->g.W, ny = (size_t)a->B * a->g.OC * a->g.OH * a->g.OW;
    if (a->transfer) pdl_memcpy_h2d(a->ctx, a->X, a->hX, nx * sizeof(float));
    if (a->backward)
        conv_backward(a);
    else
        conv_forward(a);
    if (a->transfer && !a->backward) pdl_memcpy_d2h(a->ctx, a->hY, a->Y, ny * sizeof(float));
}

static void conv_alloc(conv_arg *a, const pdl_ctx *ctx, const conv_geom *g, int B, uint64_t seed, float **hbufs) {
    memset(a, 0, sizeof(*a));
    a->ctx = ctx;
    a->g = *g;
    a->B = B;
    size_t nx = (size_t)B * g->C * g->H * g->W, nw = (size_t)g->OC * g->C * g->K * g->K;
    size_t ny = (size_t)B * g->OC * g->OH * g->OW, ncol = (size_t)B * g->C * g->K * g->K * g->OH * g->OW;
    float *hX = (float *)pdl_xmalloc(nx * sizeof(float)), *hW = (float *)pdl_xmalloc(nw * sizeof(float));
    float *hb = (float *)pdl_xmalloc((size_t)g->OC * sizeof(float)), *hdY = (float *)pdl_xmalloc(ny * sizeof(float));
    fill_random(hX, nx, seed, 11);
    fill_random(hW, nw, seed, 12);
    fill_random(hb, (size_t)g->OC, seed, 13);
    fill_random(hdY, ny, seed, 14);
    a->hX = hX;
    a->X = to_dev(ctx, hX, nx);
    a->Wt = to_dev(ctx, hW, nw);
    a->bias = to_dev(ctx, hb, (size_t)g->OC);
    a->Y = (float *)pdl_dev_malloc(ctx, ny * sizeof(float));
    a->hY = (float *)pdl_xcalloc(ny, sizeof(float));
    a->col = (float *)pdl_dev_malloc(ctx, ncol * sizeof(float));
    a->dY = to_dev(ctx, hdY, ny);
    hbufs[0] = hX;
    hbufs[1] = hW;
    hbufs[2] = hb;
    hbufs[3] = hdY;
}

static void conv_alloc_backward(conv_arg *a) {
    const conv_geom *g = &a->g;
    size_t nw = (size_t)g->OC * g->C * g->K * g->K, nx = (size_t)a->B * g->C * g->H * g->W;
    size_t ncol = (size_t)a->B * g->C * g->K * g->K * g->OH * g->OW;
    a->dW = (float *)pdl_dev_malloc(a->ctx, nw * sizeof(float));
    a->db = (float *)pdl_dev_malloc(a->ctx, (size_t)g->OC * sizeof(float));
    a->dcol = (float *)pdl_dev_malloc(a->ctx, ncol * sizeof(float));
    a->dX = (float *)pdl_dev_malloc(a->ctx, nx * sizeof(float));
}

static void conv_free(conv_arg *a, float **hbufs) {
    const pdl_ctx *ctx = a->ctx;
    pdl_dev_free(ctx, a->X);
    pdl_dev_free(ctx, a->Wt);
    pdl_dev_free(ctx, a->bias);
    pdl_dev_free(ctx, a->Y);
    pdl_dev_free(ctx, a->col);
    pdl_dev_free(ctx, a->dY);
    pdl_dev_free(ctx, a->dW);
    pdl_dev_free(ctx, a->db);
    pdl_dev_free(ctx, a->dcol);
    pdl_dev_free(ctx, a->dX);
    free(a->hY);
    for (int i = 0; i < 4; ++i) free(hbufs[i]);
}

/* Result of a conv op copied to host: forward → Y, backward → [dW, dX]. */
static float *conv_result(conv_arg *a, size_t *n) {
    const conv_geom *g = &a->g;
    if (!a->backward) {
        *n = (size_t)a->B * g->OC * g->OH * g->OW;
        float *h = (float *)pdl_xmalloc(*n * sizeof(float));
        pdl_memcpy_d2h(a->ctx, h, a->Y, *n * sizeof(float));
        return h;
    }
    size_t nw = (size_t)g->OC * g->C * g->K * g->K, nx = (size_t)a->B * g->C * g->H * g->W;
    *n = nw + nx;
    float *h = (float *)pdl_xmalloc(*n * sizeof(float));
    pdl_memcpy_d2h(a->ctx, h, a->dW, nw * sizeof(float));
    pdl_memcpy_d2h(a->ctx, h + nw, a->dX, nx * sizeof(float));
    return h;
}

static void bench_conv(out_ctx *o, const char *op, const char *conv_desc, const impl_spec *is, const pdl_ctx *ctx,
                       const conv_geom *g, int B, int backward, int warmup, int reps, int transfer, int verify,
                       uint64_t seed) {
    conv_arg a;
    float *hb[4];
    conv_alloc(&a, ctx, g, B, seed, hb);
    a.direct = is->direct;
    a.backward = backward;
    a.transfer = transfer && ctx->device == PDL_DEV_CUDA;
    if (backward) {
        conv_alloc_backward(&a);
        conv_forward(&a); /* fills the im2col buffer used by dW */
    }
    stats st = time_it(ctx, conv_run, &a, warmup, reps);
    double max_abs = 0, rel = 0;
    if (verify) {
        size_t n = 0;
        float *res = conv_result(&a, &n);
        pdl_ctx ref;
        pdl_ctx_init(&ref, PDL_DEV_CPU, PDL_IMPL_CPU_OMP, 0, 0);
        conv_arg r;
        float *rhb[4];
        conv_alloc(&r, &ref, g, B, seed, rhb);
        r.backward = backward;
        if (backward) {
            conv_alloc_backward(&r);
            conv_forward(&r);
        }
        conv_run(&r);
        size_t rn = 0;
        float *rres = conv_result(&r, &rn);
        compare(res, rres, n, &max_abs, &rel);
        free(res);
        free(rres);
        conv_free(&r, rhb);
        pdl_ctx_destroy(&ref);
    }
    const double fwd = 2.0 * B * g->OC * g->OH * g->OW * (double)g->C * g->K * g->K;
    const int Kc = g->C * g->K * g->K, P = g->OH * g->OW;
    write_record(o, op, is, ctx, g->OC, P, Kc, 0, B, conv_desc, NULL, backward ? 2.0 * fwd : fwd, st, transfer, verify,
                 max_abs, rel);
    conv_free(&a, hb);
}

/* ------------------------------------------------------------------------ */
/* Full training step                                                        */
/* ------------------------------------------------------------------------ */

typedef struct {
    pdl_model *m;
    int B;
    float *v;
    double loss;
} step_arg;

static void step_run(void *p) {
    step_arg *s = (step_arg *)p;
    int hits;
    s->loss = pdl_model_forward(s->m, s->B, 1.0f / s->B, &hits);
    pdl_model_backward(s->m, s->B, NULL, NULL);
    pdl_sgd_update(s->m->ctx, s->m->num_params, 1e-3f, 0.9f, 0.0f, s->m->params, s->m->grads, s->v);
}

static double first_loss(pdl_ctx *ctx, const char *spec, int C, int H, int W, int K, int B, const float *x,
                         const int32_t *y, uint64_t seed) {
    pdl_model *m = pdl_model_create(ctx, spec, C, H, W, K, B);
    pdl_model_init_params(m, seed);
    pdl_model_set_batch(m, x, y, B);
    int hits;
    double l = pdl_model_forward(m, B, 1.0f / B, &hits);
    pdl_model_free(m);
    return l;
}

static void bench_train_step(out_ctx *o, const impl_spec *is, pdl_ctx *ctx, const char *model_name, const char *spec,
                             int C, int H, int W, int K, int B, int warmup, int reps, int verify, uint64_t seed,
                             double max_flops) {
    {
        pdl_ctx probe;
        pdl_ctx_init(&probe, PDL_DEV_CPU, PDL_IMPL_CPU_BLOCKED, 1, 0);
        pdl_model *pm = pdl_model_create(&probe, spec, C, H, W, K, 1);
        double fl = 3.0 * pdl_model_flops_per_sample(pm) * B;
        pdl_model_free(pm);
        pdl_ctx_destroy(&probe);
        if (fl > max_flops) return; /* naive variant too slow for this size */
    }
    size_t nx = (size_t)B * C * H * W;
    float *x = (float *)pdl_xmalloc(nx * sizeof(float));
    int32_t *y = (int32_t *)pdl_xmalloc((size_t)B * sizeof(int32_t));
    fill_random(x, nx, seed, 21);
    for (int i = 0; i < B; ++i) y[i] = i % K;
    double max_abs = 0, rel = 0;
    if (verify) {
        pdl_ctx ref;
        pdl_ctx_init(&ref, PDL_DEV_CPU, PDL_IMPL_CPU_OMP, 0, 0);
        double lr = first_loss(&ref, spec, C, H, W, K, B, x, y, seed);
        double lt = first_loss(ctx, spec, C, H, W, K, B, x, y, seed);
        max_abs = fabs(lt - lr);
        rel = max_abs / PDL_MAX(fabs(lr), 1e-12);
        pdl_ctx_destroy(&ref);
    }
    pdl_model *m = pdl_model_create(ctx, spec, C, H, W, K, B);
    pdl_model_init_params(m, seed);
    pdl_model_set_batch(m, x, y, B);
    step_arg s = {m, B, (float *)pdl_dev_malloc(ctx, m->num_params * sizeof(float)), 0.0};
    stats st = time_it(ctx, step_run, &s, warmup, reps);
    /* forward + backward ≈ 3× forward FLOPs */
    write_record(o, "train-step", is, ctx, 0, 0, 0, (long long)m->num_params, B, NULL, model_name,
                 3.0 * pdl_model_flops_per_sample(m) * B, st, 0, verify, max_abs, rel);
    pdl_dev_free(ctx, s.v);
    pdl_model_free(m);
    free(x);
    free(y);
}

/* ------------------------------------------------------------------------ */

int main(int argc, char **argv) {
    int rank = 0, world = 1;
    pdl_mpi_init(&argc, &argv, &rank, &world);
    pdl_cli cli;
    if (pdl_cli_parse(&cli, k_opts, (int)PDL_ARRAY_LEN(k_opts), argc, argv, rank == 0) != 0) {
        pdl_mpi_finalize();
        return 2;
    }
    if (cli.help) {
        if (rank == 0) pdl_cli_print_help(&cli, argv[0], "Computation micro-benchmarks (CPU vs GPU kernels, paper §3).");
        pdl_mpi_finalize();
        return 0;
    }
    int hosts = pdl_count_hosts(MPI_COMM_WORLD);
    if (rank != 0) { /* single-process benchmark: extra ranks idle */
        pdl_mpi_finalize();
        return 0;
    }
    if (world > 1) PDL_WARN("pdl_bench_compute is single-process; ranks 1..%d are idle", world - 1);

    const int reps = (int)pdl_cli_int(&cli, "reps"), warmup = (int)pdl_cli_int(&cli, "warmup");
    const int threads = (int)pdl_cli_int(&cli, "threads"), gpu = (int)pdl_cli_int(&cli, "gpu-id");
    const int transfer = (int)pdl_cli_int(&cli, "include-transfer"), verify = (int)pdl_cli_int(&cli, "verify");
    const double max_naive = pdl_cli_dbl(&cli, "max-naive") * 1e9;
    const uint64_t seed = (uint64_t)pdl_cli_int(&cli, "seed");
    const char *output = pdl_cli_str(&cli, "output");

    /* implementations */
    char **impl_names;
    int nimpl = pdl_cli_list(&cli, "impls", &impl_names);
    impl_spec impls[16];
    int ni = 0;
    if (nimpl == 1 && strcmp(impl_names[0], "all") == 0) {
        const char *all[] = {"cpu-naive", "cpu-blocked", "cpu-omp", "cpu-direct", "cuda-naive", "cuda-tiled", "cublas", "cuda-direct"};
        for (size_t i = 0; i < PDL_ARRAY_LEN(all); ++i) {
            impl_spec s;
            resolve_impl(all[i], &s);
            if (impl_available(&s) && !(s.impl == PDL_IMPL_CPU_OMP && !pdl_has_openmp() && !s.direct)) impls[ni++] = s;
        }
    } else {
        for (int i = 0; i < nimpl && ni < 16; ++i) {
            impl_spec s;
            PDL_CHECK(resolve_impl(impl_names[i], &s) == 0, "unknown implementation '%s'", impl_names[i]);
            if (!impl_available(&s)) {
                PDL_WARN("skipping %s: not available in this build / machine", impl_names[i]);
                continue;
            }
            impls[ni++] = s;
        }
    }
    pdl_str_free_parts(impl_names, nimpl);
    free(impl_names);
    PDL_CHECK(ni > 0, "no implementation to run");

    long long sizes[64], batches[64], fc_batches[16];
    int nsizes = pdl_cli_int_list(&cli, "sizes", sizes, 64);
    int nbatches = pdl_cli_int_list(&cli, "batches", batches, 64);
    int nfcb = pdl_cli_int_list(&cli, "fc-batch", fc_batches, 16);

    conv_geom g;
    const char *conv_desc = pdl_cli_str(&cli, "conv");
    PDL_CHECK(sscanf(conv_desc, "%d:%d:%d:%d:%d:%d:%d", &g.C, &g.H, &g.W, &g.OC, &g.K, &g.S, &g.P) == 7,
              "--conv expects C:H:W:OC:K:S:P");
    g.OH = (g.H + 2 * g.P - g.K) / g.S + 1;
    g.OW = (g.W + 2 * g.P - g.K) / g.S + 1;
    PDL_CHECK(g.OH > 0 && g.OW > 0, "invalid conv geometry");

    int in_c, in_h, in_w;
    PDL_CHECK(sscanf(pdl_cli_str(&cli, "input"), "%d:%d:%d", &in_c, &in_h, &in_w) == 3, "--input expects C:H:W");
    const int classes = (int)pdl_cli_int(&cli, "classes");
    char *spec = pdl_model_preset_spec(pdl_cli_str(&cli, "model"), (int)pdl_cli_int(&cli, "width"), classes);

    out_ctx o = {NULL, 1};
    if (output[0]) {
        o.j = pdl_json_open(output);
        PDL_CHECK(o.j, "cannot write %s", output);
        pdl_json_begin_object(o.j, NULL);
        pdl_json_kv_str(o.j, "schema", "pdl-bench/compute/v1");
        pdl_json_kv_str(o.j, "version", PDL_VERSION_STRING);
        pdl_json_kv_str(o.j, "run_id", pdl_cli_str(&cli, "run-id"));
        pdl_json_kv_str(o.j, "tag", pdl_cli_str(&cli, "tag"));
        pdl_cli_write_json(&cli, o.j);
        pdl_ctx envctx;
        pdl_ctx_init(&envctx, PDL_DEV_CPU, pdl_has_openmp() ? PDL_IMPL_CPU_OMP : PDL_IMPL_CPU_BLOCKED, threads, 0);
        pdl_env_write_json(o.j, &envctx, argc, argv, world, hosts);
        pdl_ctx_destroy(&envctx);
        pdl_json_kv_str(o.j, "model_spec", spec);
        pdl_json_begin_array(o.j, "records");
    }

    char **ops;
    int nops = pdl_cli_list(&cli, "op", &ops);
    pdl_env_print_banner(NULL, world);
    for (int i = 0; i < ni; ++i) {
        const impl_spec *is = &impls[i];
        const int is_naive = (is->impl == PDL_IMPL_CPU_NAIVE || (is->direct && is->dev == PDL_DEV_CPU));
        pdl_ctx ctx;
        pdl_ctx_init(&ctx, is->dev, is->impl, threads, gpu);
        for (int q = 0; q < nops; ++q) {
            const char *op = ops[q];
            if (strcmp(op, "gemm") == 0 || strcmp(op, "fc") == 0) {
                if (is->direct) continue;
                int fc = strcmp(op, "fc") == 0;
                for (int si = 0; si < nsizes; ++si) {
                    int nb = fc ? nfcb : 1;
                    for (int bi = 0; bi < nb; ++bi) {
                        int n = (int)sizes[si], b = fc ? (int)fc_batches[bi] : n;
                        if (is_naive && 2.0 * b * n * n > max_naive) continue;
                        bench_gemm_like(&o, op, is, &ctx, b, n, n, fc, n, fc ? b : 0, warmup, reps, transfer, verify,
                                        seed);
                    }
                }
            } else if (strcmp(op, "conv") == 0 || strcmp(op, "conv-bwd") == 0) {
                int bwd = strcmp(op, "conv-bwd") == 0;
                if (bwd && is->direct) continue;
                for (int bi = 0; bi < nbatches; ++bi) {
                    int B = (int)batches[bi];
                    double fl = 2.0 * B * g.OC * g.OH * g.OW * (double)g.C * g.K * g.K;
                    if (is_naive && fl > max_naive) continue;
                    bench_conv(&o, op, conv_desc, is, &ctx, &g, B, bwd, warmup, reps, transfer, verify, seed);
                }
            } else if (strcmp(op, "train-step") == 0) {
                if (is->direct) continue;
                for (int bi = 0; bi < nbatches; ++bi) {
                    int B = (int)batches[bi];
                    bench_train_step(&o, is, &ctx, pdl_cli_str(&cli, "model"), spec, in_c, in_h, in_w, classes, B,
                                     warmup, reps, verify, seed, is_naive ? max_naive : 1e300);
                }
            } else {
                PDL_FATAL("unknown --op '%s'", op);
            }
        }
        pdl_ctx_destroy(&ctx);
    }
    pdl_str_free_parts(ops, nops);
    free(ops);
    free(spec);
    if (o.j) {
        pdl_json_end_array(o.j);
        pdl_json_end_object(o.j);
        pdl_json_close(o.j);
        PDL_INFO("results written to %s", output);
    }
    pdl_cli_free(&cli);
    pdl_mpi_finalize();
    return 0;
}
