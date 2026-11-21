/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * pdl_check — correctness suite ("benchmarks must not trade correctness for speed").
 *
 *   gradcheck    analytic vs central finite-difference gradients for every layer type
 *   kernels      GEMM variants (all transpose combinations) agree; CUDA kernels match CPU
 *   determinism  identical seeds → bitwise identical parameters
 *   finite       NaN / Inf detection works on every device
 *   allreduce    every algorithm returns the exact sum on every rank       (MPI, >= 2 ranks)
 *   equivalence  sync SGD on k ranks == sequential SGD with the same global batch (MPI)
 *   sync         replicas stay bitwise identical for every allreduce algorithm     (MPI)
 */
#include <math.h>
#include <string.h>

#include "pdl/backend.h"
#include "pdl/cli.h"
#include "pdl/comm.h"
#include "pdl/common.h"
#include "pdl/dataset.h"
#include "pdl/env.h"
#include "pdl/json.h"
#include "pdl/model.h"
#include "pdl/rng.h"

static const pdl_cli_opt k_opts[] = {
    {"tests", "all", "list of gradcheck, kernels, determinism, finite, allreduce, equivalence, sync"},
    {"device", "cpu", "cpu | cuda | both"},
    {"tol", "1e-3", "relative tolerance for kernel comparisons"},
    {"steps", "5", "training steps for determinism / equivalence / sync"},
    {"seed", "42", "random seed"},
    {"output", "", "result JSON path"},
    {"verbose", "1", "print every test"},
};

/* ------------------------------------------------------------------------ */
/* Result collection                                                         */
/* ------------------------------------------------------------------------ */

typedef struct {
    char name[96];
    char device[8];
    int passed;
    double max_err, tol;
    char detail[160];
} result;

static result g_results[512];
static int g_nres = 0, g_rank = 0, g_verbose = 1;

static void report(const char *name, const char *dev, int passed, double err, double tol, const char *detail) {
    if (g_nres >= (int)PDL_ARRAY_LEN(g_results)) return;
    result *r = &g_results[g_nres++];
    snprintf(r->name, sizeof(r->name), "%s", name);
    snprintf(r->device, sizeof(r->device), "%s", dev);
    r->passed = passed;
    r->max_err = err;
    r->tol = tol;
    snprintf(r->detail, sizeof(r->detail), "%s", detail ? detail : "");
    if (g_rank == 0 && (g_verbose || !passed))
        printf("[%s] %-36s %-5s err=%.3e tol=%.1e %s\n", passed ? " PASS " : "*FAIL*", name, dev, err, tol,
               detail ? detail : "");
}

static void fill(float *x, size_t n, uint64_t seed, uint64_t s) {
    pdl_rng r;
    pdl_rng_seed(&r, seed, 0xC4EC4ULL, s, 0);
    for (size_t i = 0; i < n; ++i) x[i] = pdl_rng_uniformf(&r, -1.0f, 1.0f);
}

static double rel_diff(const float *a, const float *b, size_t n) {
    double m = 0, s = 1e-30;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (isnan(d)) return INFINITY;
        m = PDL_MAX(m, d);
        s = PDL_MAX(s, fabs((double)b[i]));
    }
    return m / s;
}

/* ------------------------------------------------------------------------ */
/* gradcheck                                                                 */
/* ------------------------------------------------------------------------ */

typedef struct {
    const char *name, *spec;
    int c, h, w, classes, batch;
} gc_case;

/* Mean loss plus a signature of every non-differentiable decision (ReLU
 * masks, max-pool argmax). Finite differences are only meaningful when the
 * +eps and -eps evaluations take the same decisions (no kink crossed). */
static double batch_loss(pdl_model *m, int B, uint64_t *signature) {
    int hits;
    double loss = pdl_model_eval_batch(m, B, &hits) / B;
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < m->num_layers; ++i) {
        const pdl_layer *L = &m->layers[i];
        size_t n = (size_t)B * L->out_c * L->out_h * L->out_w;
        if (L->kind == PDL_LAYER_RELU) {
            float *o = (float *)pdl_xmalloc(n * sizeof(float));
            pdl_memcpy_d2h(m->ctx, o, L->out, n * sizeof(float));
            for (size_t k = 0; k < n; ++k) h = (h ^ (uint64_t)(o[k] > 0.0f)) * 1099511628211ULL;
            free(o);
        } else if (L->kind == PDL_LAYER_MAXPOOL) {
            int32_t *a = (int32_t *)pdl_xmalloc(n * sizeof(int32_t));
            pdl_memcpy_d2h(m->ctx, a, L->argmax, n * sizeof(int32_t));
            for (size_t k = 0; k < n; ++k) h = (h ^ (uint64_t)(uint32_t)a[k]) * 1099511628211ULL;
            free(a);
        }
    }
    *signature = h;
    return loss;
}

static void test_gradcheck(pdl_ctx *ctx, uint64_t seed) {
    const gc_case cases[] = {
        {"gradcheck/conv+relu+pool+fc", "conv:3:3:1:1,relu,pool:2:2,fc:4", 2, 6, 6, 4, 3},
        {"gradcheck/strided-conv+overlap-pool", "conv:4:3:2:1,relu,pool:3:2,fc:3", 1, 9, 9, 3, 2},
        {"gradcheck/mlp", "fc:6,relu,fc:5,relu,fc:3", 1, 4, 4, 3, 4},
        {"gradcheck/two-conv", "conv:2:3:1:0,relu,conv:3:2:1:0,relu,fc:4", 1, 7, 7, 4, 2},
    };
    const double eps = 1e-2, tol = 2e-2;
    for (size_t ci = 0; ci < PDL_ARRAY_LEN(cases); ++ci) {
        const gc_case *c = &cases[ci];
        pdl_model *m = pdl_model_create(ctx, c->spec, c->c, c->h, c->w, c->classes, c->batch);
        pdl_model_init_params(m, seed + ci);
        size_t nx = (size_t)c->batch * c->c * c->h * c->w, P = m->num_params;
        float *x = (float *)pdl_xmalloc(nx * sizeof(float));
        int32_t y[16];
        fill(x, nx, seed, 100 + ci);
        for (int i = 0; i < c->batch; ++i) y[i] = (i * 7 + 1) % c->classes;
        pdl_model_set_batch(m, x, y, c->batch);
        int hits;
        pdl_model_forward(m, c->batch, 1.0f / c->batch, &hits);
        pdl_model_backward(m, c->batch, NULL, NULL);
        float *g = (float *)pdl_xmalloc(P * sizeof(float)), *p = (float *)pdl_xmalloc(P * sizeof(float));
        pdl_memcpy_d2h(ctx, g, m->grads, P * sizeof(float));
        pdl_memcpy_d2h(ctx, p, m->params, P * sizeof(float));
        /* check up to 40 parameters per layer (all of them for small layers) */
        double worst = 0.0;
        int checked = 0, kinks = 0;
        pdl_rng r;
        pdl_rng_seed(&r, seed, 0x6C6CULL, ci, 0);
        for (int li = 0; li < m->num_layers; ++li) {
            const pdl_layer *L = &m->layers[li];
            size_t cnt = L->w_count + L->b_count;
            if (!cnt) continue;
            int probes = (int)PDL_MIN(cnt, (size_t)40);
            for (int q = 0; q < probes; ++q) {
                size_t i = L->w_off + (cnt <= 40 ? (size_t)q : (size_t)pdl_rng_below(&r, cnt));
                float orig = p[i];
                uint64_t sp, sm;
                p[i] = orig + (float)eps;
                pdl_memcpy_h2d(ctx, m->params, p, P * sizeof(float));
                double lp = batch_loss(m, c->batch, &sp);
                p[i] = orig - (float)eps;
                pdl_memcpy_h2d(ctx, m->params, p, P * sizeof(float));
                double lm = batch_loss(m, c->batch, &sm);
                p[i] = orig;
                if (sp != sm) { /* ReLU / max-pool kink inside [-eps, +eps]: skip */
                    kinks++;
                    continue;
                }
                double num = (lp - lm) / (2 * eps), ana = g[i];
                double err = fabs(num - ana) / PDL_MAX(1e-2, fabs(num) + fabs(ana));
                worst = PDL_MAX(worst, err);
                checked++;
            }
        }
        pdl_memcpy_h2d(ctx, m->params, p, P * sizeof(float));
        char detail[96];
        snprintf(detail, sizeof(detail), "%d params probed, %d skipped at kinks", checked, kinks);
        report(c->name, pdl_device_name(ctx->device), worst < tol && checked >= kinks, worst, tol, detail);
        free(x);
        free(g);
        free(p);
        pdl_model_free(m);
    }
}

/* ------------------------------------------------------------------------ */
/* kernels                                                                   */
/* ------------------------------------------------------------------------ */

static void gemm_case(pdl_ctx *ctx, pdl_ctx *ref, int ta, int tb, double tol, uint64_t seed, const char *label) {
    const int M = 37, N = 53, K = 29, batch = 3;
    const int lda = ta ? M : K, ldb = tb ? K : N;
    size_t na = (size_t)M * K * batch, nb = (size_t)K * N * batch, nc = (size_t)M * N * batch;
    float *A = (float *)pdl_xmalloc(na * 4), *B = (float *)pdl_xmalloc(nb * 4), *C0 = (float *)pdl_xmalloc(nc * 4);
    float *R = (float *)pdl_xmalloc(nc * 4), *H = (float *)pdl_xmalloc(nc * 4);
    fill(A, na, seed, 1);
    fill(B, nb, seed, 2);
    fill(C0, nc, seed, 3);
    memcpy(R, C0, nc * 4);
    /* reference: per-matrix gemm with beta = 0.5 */
    for (int b = 0; b < batch; ++b)
        pdl_gemm(ref, ta, tb, M, N, K, 1.5f, A + (size_t)b * M * K, lda, B + (size_t)b * K * N, ldb, 0.5f,
                 R + (size_t)b * M * N, N);
    float *dA = (float *)pdl_dev_malloc(ctx, na * 4), *dB = (float *)pdl_dev_malloc(ctx, nb * 4);
    float *dC = (float *)pdl_dev_malloc(ctx, nc * 4);
    pdl_memcpy_h2d(ctx, dA, A, na * 4);
    pdl_memcpy_h2d(ctx, dB, B, nb * 4);
    pdl_memcpy_h2d(ctx, dC, C0, nc * 4);
    pdl_gemm_strided_batched(ctx, ta, tb, M, N, K, 1.5f, dA, lda, (long long)M * K, dB, ldb, (long long)K * N, 0.5f,
                             dC, N, (long long)M * N, batch);
    pdl_memcpy_d2h(ctx, H, dC, nc * 4);
    double e = rel_diff(H, R, nc);
    char name[96];
    snprintf(name, sizeof(name), "kernels/gemm-%s-%c%c", label, ta ? 'T' : 'N', tb ? 'T' : 'N');
    report(name, pdl_device_name(ctx->device), e < tol, e, tol, "37x53x29 batch 3, alpha 1.5 beta 0.5");
    pdl_dev_free(ctx, dA);
    pdl_dev_free(ctx, dB);
    pdl_dev_free(ctx, dC);
    free(A), free(B), free(C0), free(R), free(H);
}

/* Runs one conv/pool/activation pipeline on ctx and returns concatenated host outputs. */
static float *layer_pipeline(pdl_ctx *ctx, uint64_t seed, size_t *n_out, float *loss_out) {
    const int N = 2, C = 3, H = 11, W = 9, OC = 4, K = 3, S = 2, Pd = 1;
    const int OH = (H + 2 * Pd - K) / S + 1, OW = (W + 2 * Pd - K) / S + 1;
    const int PK = 3, PS = 2, POH = (OH - PK) / PS + 1, POW = (OW - PK) / PS + 1;
    size_t nx = (size_t)N * C * H * W, nw = (size_t)OC * C * K * K, ncol = (size_t)N * C * K * K * OH * OW;
    size_t ny = (size_t)N * OC * OH * OW, np = (size_t)N * OC * POH * POW;
    float *hx = (float *)pdl_xmalloc(nx * 4), *hw = (float *)pdl_xmalloc(nw * 4), *hb = (float *)pdl_xmalloc(OC * 4);
    fill(hx, nx, seed, 31);
    fill(hw, nw, seed, 32);
    fill(hb, OC, seed, 33);
    float *x = (float *)pdl_dev_malloc(ctx, nx * 4), *w = (float *)pdl_dev_malloc(ctx, nw * 4);
    float *b = (float *)pdl_dev_malloc(ctx, OC * 4), *col = (float *)pdl_dev_malloc(ctx, ncol * 4);
    float *y = (float *)pdl_dev_malloc(ctx, ny * 4), *yd = (float *)pdl_dev_malloc(ctx, ny * 4);
    float *pr = (float *)pdl_dev_malloc(ctx, ny * 4), *po = (float *)pdl_dev_malloc(ctx, np * 4);
    float *dpool = (float *)pdl_dev_malloc(ctx, ny * 4), *dx = (float *)pdl_dev_malloc(ctx, nx * 4);
    int32_t *am = (int32_t *)pdl_dev_malloc(ctx, np * 4);
    pdl_memcpy_h2d(ctx, x, hx, nx * 4);
    pdl_memcpy_h2d(ctx, w, hw, nw * 4);
    pdl_memcpy_h2d(ctx, b, hb, OC * 4);
    /* im2col+gemm conv, direct conv, relu, overlapping max-pool fwd/bwd, col2im */
    pdl_im2col(ctx, x, N, C, H, W, K, K, S, Pd, OH, OW, col);
    pdl_gemm_strided_batched(ctx, 0, 0, OC, OH * OW, C * K * K, 1.0f, w, C * K * K, 0, col, OH * OW,
                             (long long)C * K * K * OH * OW, 0.0f, y, OH * OW, (long long)OC * OH * OW, N);
    pdl_add_bias_nchw(ctx, y, b, N, OC, OH * OW);
    pdl_conv2d_direct(ctx, x, w, b, N, C, H, W, OC, K, K, S, Pd, OH, OW, yd);
    pdl_relu_forward(ctx, ny, y, pr);
    pdl_maxpool_forward(ctx, pr, N, OC, OH, OW, PK, PS, POH, POW, po, am);
    pdl_dev_memset0(ctx, dpool, ny * 4);
    pdl_maxpool_backward(ctx, po, am, N, OC, OH, OW, PK, PS, POH, POW, dpool);
    pdl_dev_memset0(ctx, dx, nx * 4);
    pdl_col2im(ctx, col, N, C, H, W, K, K, S, Pd, OH, OW, dx);
    /* softmax over the pooled map viewed as [N][OC*POH*POW] with labels */
    int32_t lab[2] = {1, 5};
    int32_t *dlab = (int32_t *)pdl_dev_malloc(ctx, sizeof(lab));
    pdl_memcpy_h2d(ctx, dlab, lab, sizeof(lab));
    float *dl = (float *)pdl_dev_malloc(ctx, np * 4);
    int hits;
    *loss_out = (float)pdl_softmax_xent(ctx, po, dlab, N, (int)(np / N), 0.5f, dl, &hits);
    *n_out = ny * 2 + np + ny + nx + np;
    float *out = (float *)pdl_xmalloc(*n_out * 4);
    size_t o = 0;
    pdl_memcpy_d2h(ctx, out + o, y, ny * 4), o += ny;
    pdl_memcpy_d2h(ctx, out + o, yd, ny * 4), o += ny;
    pdl_memcpy_d2h(ctx, out + o, po, np * 4), o += np;
    pdl_memcpy_d2h(ctx, out + o, dpool, ny * 4), o += ny;
    pdl_memcpy_d2h(ctx, out + o, dx, nx * 4), o += nx;
    pdl_memcpy_d2h(ctx, out + o, dl, np * 4), o += np;
    void *bufs[] = {x, w, b, col, y, yd, pr, po, dpool, dx, am, dlab, dl};
    for (size_t i = 0; i < PDL_ARRAY_LEN(bufs); ++i) pdl_dev_free(ctx, bufs[i]);
    free(hx), free(hw), free(hb);
    return out;
}

static void test_kernels(int with_cuda, double tol, uint64_t seed) {
    pdl_ctx ref;
    pdl_ctx_init(&ref, PDL_DEV_CPU, PDL_IMPL_CPU_BLOCKED, 1, 0);
    const pdl_impl cpu_impls[] = {PDL_IMPL_CPU_NAIVE, PDL_IMPL_CPU_BLOCKED, PDL_IMPL_CPU_OMP};
    for (size_t i = 0; i < PDL_ARRAY_LEN(cpu_impls); ++i) {
        pdl_ctx c;
        pdl_ctx_init(&c, PDL_DEV_CPU, cpu_impls[i], 0, 0);
        for (int ta = 0; ta < 2; ++ta)
            for (int tb = 0; tb < 2; ++tb) gemm_case(&c, &ref, ta, tb, tol, seed, pdl_impl_name(cpu_impls[i]));
        pdl_ctx_destroy(&c);
    }
    /* direct conv vs im2col conv on CPU */
    size_t n = 0;
    float loss_ref = 0;
    float *cpu_out = layer_pipeline(&ref, seed, &n, &loss_ref);
    {
        const size_t ny = 2 * 4 * 6 * 5; /* N*OC*OH*OW for the pipeline geometry */
        double e = rel_diff(cpu_out + ny, cpu_out, ny);
        report("kernels/conv-direct-vs-im2col", "cpu", e < tol, e, tol, NULL);
    }
    if (with_cuda) {
        const pdl_impl cuda_impls[] = {PDL_IMPL_CUDA_NAIVE, PDL_IMPL_CUDA_TILED, PDL_IMPL_CUDA_CUBLAS};
        for (size_t i = 0; i < PDL_ARRAY_LEN(cuda_impls); ++i) {
            if (cuda_impls[i] == PDL_IMPL_CUDA_CUBLAS && !pdl_has_cublas()) continue;
            pdl_ctx c;
            pdl_ctx_init(&c, PDL_DEV_CUDA, cuda_impls[i], 0, 0);
            for (int ta = 0; ta < 2; ++ta)
                for (int tb = 0; tb < 2; ++tb) gemm_case(&c, &ref, ta, tb, tol, seed, pdl_impl_name(cuda_impls[i]));
            if (cuda_impls[i] == PDL_IMPL_CUDA_TILED) {
                size_t gn = 0;
                float loss_gpu = 0;
                float *gpu_out = layer_pipeline(&c, seed, &gn, &loss_gpu);
                double e = rel_diff(gpu_out, cpu_out, n);
                report("kernels/cnn-pipeline-cuda-vs-cpu", "cuda", e < tol, e, tol,
                       "conv, direct conv, relu, maxpool fwd/bwd, col2im, softmax");
                double le = fabs(loss_gpu - loss_ref) / PDL_MAX(1e-6, fabs(loss_ref));
                report("kernels/softmax-loss-cuda-vs-cpu", "cuda", le < tol, le, tol, NULL);
                /* reductions and optimizer */
                const size_t vn = 100003;
                float *hv = (float *)pdl_xmalloc(vn * 4), *hg = (float *)pdl_xmalloc(vn * 4);
                float *hm = (float *)pdl_xcalloc(vn, 4), *hr = (float *)pdl_xmalloc(vn * 4);
                fill(hv, vn, seed, 41);
                fill(hg, vn, seed, 42);
                double s_cpu = pdl_sumsq(&ref, vn, hv);
                float *dv = (float *)pdl_dev_malloc(&c, vn * 4), *dg = (float *)pdl_dev_malloc(&c, vn * 4);
                float *dm = (float *)pdl_dev_malloc(&c, vn * 4);
                pdl_memcpy_h2d(&c, dv, hv, vn * 4);
                pdl_memcpy_h2d(&c, dg, hg, vn * 4);
                double s_gpu = pdl_sumsq(&c, vn, dv);
                report("kernels/sumsq-cuda-vs-cpu", "cuda", fabs(s_gpu - s_cpu) / s_cpu < 1e-6,
                       fabs(s_gpu - s_cpu) / s_cpu, 1e-6, NULL);
                for (int it = 0; it < 3; ++it) {
                    pdl_sgd_update(&ref, vn, 0.1f, 0.9f, 1e-4f, hv, hg, hm);
                    pdl_sgd_update(&c, vn, 0.1f, 0.9f, 1e-4f, dv, dg, dm);
                }
                pdl_memcpy_d2h(&c, hr, dv, vn * 4);
                double se = rel_diff(hr, hv, vn);
                report("kernels/sgd-momentum-cuda-vs-cpu", "cuda", se < tol, se, tol, NULL);
                pdl_dev_free(&c, dv), pdl_dev_free(&c, dg), pdl_dev_free(&c, dm);
                free(hv), free(hg), free(hm), free(hr);
                free(gpu_out);
            }
            pdl_ctx_destroy(&c);
        }
    }
    free(cpu_out);
    pdl_ctx_destroy(&ref);
}

/* ------------------------------------------------------------------------ */
/* Small training helpers                                                    */
/* ------------------------------------------------------------------------ */

#define TINY_SPEC "conv:4:3:1:1,relu,pool:2:2,fc:16,relu,fc:4"

static pdl_dataset *tiny_data(uint64_t seed) {
    pdl_dataset *lazy = pdl_dataset_load("synthetic:512:1:12:12:4", 0, seed);
    uint32_t *idx = NULL;
    int n = pdl_partition_indices(lazy, PDL_PART_GLOBAL, 0, 1, seed, &idx);
    pdl_dataset *d = pdl_dataset_subset(lazy, idx, n);
    free(idx);
    pdl_dataset_free(lazy);
    return d;
}

/* One SGD step on samples idx[0..b): returns summed loss. */
static double sgd_step(pdl_model *m, const pdl_dataset *d, const uint32_t *idx, int b, float *xb, int32_t *yb,
                       float *v, int allreduce_world, pdl_allreduce_algo algo, float *hbuf, float *scratch) {
    pdl_dataset_gather(d, idx, b, xb, yb);
    pdl_model_set_batch(m, xb, yb, b);
    int hits;
    double loss = pdl_model_forward(m, b, 1.0f / b, &hits);
    pdl_model_backward(m, b, NULL, NULL);
    if (allreduce_world > 1) {
        pdl_memcpy_d2h(m->ctx, hbuf, m->grads, m->num_params * 4);
        pdl_allreduce_sum(hbuf, m->num_params, algo, MPI_COMM_WORLD, scratch, NULL);
        pdl_memcpy_h2d(m->ctx, m->grads, hbuf, m->num_params * 4);
        pdl_scal(m->ctx, m->num_params, 1.0f / allreduce_world, m->grads);
    }
    pdl_sgd_update(m->ctx, m->num_params, 0.05f, 0.9f, 0.0f, m->params, m->grads, v);
    return loss;
}

static void test_determinism(pdl_ctx *ctx, int steps, uint64_t seed) {
    pdl_dataset *d = tiny_data(seed);
    uint64_t h[2];
    for (int run = 0; run < 2; ++run) {
        pdl_model *m = pdl_model_create(ctx, TINY_SPEC, 1, 12, 12, 4, 16);
        pdl_model_init_params(m, seed);
        float *v = (float *)pdl_dev_malloc(ctx, m->num_params * 4);
        float xb[16 * 144];
        int32_t yb[16];
        uint32_t idx[16];
        for (int s = 0; s < steps; ++s) {
            for (int i = 0; i < 16; ++i) idx[i] = (uint32_t)((s * 16 + i) % d->n);
            sgd_step(m, d, idx, 16, xb, yb, v, 1, PDL_AR_MPI, NULL, NULL);
        }
        h[run] = pdl_model_param_hash(m);
        pdl_dev_free(ctx, v);
        pdl_model_free(m);
    }
    report("determinism/same-seed-same-params", pdl_device_name(ctx->device), h[0] == h[1], h[0] == h[1] ? 0.0 : 1.0,
           0.0, "bitwise parameter hash");
    pdl_dataset_free(d);
}

static void test_finite(pdl_ctx *ctx) {
    float h[1000];
    for (int i = 0; i < 1000; ++i) h[i] = (float)i;
    h[17] = NAN;
    h[512] = INFINITY;
    float *d = (float *)pdl_dev_malloc(ctx, sizeof(h));
    pdl_memcpy_h2d(ctx, d, h, sizeof(h));
    size_t c = pdl_count_nonfinite(ctx, 1000, d);
    report("finite/detects-nan-and-inf", pdl_device_name(ctx->device), c == 2, (double)c, 2.0, "expects exactly 2");
    pdl_dev_free(ctx, d);
}

/* ------------------------------------------------------------------------ */
/* MPI tests                                                                 */
/* ------------------------------------------------------------------------ */

static void test_allreduce(int k, int rank) {
    const size_t sizes[] = {1, 7, 1000, (1u << 16) + 3};
    float *buf = (float *)pdl_xmalloc(sizes[3] * 4), *scratch = (float *)pdl_xmalloc(sizes[3] * 4);
    for (int a = 0; a < PDL_AR_COUNT; ++a) {
        int ok = 1, identical = 1;
        for (size_t si = 0; si < PDL_ARRAY_LEN(sizes); ++si) {
            size_t n = sizes[si];
            for (size_t i = 0; i < n; ++i) buf[i] = (float)((rank + 1) * (1 + (int)(i % 5)));
            pdl_allreduce_sum(buf, n, (pdl_allreduce_algo)a, MPI_COMM_WORLD, scratch, NULL);
            const float s = (float)k * (k + 1) / 2.0f;
            for (size_t i = 0; i < n; ++i)
                if (buf[i] != s * (float)(1 + (int)(i % 5))) ok = 0;
            /* non-integer data: results must still be bitwise identical on all ranks */
            pdl_rng r;
            pdl_rng_seed(&r, 9, (uint64_t)rank, si, 0);
            for (size_t i = 0; i < n; ++i) buf[i] = pdl_rng_uniformf(&r, -1.0f, 1.0f);
            pdl_allreduce_sum(buf, n, (pdl_allreduce_algo)a, MPI_COMM_WORLD, scratch, NULL);
            uint64_t hsh = 1469598103934665603ULL;
            const unsigned char *p = (const unsigned char *)buf;
            for (size_t i = 0; i < n * 4; ++i) hsh = (hsh ^ p[i]) * 1099511628211ULL;
            identical &= pdl_all_equal_u64(hsh, MPI_COMM_WORLD);
        }
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        char name[64];
        snprintf(name, sizeof(name), "allreduce/%s-exact-sum", pdl_allreduce_name((pdl_allreduce_algo)a));
        report(name, "cpu", ok, ok ? 0.0 : 1.0, 0.0, "sizes 1, 7, 1000, 65539");
        snprintf(name, sizeof(name), "allreduce/%s-identical-on-ranks", pdl_allreduce_name((pdl_allreduce_algo)a));
        report(name, "cpu", identical, identical ? 0.0 : 1.0, 0.0, NULL);
    }
    free(buf);
    free(scratch);
}

/* Sync SGD on k ranks (local batch b) vs sequential SGD on rank 0 (batch k*b). */
static void test_equivalence(pdl_ctx *ctx, int k, int rank, int steps, uint64_t seed) {
    const int b = 8, B = b * k;
    pdl_dataset *d = tiny_data(seed);
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)d->n * 4);
    for (int i = 0; i < d->n; ++i) order[i] = (uint32_t)i;
    pdl_rng r;
    pdl_rng_seed(&r, seed, 0xE9ULL, 0, 0);
    pdl_rng_shuffle_u32(&r, order, (size_t)d->n);

    pdl_model *dist = pdl_model_create(ctx, TINY_SPEC, 1, 12, 12, 4, b);
    pdl_model_init_params(dist, seed);
    float *vd = (float *)pdl_dev_malloc(ctx, dist->num_params * 4);
    float *hbuf = (float *)pdl_xmalloc(dist->num_params * 4), *scr = (float *)pdl_xmalloc(dist->num_params * 4);
    float *xb = (float *)pdl_xmalloc((size_t)B * 144 * 4);
    int32_t *yb = (int32_t *)pdl_xmalloc((size_t)B * 4);
    for (int s = 0; s < steps; ++s) {
        const uint32_t *gidx = order + ((size_t)s * B) % (size_t)(d->n - B);
        sgd_step(dist, d, gidx + (size_t)rank * b, b, xb, yb, vd, k, PDL_AR_RING, hbuf, scr);
    }
    float *pd = (float *)pdl_xmalloc(dist->num_params * 4);
    pdl_memcpy_d2h(ctx, pd, dist->params, dist->num_params * 4);

    double err = 0.0;
    if (rank == 0) {
        pdl_model *seq = pdl_model_create(ctx, TINY_SPEC, 1, 12, 12, 4, B);
        pdl_model_init_params(seq, seed);
        float *vs = (float *)pdl_dev_malloc(ctx, seq->num_params * 4);
        for (int s = 0; s < steps; ++s) {
            const uint32_t *gidx = order + ((size_t)s * B) % (size_t)(d->n - B);
            sgd_step(seq, d, gidx, B, xb, yb, vs, 1, PDL_AR_MPI, NULL, NULL);
        }
        float *ps = (float *)pdl_xmalloc(seq->num_params * 4);
        pdl_memcpy_d2h(ctx, ps, seq->params, seq->num_params * 4);
        err = rel_diff(pd, ps, seq->num_params);
        free(ps);
        pdl_dev_free(ctx, vs);
        pdl_model_free(seq);
    }
    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    char detail[96];
    snprintf(detail, sizeof(detail), "%d ranks x batch %d vs 1 rank x batch %d, %d steps", k, b, B, steps);
    report("equivalence/sync-sgd-vs-sequential", pdl_device_name(ctx->device), err < 1e-4, err, 1e-4, detail);
    free(pd), free(hbuf), free(scr), free(xb), free(yb), free(order);
    pdl_dev_free(ctx, vd);
    pdl_model_free(dist);
    pdl_dataset_free(d);
}

static void test_sync(pdl_ctx *ctx, int k, int rank, int steps, uint64_t seed) {
    pdl_dataset *d = tiny_data(seed);
    for (int a = 0; a < PDL_AR_COUNT; ++a) {
        pdl_model *m = pdl_model_create(ctx, TINY_SPEC, 1, 12, 12, 4, 8);
        pdl_model_init_params(m, seed);
        float *v = (float *)pdl_dev_malloc(ctx, m->num_params * 4);
        float *hbuf = (float *)pdl_xmalloc(m->num_params * 4), *scr = (float *)pdl_xmalloc(m->num_params * 4);
        float xb[8 * 144];
        int32_t yb[8];
        uint32_t idx[8];
        for (int s = 0; s < steps; ++s) {
            for (int i = 0; i < 8; ++i) idx[i] = (uint32_t)((s * 8 * k + rank * 8 + i) % d->n);
            sgd_step(m, d, idx, 8, xb, yb, v, k, (pdl_allreduce_algo)a, hbuf, scr);
        }
        int same = pdl_all_equal_u64(pdl_model_param_hash(m), MPI_COMM_WORLD);
        char name[64];
        snprintf(name, sizeof(name), "sync/replicas-identical-%s", pdl_allreduce_name((pdl_allreduce_algo)a));
        report(name, pdl_device_name(ctx->device), same, same ? 0.0 : 1.0, 0.0, "bitwise parameter hash");
        free(hbuf), free(scr);
        pdl_dev_free(ctx, v);
        pdl_model_free(m);
    }
    pdl_dataset_free(d);
}

/* ------------------------------------------------------------------------ */

static int wants(char **tests, int n, const char *name) {
    for (int i = 0; i < n; ++i)
        if (strcmp(tests[i], "all") == 0 || strcmp(tests[i], name) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    int rank = 0, k = 1;
    pdl_mpi_init(&argc, &argv, &rank, &k);
    g_rank = rank;
    pdl_cli cli;
    if (pdl_cli_parse(&cli, k_opts, (int)PDL_ARRAY_LEN(k_opts), argc, argv, rank == 0) != 0) {
        pdl_mpi_finalize();
        return 2;
    }
    if (cli.help) {
        if (rank == 0) pdl_cli_print_help(&cli, argv[0], "Correctness suite: gradients, kernels, determinism, MPI consistency.");
        pdl_mpi_finalize();
        return 0;
    }
    const int hosts = pdl_count_hosts(MPI_COMM_WORLD);
    g_verbose = (int)pdl_cli_int(&cli, "verbose");
    const double tol = pdl_cli_dbl(&cli, "tol");
    const int steps = (int)pdl_cli_int(&cli, "steps");
    const uint64_t seed = (uint64_t)pdl_cli_int(&cli, "seed");
    const char *dev = pdl_cli_str(&cli, "device");
    char **tests;
    int nt = pdl_cli_list(&cli, "tests", &tests);

    int devices[2], ndev = 0;
    if (strcmp(dev, "cpu") == 0 || strcmp(dev, "both") == 0) devices[ndev++] = PDL_DEV_CPU;
    if (strcmp(dev, "cuda") == 0 || strcmp(dev, "both") == 0) {
        if (pdl_has_cuda())
            devices[ndev++] = PDL_DEV_CUDA;
        else if (rank == 0)
            PDL_WARN("CUDA requested but not available in this build/machine — CUDA tests skipped");
    }
    PDL_CHECK(ndev > 0 || !wants(tests, nt, "gradcheck"), "no device to test");

    for (int di = 0; di < ndev; ++di) {
        pdl_ctx ctx;
        pdl_device d = (pdl_device)devices[di];
        pdl_ctx_init(&ctx, d, (pdl_impl)pdl_impl_parse("auto", d), 0, 0);
        if (rank == 0) {
            if (wants(tests, nt, "gradcheck")) test_gradcheck(&ctx, seed);
            if (wants(tests, nt, "determinism")) test_determinism(&ctx, steps, seed);
            if (wants(tests, nt, "finite")) test_finite(&ctx);
        }
        if (k > 1) {
            if (wants(tests, nt, "equivalence")) test_equivalence(&ctx, k, rank, steps, seed);
            if (wants(tests, nt, "sync")) test_sync(&ctx, k, rank, steps, seed);
        }
        pdl_ctx_destroy(&ctx);
    }
    if (rank == 0 && wants(tests, nt, "kernels")) {
        int with_cuda = 0;
        for (int di = 0; di < ndev; ++di) with_cuda |= devices[di] == PDL_DEV_CUDA;
        test_kernels(with_cuda, tol, seed);
    }
    if (k > 1 && wants(tests, nt, "allreduce")) test_allreduce(k, rank);
    if (k == 1 && rank == 0 &&
        (wants(tests, nt, "allreduce") || wants(tests, nt, "equivalence") || wants(tests, nt, "sync")))
        printf("[ SKIP ] MPI tests (allreduce, equivalence, sync) need >= 2 ranks: mpirun -np 4 %s\n", argv[0]);

    int failed = 0;
    for (int i = 0; i < g_nres; ++i) failed += !g_results[i].passed;
    const char *output = pdl_cli_str(&cli, "output");
    if (rank == 0 && output[0]) {
        pdl_json *j = pdl_json_open(output);
        if (j) {
            pdl_ctx envctx;
            pdl_ctx_init(&envctx, PDL_DEV_CPU, PDL_IMPL_CPU_BLOCKED, 0, 0);
            pdl_json_begin_object(j, NULL);
            pdl_json_kv_str(j, "schema", "pdl-bench/check/v1");
            pdl_json_kv_str(j, "version", PDL_VERSION_STRING);
            pdl_json_kv_str(j, "run_id", "");
            pdl_json_kv_str(j, "tag", "");
            pdl_cli_write_json(&cli, j);
            pdl_env_write_json(j, &envctx, argc, argv, k, hosts);
            pdl_json_kv_int(j, "world_size", k);
            pdl_json_kv_bool(j, "passed", failed == 0);
            pdl_json_begin_array(j, "tests");
            for (int i = 0; i < g_nres; ++i) {
                const result *r = &g_results[i];
                pdl_json_begin_object(j, NULL);
                pdl_json_kv_str(j, "name", r->name);
                pdl_json_kv_str(j, "device", r->device);
                pdl_json_kv_bool(j, "passed", r->passed);
                pdl_json_kv_num(j, "max_err", r->max_err);
                pdl_json_kv_num(j, "tolerance", r->tol);
                pdl_json_kv_str(j, "detail", r->detail);
                pdl_json_end_object(j);
            }
            pdl_json_end_array(j);
            pdl_json_end_object(j);
            pdl_json_close(j);
            pdl_ctx_destroy(&envctx);
        }
    }
    if (rank == 0) printf("\n%d tests, %d failed\n", g_nres, failed);
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    pdl_str_free_parts(tests, nt);
    free(tests);
    pdl_cli_free(&cli);
    pdl_mpi_finalize();
    return failed ? 1 : 0;
}
