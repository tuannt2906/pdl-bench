/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * model.c — sequential CNN: spec parsing, presets, forward/backward passes.
 *
 * Convolution (per mini-batch of B images, K = C*kh*kw, P = OH*OW):
 *   forward : col[b]  = im2col(x[b])                    [K x P]
 *             y[b]    = W [OC x K] * col[b] + bias        (batched GEMM)
 *   backward: dW     += dy[b] [OC x P] * col[b]^T         (accumulated over b)
 *             dcol[b] = W^T [K x OC] * dy[b]              (batched GEMM)
 *             dx[b]   = col2im(dcol[b])
 * Dense: y = x W^T + b ; dW = dy^T x ; dx = dy W.
 */
#include "pdl/model.h"

#include <math.h>
#include <string.h>

#include "pdl/common.h"
#include "pdl/rng.h"

#define MAX_LAYERS 64

/* ------------------------------------------------------------------------ */
/* Presets                                                                   */
/* ------------------------------------------------------------------------ */

char *pdl_model_preset_spec(const char *name, int width, int num_classes) {
    if (strchr(name, ':') || strchr(name, ',')) return pdl_xstrdup(name);
    int w = width > 0 ? width : 1;
    char buf[1024];
    if (strcmp(name, "mlp") == 0) {
        snprintf(buf, sizeof(buf), "fc:%d,relu,fc:%d,relu,fc:%d", 256 * w, 256 * w, num_classes);
    } else if (strcmp(name, "lenet") == 0) {
        snprintf(buf, sizeof(buf),
                 "conv:%d:5:1:2,relu,pool:2:2,conv:%d:5:1:0,relu,pool:2:2,fc:%d,relu,fc:%d,relu,fc:%d",
                 6 * w, 16 * w, 120 * w, 84 * w, num_classes);
    } else if (strcmp(name, "cnn") == 0) {
        snprintf(buf, sizeof(buf), "conv:%d:3:1:1,relu,pool:2:2,conv:%d:3:1:1,relu,pool:2:2,fc:%d,relu,fc:%d",
                 16 * w, 32 * w, 128 * w, num_classes);
    } else if (strcmp(name, "cifar") == 0) {
        snprintf(buf, sizeof(buf),
                 "conv:%d:3:1:1,relu,conv:%d:3:1:1,relu,pool:2:2,"
                 "conv:%d:3:1:1,relu,conv:%d:3:1:1,relu,pool:2:2,fc:%d,relu,fc:%d",
                 32 * w, 32 * w, 64 * w, 64 * w, 256 * w, num_classes);
    } else if (strcmp(name, "paper") == 0) {
        /* Paper §3: first layer = 96 filters 11x11 stride 4 on 224x224 greyscale → 54x54. */
        snprintf(buf, sizeof(buf), "conv:%d:11:4:0,relu,pool:3:2,conv:%d:5:1:2,relu,pool:3:2,fc:%d,relu,fc:%d",
                 96 * w, 128 * w, 256 * w, num_classes);
    } else {
        PDL_FATAL("unknown model preset '%s' (mlp, lenet, cnn, cifar, paper or a spec string)", name);
    }
    return pdl_xstrdup(buf);
}

/* ------------------------------------------------------------------------ */
/* Construction                                                              */
/* ------------------------------------------------------------------------ */

static size_t layer_out_size(const pdl_layer *L) { return (size_t)L->out_c * L->out_h * L->out_w; }
static size_t layer_in_size(const pdl_layer *L) { return (size_t)L->in_c * L->in_h * L->in_w; }

static int parse_int(const char *s, const char *tok) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > 1 << 24) PDL_FATAL("bad number '%s' in layer '%s'", s, tok);
    return (int)v;
}

static int parse_nonneg(const char *s, const char *tok) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0' || v < 0 || v > 1 << 24) PDL_FATAL("bad number '%s' in layer '%s'", s, tok);
    return (int)v;
}

pdl_model *pdl_model_create(pdl_ctx *ctx, const char *spec, int in_c, int in_h, int in_w,
                            int num_classes, int max_batch) {
    pdl_model *m = (pdl_model *)pdl_xcalloc(1, sizeof(pdl_model));
    m->ctx = ctx;
    m->in_c = in_c;
    m->in_h = in_h;
    m->in_w = in_w;
    m->num_classes = num_classes;
    m->max_batch = max_batch;
    snprintf(m->spec, sizeof(m->spec), "%s", spec);
    m->layers = (pdl_layer *)pdl_xcalloc(MAX_LAYERS, sizeof(pdl_layer));

    char *tokens[MAX_LAYERS];
    int ntok = pdl_str_split(spec, ',', tokens, MAX_LAYERS);
    int c = in_c, h = in_h, w = in_w;
    size_t poff = 0;
    int nconv = 0, nfc = 0, nrelu = 0, npool = 0;
    for (int t = 0; t < ntok; ++t) {
        char *f[6];
        int nf = pdl_str_split(tokens[t], ':', f, 6);
        const char *kind = f[0];
        if (strcmp(kind, "flatten") == 0 || kind[0] == '\0') {
            pdl_str_free_parts(f, nf);
            continue;
        }
        PDL_CHECK(m->num_layers < MAX_LAYERS, "too many layers");
        pdl_layer *L = &m->layers[m->num_layers];
        L->in_c = c;
        L->in_h = h;
        L->in_w = w;
        if (strcmp(kind, "conv") == 0) {
            PDL_CHECK(nf >= 3, "conv needs conv:OUT:K[:STRIDE[:PAD]] (got '%s')", tokens[t]);
            L->kind = PDL_LAYER_CONV;
            L->out_c = parse_int(f[1], tokens[t]);
            L->k = parse_int(f[2], tokens[t]);
            L->stride = nf > 3 ? parse_int(f[3], tokens[t]) : 1;
            L->pad = nf > 4 ? parse_nonneg(f[4], tokens[t]) : 0;
            L->out_h = (h + 2 * L->pad - L->k) / L->stride + 1;
            L->out_w = (w + 2 * L->pad - L->k) / L->stride + 1;
            PDL_CHECK(L->out_h > 0 && L->out_w > 0, "conv '%s' produces empty output from %dx%d",
                      tokens[t], h, w);
            L->w_count = (size_t)L->out_c * c * L->k * L->k;
            L->b_count = (size_t)L->out_c;
            L->flops_fwd = 2.0 * L->out_c * L->out_h * L->out_w * (double)c * L->k * L->k;
            snprintf(L->name, sizeof(L->name), "conv%d", ++nconv);
        } else if (strcmp(kind, "fc") == 0 || strcmp(kind, "dense") == 0) {
            PDL_CHECK(nf >= 2, "fc needs fc:OUT (got '%s')", tokens[t]);
            L->kind = PDL_LAYER_DENSE;
            L->out_c = parse_int(f[1], tokens[t]);
            L->out_h = L->out_w = 1;
            size_t din = (size_t)c * h * w;
            L->w_count = (size_t)L->out_c * din;
            L->b_count = (size_t)L->out_c;
            L->flops_fwd = 2.0 * (double)din * L->out_c;
            snprintf(L->name, sizeof(L->name), "fc%d", ++nfc);
        } else if (strcmp(kind, "relu") == 0) {
            L->kind = PDL_LAYER_RELU;
            L->out_c = c;
            L->out_h = h;
            L->out_w = w;
            L->flops_fwd = (double)c * h * w;
            snprintf(L->name, sizeof(L->name), "relu%d", ++nrelu);
        } else if (strcmp(kind, "pool") == 0 || strcmp(kind, "maxpool") == 0) {
            PDL_CHECK(nf >= 2, "pool needs pool:K[:STRIDE] (got '%s')", tokens[t]);
            L->kind = PDL_LAYER_MAXPOOL;
            L->k = parse_int(f[1], tokens[t]);
            L->stride = nf > 2 ? parse_int(f[2], tokens[t]) : L->k;
            L->out_c = c;
            L->out_h = (h - L->k) / L->stride + 1;
            L->out_w = (w - L->k) / L->stride + 1;
            PDL_CHECK(L->out_h > 0 && L->out_w > 0, "pool '%s' on %dx%d input", tokens[t], h, w);
            L->flops_fwd = (double)c * L->out_h * L->out_w * L->k * L->k;
            snprintf(L->name, sizeof(L->name), "pool%d", ++npool);
        } else {
            PDL_FATAL("unknown layer '%s' in model spec", tokens[t]);
        }
        if (L->w_count) {
            L->w_off = poff;
            poff += L->w_count;
            L->b_off = poff;
            poff += L->b_count;
        }
        c = L->out_c;
        h = L->out_h;
        w = L->out_w;
        m->num_layers++;
        pdl_str_free_parts(f, nf);
    }
    pdl_str_free_parts(tokens, ntok);

    PDL_CHECK(m->num_layers > 0, "empty model spec");
    pdl_layer *last = &m->layers[m->num_layers - 1];
    PDL_CHECK(last->kind == PDL_LAYER_DENSE && last->out_c == num_classes,
              "the last layer must be fc:%d (number of classes)", num_classes);

    m->num_params = poff;
    m->params = (float *)pdl_dev_malloc(ctx, poff * sizeof(float));
    m->grads = (float *)pdl_dev_malloc(ctx, poff * sizeof(float));
    m->input = (float *)pdl_dev_malloc(ctx, (size_t)max_batch * in_c * in_h * in_w * sizeof(float));
    m->labels = (int32_t *)pdl_dev_malloc(ctx, (size_t)max_batch * sizeof(int32_t));
    for (int i = 0; i < m->num_layers; ++i) {
        pdl_layer *L = &m->layers[i];
        size_t out_n = (size_t)max_batch * layer_out_size(L);
        L->out = (float *)pdl_dev_malloc(ctx, out_n * sizeof(float));
        L->dout = (float *)pdl_dev_malloc(ctx, out_n * sizeof(float));
        if (L->kind == PDL_LAYER_CONV) {
            size_t col_n = (size_t)max_batch * L->in_c * L->k * L->k * L->out_h * L->out_w;
            L->col = (float *)pdl_dev_malloc(ctx, col_n * sizeof(float));
        } else if (L->kind == PDL_LAYER_MAXPOOL) {
            L->argmax = (int32_t *)pdl_dev_malloc(ctx, out_n * sizeof(int32_t));
        }
    }
    m->dlogits = last->dout;
    return m;
}

void pdl_model_free(pdl_model *m) {
    if (!m) return;
    const pdl_ctx *ctx = m->ctx;
    for (int i = 0; i < m->num_layers; ++i) {
        pdl_dev_free(ctx, m->layers[i].out);
        pdl_dev_free(ctx, m->layers[i].dout);
        pdl_dev_free(ctx, m->layers[i].col);
        pdl_dev_free(ctx, m->layers[i].argmax);
    }
    pdl_dev_free(ctx, m->params);
    pdl_dev_free(ctx, m->grads);
    pdl_dev_free(ctx, m->input);
    pdl_dev_free(ctx, m->labels);
    free(m->layers);
    free(m);
}

void pdl_model_init_params(pdl_model *m, uint64_t seed) {
    float *host = (float *)pdl_xcalloc(m->num_params ? m->num_params : 1, sizeof(float));
    for (int i = 0; i < m->num_layers; ++i) {
        pdl_layer *L = &m->layers[i];
        if (!L->w_count) continue;
        pdl_rng r;
        pdl_rng_seed(&r, seed, 0xA11CEULL, (uint64_t)i, 0);
        double fan_in = (double)L->w_count / (double)L->out_c;
        float std = (float)sqrt(2.0 / fan_in); /* He et al. for ReLU networks */
        for (size_t k = 0; k < L->w_count; ++k) host[L->w_off + k] = std * pdl_rng_normal(&r);
        /* biases stay zero */
    }
    pdl_memcpy_h2d(m->ctx, m->params, host, m->num_params * sizeof(float));
    free(host);
}

void pdl_model_set_batch(pdl_model *m, const float *x_host, const int32_t *y_host, int batch) {
    PDL_CHECK(batch <= m->max_batch, "batch %d exceeds model max_batch %d", batch, m->max_batch);
    size_t in_n = (size_t)batch * m->in_c * m->in_h * m->in_w;
    pdl_memcpy_h2d(m->ctx, m->input, x_host, in_n * sizeof(float));
    if (y_host) pdl_memcpy_h2d(m->ctx, m->labels, y_host, (size_t)batch * sizeof(int32_t));
}

/* ------------------------------------------------------------------------ */
/* Forward / backward                                                        */
/* ------------------------------------------------------------------------ */

static const float *layer_input(const pdl_model *m, int i) {
    return i == 0 ? m->input : m->layers[i - 1].out;
}

static void forward_layers(pdl_model *m, int B) {
    const pdl_ctx *ctx = m->ctx;
    for (int i = 0; i < m->num_layers; ++i) {
        pdl_layer *L = &m->layers[i];
        const float *x = layer_input(m, i);
        const float *W = m->params + L->w_off;
        const float *bias = m->params + L->b_off;
        switch (L->kind) {
        case PDL_LAYER_CONV: {
            const int K = L->in_c * L->k * L->k;
            const int P = L->out_h * L->out_w;
            pdl_im2col(ctx, x, B, L->in_c, L->in_h, L->in_w, L->k, L->k, L->stride, L->pad,
                       L->out_h, L->out_w, L->col);
            pdl_gemm_strided_batched(ctx, 0, 0, L->out_c, P, K, 1.0f, W, K, 0, L->col, P,
                                     (long long)K * P, 0.0f, L->out, P, (long long)L->out_c * P, B);
            pdl_add_bias_nchw(ctx, L->out, bias, B, L->out_c, P);
            break;
        }
        case PDL_LAYER_DENSE: {
            const int din = (int)layer_in_size(L);
            pdl_gemm(ctx, 0, 1, B, L->out_c, din, 1.0f, x, din, W, din, 0.0f, L->out, L->out_c);
            pdl_add_bias_rows(ctx, L->out, bias, B, L->out_c);
            break;
        }
        case PDL_LAYER_RELU:
            pdl_relu_forward(ctx, (size_t)B * layer_out_size(L), x, L->out);
            break;
        case PDL_LAYER_MAXPOOL:
            pdl_maxpool_forward(ctx, x, B, L->in_c, L->in_h, L->in_w, L->k, L->stride, L->out_h,
                                L->out_w, L->out, L->argmax);
            break;
        }
    }
}

double pdl_model_forward(pdl_model *m, int B, float grad_scale, int *correct) {
    PDL_CHECK(B > 0 && B <= m->max_batch, "invalid batch %d (max %d)", B, m->max_batch);
    forward_layers(m, B);
    pdl_layer *last = &m->layers[m->num_layers - 1];
    return pdl_softmax_xent(m->ctx, last->out, m->labels, B, m->num_classes, grad_scale, last->dout,
                            correct);
}

double pdl_model_eval_batch(pdl_model *m, int B, int *correct) {
    PDL_CHECK(B > 0 && B <= m->max_batch, "invalid batch %d (max %d)", B, m->max_batch);
    forward_layers(m, B);
    pdl_layer *last = &m->layers[m->num_layers - 1];
    return pdl_softmax_xent(m->ctx, last->out, m->labels, B, m->num_classes, 1.0f, NULL, correct);
}

void pdl_model_backward(pdl_model *m, int B, pdl_grad_ready_fn cb, void *user) {
    const pdl_ctx *ctx = m->ctx;
    pdl_dev_memset0(ctx, m->grads, m->num_params * sizeof(float));
    for (int i = m->num_layers - 1; i >= 0; --i) {
        pdl_layer *L = &m->layers[i];
        const float *x = layer_input(m, i);
        float *dx = i > 0 ? m->layers[i - 1].dout : NULL; /* no input gradient for layer 0 */
        const float *W = m->params + L->w_off;
        float *dW = m->grads + L->w_off;
        float *db = m->grads + L->b_off;
        switch (L->kind) {
        case PDL_LAYER_CONV: {
            const int K = L->in_c * L->k * L->k;
            const int P = L->out_h * L->out_w;
            /* dW = sum_b dy[b] * col[b]^T  (shared output → accumulated) */
            pdl_gemm_strided_batched(ctx, 0, 1, L->out_c, K, P, 1.0f, L->dout, P,
                                     (long long)L->out_c * P, L->col, P, (long long)K * P, 0.0f, dW,
                                     K, 0, B);
            pdl_bias_grad_nchw(ctx, L->dout, db, B, L->out_c, P);
            if (dx) {
                /* dcol[b] = W^T dy[b]; the im2col buffer is reused as dcol. */
                pdl_gemm_strided_batched(ctx, 1, 0, K, P, L->out_c, 1.0f, W, K, 0, L->dout, P,
                                         (long long)L->out_c * P, 0.0f, L->col, P, (long long)K * P, B);
                pdl_dev_memset0(ctx, dx, (size_t)B * layer_in_size(L) * sizeof(float));
                pdl_col2im(ctx, L->col, B, L->in_c, L->in_h, L->in_w, L->k, L->k, L->stride, L->pad,
                           L->out_h, L->out_w, dx);
            }
            break;
        }
        case PDL_LAYER_DENSE: {
            const int din = (int)layer_in_size(L);
            pdl_gemm(ctx, 1, 0, L->out_c, din, B, 1.0f, L->dout, L->out_c, x, din, 0.0f, dW, din);
            pdl_bias_grad_rows(ctx, L->dout, db, B, L->out_c);
            if (dx) pdl_gemm(ctx, 0, 0, B, din, L->out_c, 1.0f, L->dout, L->out_c, W, din, 0.0f, dx, din);
            break;
        }
        case PDL_LAYER_RELU:
            if (dx) pdl_relu_backward(ctx, (size_t)B * layer_out_size(L), x, L->dout, dx);
            break;
        case PDL_LAYER_MAXPOOL:
            if (dx) {
                pdl_dev_memset0(ctx, dx, (size_t)B * layer_in_size(L) * sizeof(float));
                pdl_maxpool_backward(ctx, L->dout, L->argmax, B, L->in_c, L->in_h, L->in_w, L->k,
                                     L->stride, L->out_h, L->out_w, dx);
            }
            break;
        }
        if (cb && L->w_count) cb(i, user);
    }
}

/* ------------------------------------------------------------------------ */
/* Introspection                                                             */
/* ------------------------------------------------------------------------ */

double pdl_model_flops_per_sample(const pdl_model *m) {
    double f = 0.0;
    for (int i = 0; i < m->num_layers; ++i) f += m->layers[i].flops_fwd;
    return f;
}

static const char *kind_name(pdl_layer_kind k) {
    switch (k) {
    case PDL_LAYER_CONV: return "conv";
    case PDL_LAYER_DENSE: return "fc";
    case PDL_LAYER_RELU: return "relu";
    case PDL_LAYER_MAXPOOL: return "maxpool";
    }
    return "?";
}

void pdl_model_print(const pdl_model *m, FILE *f) {
    fprintf(f, "Model: %s\n", m->spec);
    fprintf(f, "  %-8s %-8s %-16s %-16s %12s %14s\n", "layer", "kind", "input", "output", "params",
            "MFLOP/sample");
    for (int i = 0; i < m->num_layers; ++i) {
        const pdl_layer *L = &m->layers[i];
        char in[32], out[32];
        snprintf(in, sizeof(in), "%dx%dx%d", L->in_c, L->in_h, L->in_w);
        snprintf(out, sizeof(out), "%dx%dx%d", L->out_c, L->out_h, L->out_w);
        fprintf(f, "  %-8s %-8s %-16s %-16s %12zu %14.3f\n", L->name, kind_name(L->kind), in, out,
                L->w_count + L->b_count, L->flops_fwd / 1e6);
    }
    fprintf(f, "  total params: %zu (%.2f MB), forward: %.3f MFLOP/sample\n", m->num_params,
            m->num_params * 4.0 / 1e6, pdl_model_flops_per_sample(m) / 1e6);
}

uint64_t pdl_model_param_hash(pdl_model *m) {
    size_t bytes = m->num_params * sizeof(float);
    unsigned char *h = (unsigned char *)pdl_xmalloc(bytes ? bytes : 1);
    pdl_memcpy_d2h(m->ctx, h, m->params, bytes);
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= h[i];
        hash *= 0x100000001b3ULL;
    }
    free(h);
    return hash;
}

const char *pdl_layer_kind_name(pdl_layer_kind k) { return kind_name(k); }
