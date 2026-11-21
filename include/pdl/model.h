/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * model.h — CNN layers and sequential models with a flat parameter buffer.
 *
 * Design notes
 *  - All parameters live in ONE contiguous buffer `params` (and gradients in
 *    `grads`) so that gradient synchronization is a single collective over a
 *    float array of length `num_params` ("fused" mode). Each layer stores
 *    offsets into those buffers, which also enables per-layer communication.
 *  - Convolution uses im2col + GEMM (Caffe style, paper §3.1), batched over the
 *    images of a mini-batch.
 *  - Model architectures are described by a compact spec string, e.g.
 *        "conv:32:5:1:2,relu,pool:2:2,fc:128,relu,fc:10"
 *    or by a preset name (see pdl_model_preset_spec).
 */
#ifndef PDL_MODEL_H
#define PDL_MODEL_H

#include <stdio.h>

#include "pdl/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PDL_LAYER_CONV = 0,
    PDL_LAYER_DENSE,
    PDL_LAYER_RELU,
    PDL_LAYER_MAXPOOL,
} pdl_layer_kind;

typedef struct {
    pdl_layer_kind kind;
    char name[32];
    /* per-sample shapes (C, H, W); dense layers use H = W = 1 */
    int in_c, in_h, in_w;
    int out_c, out_h, out_w;
    /* conv / pool hyper-parameters */
    int k, stride, pad;
    /* parameter slices in the model flat buffers (counts in floats) */
    size_t w_off, w_count, b_off, b_count;
    /* buffers on device, sized for max_batch */
    float *out;      /* activations  [B][out_c*out_h*out_w] */
    float *dout;     /* d loss / d out (same shape)          */
    float *col;      /* conv: im2col workspace [B][K][OH*OW] */
    int32_t *argmax; /* maxpool: argmax indices              */
    /* FLOP count for one sample (forward), used for reporting */
    double flops_fwd;
} pdl_layer;

/* Optional callback fired during backward right after layer `idx` has
 * finished writing its parameter gradients (used for comm/compute overlap). */
typedef void (*pdl_grad_ready_fn)(int layer_idx, void *user);

typedef struct {
    pdl_ctx *ctx;
    int num_layers;
    pdl_layer *layers;
    int in_c, in_h, in_w; /* input sample shape */
    int num_classes;
    int max_batch;
    size_t num_params;
    float *params;   /* device */
    float *grads;    /* device */
    float *input;    /* device [max_batch][in] */
    int32_t *labels; /* device [max_batch] */
    float *dlogits;  /* alias of last layer dout */
    char spec[512];
} pdl_model;

/* Resolves preset names ("lenet", "cnn", "mlp", "paper", "cifar") into a spec
 * string; `width` scales channel/hidden sizes (model-size scaling). Returns a
 * newly allocated string. If `name` already is a spec it is copied verbatim. */
char *pdl_model_preset_spec(const char *name, int width, int num_classes);

/* Builds the model and allocates all buffers on ctx->device. */
pdl_model *pdl_model_create(pdl_ctx *ctx, const char *spec, int in_c, int in_h, int in_w,
                            int num_classes, int max_batch);
void pdl_model_free(pdl_model *m);

/* He-normal init for weights, zeros for biases; deterministic in `seed`. */
void pdl_model_init_params(pdl_model *m, uint64_t seed);

/* Copies a host batch (normalized floats, int labels) into the model input. */
void pdl_model_set_batch(pdl_model *m, const float *x_host, const int32_t *y_host, int batch);

/* Forward pass on the first `batch` samples of m->input. Returns the summed
 * loss; *correct receives the number of correct predictions. dlogits are
 * scaled by grad_scale (normally 1/batch). */
double pdl_model_forward(pdl_model *m, int batch, float grad_scale, int *correct);
/* Backward pass; overwrites m->grads (gradients are NOT accumulated). */
void pdl_model_backward(pdl_model *m, int batch, pdl_grad_ready_fn cb, void *user);
/* Inference only (no gradient). */
double pdl_model_eval_batch(pdl_model *m, int batch, int *correct);

/* Total forward FLOPs per sample (multiply-add = 2 FLOPs). */
double pdl_model_flops_per_sample(const pdl_model *m);
void pdl_model_print(const pdl_model *m, FILE *f);
const char *pdl_layer_kind_name(pdl_layer_kind k);
/* FNV-1a hash over the raw parameter bytes (host copy) — model-sync checks. */
uint64_t pdl_model_param_hash(pdl_model *m);

#ifdef __cplusplus
}
#endif

#endif /* PDL_MODEL_H */
