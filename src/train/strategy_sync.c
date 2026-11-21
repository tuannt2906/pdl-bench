/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * strategy_sync.c — sequential baseline and synchronous data-parallel SGD
 * (paper §5.1: all loss gradients of a mini-batch are computed with the same
 * weights and the full average gradient is used for the update).
 *
 * Per step:  forward/backward on the local slice of the global batch
 *            → [barrier: sync_wait] → allreduce(grads)/k → SGD update.
 * Communication variants:
 *   --fusion flat   one allreduce over the fused gradient buffer
 *   --fusion layer  one allreduce per layer (more latency terms, same bytes)
 *   --overlap 1     per-layer MPI_Iallreduce issued as soon as backward has
 *                   produced that layer's gradients (comm/compute overlap)
 */
#include <string.h>

#include "pdl/common.h"
#include "pdl/train.h"

#define MAX_REQS 128

typedef struct {
    pdl_trainer *t;
    MPI_Request reqs[MAX_REQS];
    int nreq;
} overlap_state;

static void overlap_grad_ready(int layer, void *user) {
    overlap_state *ov = (overlap_state *)user;
    pdl_trainer *t = ov->t;
    const pdl_layer *L = &t->model->layers[layer];
    const size_t off = L->w_off, n = L->w_count + L->b_count;
    float *h = t->model->grads + off;
    if (t->ctx.device == PDL_DEV_CUDA) {
        pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
        h = t->host_buf + off;
        pdl_memcpy_d2h(&t->ctx, h, t->model->grads + off, n * sizeof(float));
    }
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    PDL_CHECK(ov->nreq < MAX_REQS, "too many layers for overlap mode");
    MPI_Iallreduce(MPI_IN_PLACE, h, (int)n, MPI_FLOAT, MPI_SUM, t->comm, &ov->reqs[ov->nreq++]);
    t->comm_stats.calls++;
    t->comm_stats.messages += 2 * (t->world - 1);
    t->comm_stats.bytes_sent += 2.0 * (t->world - 1) / t->world * (double)n * sizeof(float);
    pdl_phase_begin(&t->phases, PDL_PHASE_BACKWARD);
}

static void overlap_finish(pdl_trainer *t, overlap_state *ov) {
    const size_t P = t->model->num_params;
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    MPI_Waitall(ov->nreq, ov->reqs, MPI_STATUSES_IGNORE);
    ov->nreq = 0;
    if (t->ctx.device == PDL_DEV_CUDA) {
        pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
        pdl_memcpy_h2d(&t->ctx, t->model->grads, t->host_buf, P * sizeof(float));
    }
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    pdl_scal(&t->ctx, P, 1.0f / (float)t->num_workers, t->model->grads);
    pdl_phase_end(&t->phases);
}

static void sync_gradients(pdl_trainer *t) {
    pdl_model *m = t->model;
    const float k = (float)t->num_workers;
    if (t->cfg->fusion_layer) {
        for (int i = 0; i < m->num_layers; ++i) {
            const pdl_layer *L = &m->layers[i];
            if (L->w_count) pdl_trainer_allreduce_device(t, m->grads + L->w_off, L->w_count + L->b_count, k);
        }
    } else {
        pdl_trainer_allreduce_device(t, m->grads, m->num_params, k);
    }
}

static int run_data_parallel(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const int distributed = t->world > 1;
    const int use_overlap = distributed && c->overlap;
    overlap_state ov;
    memset(&ov, 0, sizeof(ov));
    ov.t = t;
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)(t->n_local > 0 ? t->n_local : 1) * sizeof(uint32_t));

    pdl_trainer_warmup(t, distributed);
    MPI_Barrier(t->comm);
    for (int epoch = 1; epoch <= c->epochs; ++epoch) {
        pdl_trainer_begin_epoch(t, epoch);
        pdl_trainer_epoch_order(t, epoch, order);
        double loss_sum = 0.0;
        long correct = 0, samples = 0;
        const double t0 = pdl_now();
        for (int s = 0; s < t->steps_per_epoch; ++s) {
            const uint32_t *idx = pdl_trainer_batch(t, order, s);
            int hits = 0;
            double loss;
            if (use_overlap) {
                loss = pdl_trainer_compute_gradients(t, idx, t->local_batch, 1.0f, &hits, overlap_grad_ready, &ov);
                overlap_finish(t, &ov);
            } else {
                loss = pdl_trainer_compute_gradients(t, idx, t->local_batch, 1.0f, &hits, NULL, NULL);
                if (distributed) sync_gradients(t);
            }
            pdl_trainer_apply_update(t);
            loss_sum += loss;
            correct += hits;
            samples += t->local_batch;
            pdl_trainer_log_step(t, epoch, loss / t->local_batch, (double)hits / t->local_batch);
            pdl_trainer_check_finite(t, loss);
            if (distributed && c->check_sync > 0 && t->global_step % c->check_sync == 0) pdl_trainer_check_sync(t);
        }
        const double epoch_time = pdl_now() - t0;
        t->train_time += epoch_time;
        double tl = -1.0, ta = -1.0;
        if (pdl_trainer_should_eval(t, epoch)) pdl_trainer_evaluate(t, NULL, 1, &tl, &ta);
        pdl_epoch_record *r = pdl_trainer_end_epoch(t, epoch, loss_sum, correct, samples, t->steps_per_epoch, epoch_time);
        r->test_loss = tl;
        r->test_acc = ta;
        if (t->rank == 0) pdl_trainer_report_epoch(t, r);
    }
    if (distributed && c->check_sync > 0) pdl_trainer_check_sync(t);
    free(order);
    return 0;
}

int pdl_run_sequential(pdl_trainer *t) { return run_data_parallel(t); }

int pdl_run_sync(pdl_trainer *t) { return run_data_parallel(t); }
