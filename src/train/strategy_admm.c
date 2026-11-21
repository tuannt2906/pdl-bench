/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * strategy_admm.c — consensus ADMM (Boyd et al. 2011, §7.1; paper §5.1.2)
 * with stochastic local solves.
 *
 *   minimize Σ_i f_i(x_i)  s.t.  x_i = z      (f_i = loss on worker i's shard)
 *
 * Scaled-form iterations, one round = --admm-local-steps SGD steps:
 *   x_i ← argmin f_i(x) + ρ/2 ||x − z + u_i||²   (approximate: L SGD steps on
 *                                                 ∇f_i(x) + ρ (x − z + u_i))
 *   z   ← (1/k) Σ_i (x_i + u_i)                  (one allreduce of p floats)
 *   u_i ← u_i + x_i − z
 * Communication: one allreduce per round instead of per step (cost O(kp) per
 * round, paper's analysis). The consensus variable z is the model that is
 * evaluated. The primal residual ||x − z|| is reported per run.
 */
#include <math.h>

#include "pdl/common.h"
#include "pdl/train.h"

typedef struct {
    float *z, *u, *w; /* device buffers, num_params each */
} admm_state;

static void consensus(pdl_trainer *t, admm_state *s) {
    const pdl_ctx *ctx = &t->ctx;
    const size_t P = t->model->num_params;
    float *x = t->model->params;
    /* w = x + u ; z = mean(w) */
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    pdl_copy(ctx, P, x, s->w);
    pdl_axpy(ctx, P, 1.0f, s->u, s->w);
    pdl_phase_end(&t->phases);
    pdl_trainer_allreduce_device(t, s->w, P, (float)t->num_workers);
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    pdl_copy(ctx, P, s->w, s->z);
    /* u += x − z */
    pdl_axpy(ctx, P, 1.0f, x, s->u);
    pdl_axpy(ctx, P, -1.0f, s->z, s->u);
    /* primal residual r = x − z */
    pdl_copy(ctx, P, x, s->w);
    pdl_axpy(ctx, P, -1.0f, s->z, s->w);
    double r2 = pdl_sumsq(ctx, P, s->w);
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    if (t->world > 1) MPI_Allreduce(MPI_IN_PLACE, &r2, 1, MPI_DOUBLE, MPI_SUM, t->comm);
    pdl_phase_end(&t->phases);
    t->admm_primal_residual = sqrt(r2);
    t->admm_rounds++;
}

int pdl_run_admm(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const pdl_ctx *ctx = &t->ctx;
    const size_t P = t->model->num_params;
    const float rho = (float)c->admm_rho;
    const int L = c->admm_local_steps;
    admm_state s;
    s.z = (float *)pdl_dev_malloc(ctx, P * sizeof(float));
    s.u = (float *)pdl_dev_malloc(ctx, P * sizeof(float)); /* zero */
    s.w = (float *)pdl_dev_malloc(ctx, P * sizeof(float));
    pdl_copy(ctx, P, t->model->params, s.z);

    pdl_trainer_warmup(t, 0);
    MPI_Barrier(t->comm);
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)(t->n_local > 0 ? t->n_local : 1) * sizeof(uint32_t));
    for (int epoch = 1; epoch <= c->epochs; ++epoch) {
        pdl_trainer_begin_epoch(t, epoch);
        pdl_trainer_epoch_order(t, epoch, order);
        double loss_sum = 0.0;
        long correct = 0, samples = 0;
        int local = 0;
        const double t0 = pdl_now();
        for (int st = 0; st < t->steps_per_epoch; ++st) {
            int hits = 0;
            double loss = pdl_trainer_compute_gradients(t, pdl_trainer_batch(t, order, st), t->local_batch, 1.0f,
                                                        &hits, NULL, NULL);
            if (rho > 0.0f) {
                /* augmented-Lagrangian gradient: g += ρ (x − z + u) */
                pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
                pdl_axpy(ctx, P, rho, t->model->params, t->model->grads);
                pdl_axpy(ctx, P, -rho, s.z, t->model->grads);
                pdl_axpy(ctx, P, rho, s.u, t->model->grads);
                pdl_phase_end(&t->phases);
            }
            pdl_trainer_apply_update(t);
            loss_sum += loss;
            correct += hits;
            samples += t->local_batch;
            pdl_trainer_log_step(t, epoch, loss / t->local_batch, (double)hits / t->local_batch);
            pdl_trainer_check_finite(t, loss);
            if (++local == L || st == t->steps_per_epoch - 1) {
                consensus(t, &s);
                local = 0;
            }
        }
        const double epoch_time = pdl_now() - t0;
        t->train_time += epoch_time;
        double tl = -1.0, ta = -1.0;
        if (pdl_trainer_should_eval(t, epoch)) pdl_trainer_evaluate(t, s.z, 1, &tl, &ta);
        pdl_epoch_record *r = pdl_trainer_end_epoch(t, epoch, loss_sum, correct, samples, t->steps_per_epoch, epoch_time);
        r->test_loss = tl;
        r->test_acc = ta;
        if (t->rank == 0) {
            pdl_trainer_report_epoch(t, r);
            if (c->verbose) PDL_INFO("          admm rounds %ld, primal residual %.4e", t->admm_rounds, t->admm_primal_residual);
        }
    }
    /* final model = consensus variable */
    pdl_copy(ctx, P, s.z, t->model->params);
    pdl_ctx_sync(ctx);
    free(order);
    pdl_dev_free(ctx, s.z);
    pdl_dev_free(ctx, s.u);
    pdl_dev_free(ctx, s.w);
    return 0;
}
