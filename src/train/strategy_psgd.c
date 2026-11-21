/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * strategy_psgd.c — Parallel SGD (Zinkevich et al. 2010; paper §5.1.1,
 * Algorithms 1–2).
 *
 * Each worker runs plain SGD on its own shard; models are averaged
 *   v = (1/k) Σ v_i
 * every --avg-period local steps (0 = once per epoch, -1 = one-shot at the end
 * of training, the original algorithm). Because deep-learning losses are not
 * strongly convex, the paper suggests re-shuffling data between machines and
 * repeating: --reshuffle 1 performs ShuffleSGD — a balanced all-to-all that
 * sends 1/k of every shard to every worker after each epoch (communication
 * cost O(D) bytes in total, measured in the "shuffle" phase).
 */
#include <limits.h>
#include <string.h>

#include "pdl/common.h"
#include "pdl/rng.h"
#include "pdl/train.h"

#define TAG_SHUFFLE 7101

static void average_params(pdl_trainer *t) {
    pdl_trainer_allreduce_device(t, t->model->params, t->model->num_params, (float)t->num_workers);
}

/* ShuffleSGD: balanced all-to-all redistribution of the local shards. */
static void shuffle_sgd(pdl_trainer *t, int epoch) {
    const int k = t->world, n = t->n_local;
    const size_t sb = t->train->sample_bytes;
    pdl_phase_begin(&t->phases, PDL_PHASE_SHUFFLE);

    uint32_t *perm = (uint32_t *)pdl_xmalloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
    for (int i = 0; i < n; ++i) perm[i] = (uint32_t)i;
    pdl_rng r;
    pdl_rng_seed(&r, (uint64_t)t->cfg->seed, 0x5F5FULL, (uint64_t)epoch, (uint64_t)t->rank);
    pdl_rng_shuffle_u32(&r, perm, (size_t)n);

    int *scnt = (int *)pdl_xcalloc((size_t)k, sizeof(int)), *rcnt = (int *)pdl_xcalloc((size_t)k, sizeof(int));
    int *sdis = (int *)pdl_xcalloc((size_t)k, sizeof(int)), *rdis = (int *)pdl_xcalloc((size_t)k, sizeof(int));
    int *sbytes = (int *)pdl_xcalloc((size_t)k, sizeof(int)), *rbytes = (int *)pdl_xcalloc((size_t)k, sizeof(int));
    int *sbdis = (int *)pdl_xcalloc((size_t)k, sizeof(int)), *rbdis = (int *)pdl_xcalloc((size_t)k, sizeof(int));
    for (int j = 0; j < k; ++j) scnt[j] = (int)((long)n * (j + 1) / k - (long)n * j / k);
    MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, t->comm);
    int nrecv = 0;
    for (int j = 0; j < k; ++j) {
        PDL_CHECK((double)scnt[j] * sb < INT_MAX && (double)rcnt[j] * sb < INT_MAX,
                  "shuffle chunk exceeds 2 GiB; use more ranks or fewer samples");
        sdis[j] = j ? sdis[j - 1] + scnt[j - 1] : 0;
        rdis[j] = j ? rdis[j - 1] + rcnt[j - 1] : 0;
        sbytes[j] = (int)(scnt[j] * sb);
        rbytes[j] = (int)(rcnt[j] * sb);
        sbdis[j] = (int)(sdis[j] * sb);
        rbdis[j] = (int)(rdis[j] * sb);
        nrecv += rcnt[j];
    }
    PDL_CHECK((double)n * sb < INT_MAX && (double)nrecv * sb < INT_MAX, "shard exceeds 2 GiB");
    uint8_t *spix = (uint8_t *)pdl_xmalloc((size_t)(n > 0 ? n : 1) * sb);
    uint8_t *slab = (uint8_t *)pdl_xmalloc((size_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; ++i) {
        memcpy(spix + (size_t)i * sb, t->train->pixels + (size_t)perm[i] * sb, sb);
        slab[i] = t->train->labels[perm[i]];
    }
    pdl_dataset *nd = pdl_dataset_alloc(nrecv, t->train->c, t->train->h, t->train->w, t->train->num_classes);
    MPI_Alltoallv(spix, sbytes, sbdis, MPI_BYTE, nd->pixels, rbytes, rbdis, MPI_BYTE, t->comm);
    MPI_Alltoallv(slab, scnt, sdis, MPI_BYTE, nd->labels, rcnt, rdis, MPI_BYTE, t->comm);
    for (int j = 0; j < k; ++j)
        if (j != t->rank) {
            t->comm_stats.bytes_sent += (double)sbytes[j] + scnt[j];
            t->comm_stats.messages += 2;
        }
    t->comm_stats.calls += 1;
    snprintf(nd->source, sizeof(nd->source), "%s", t->train->source);
    pdl_dataset_free(t->train);
    t->train = nd;
    t->n_local = nrecv;

    free(perm);
    free(scnt), free(rcnt), free(sdis), free(rdis), free(sbytes), free(rbytes), free(sbdis), free(rbdis);
    free(spix);
    free(slab);
    pdl_phase_end(&t->phases);
}

/* Evaluates the average of all replicas without changing them (one-shot mode). */
static void evaluate_average(pdl_trainer *t, double *tl, double *ta) {
    const size_t P = t->model->num_params;
    float *avg = (float *)pdl_dev_malloc(&t->ctx, P * sizeof(float));
    float *h = (float *)pdl_xmalloc(P * sizeof(float));
    pdl_phase_begin(&t->phases, PDL_PHASE_EVAL);
    pdl_memcpy_d2h(&t->ctx, h, t->model->params, P * sizeof(float));
    MPI_Allreduce(MPI_IN_PLACE, h, (int)P, MPI_FLOAT, MPI_SUM, t->comm);
    for (size_t i = 0; i < P; ++i) h[i] /= (float)t->world;
    pdl_memcpy_h2d(&t->ctx, avg, h, P * sizeof(float));
    pdl_phase_end(&t->phases);
    pdl_trainer_evaluate(t, avg, 1, tl, ta);
    free(h);
    pdl_dev_free(&t->ctx, avg);
}

int pdl_run_psgd(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const int H = c->avg_period;
    const int distributed = t->world > 1;
    pdl_trainer_warmup(t, 0);
    MPI_Barrier(t->comm);
    for (int epoch = 1; epoch <= c->epochs; ++epoch) {
        pdl_trainer_begin_epoch(t, epoch);
        uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)(t->n_local > 0 ? t->n_local : 1) * sizeof(uint32_t));
        pdl_trainer_epoch_order(t, epoch, order);
        double loss_sum = 0.0;
        long correct = 0, samples = 0;
        int since_avg = 0;
        const double t0 = pdl_now();
        for (int s = 0; s < t->steps_per_epoch; ++s) {
            int hits = 0;
            double loss = pdl_trainer_compute_gradients(t, pdl_trainer_batch(t, order, s), t->local_batch, 1.0f,
                                                        &hits, NULL, NULL);
            pdl_trainer_apply_update(t);
            since_avg++;
            loss_sum += loss;
            correct += hits;
            samples += t->local_batch;
            pdl_trainer_log_step(t, epoch, loss / t->local_batch, (double)hits / t->local_batch);
            pdl_trainer_check_finite(t, loss);
            if (distributed && H > 0 && since_avg == H) {
                average_params(t);
                since_avg = 0;
            }
        }
        /* end-of-round averaging: every epoch for H >= 0, only the last epoch for one-shot */
        if (distributed && since_avg > 0 && (H >= 0 || epoch == c->epochs)) average_params(t);
        if (distributed && c->reshuffle && epoch < c->epochs) {
            shuffle_sgd(t, epoch);
            pdl_trainer_recompute_steps(t, t->comm);
        }
        const double epoch_time = pdl_now() - t0;
        t->train_time += epoch_time;
        free(order);

        double tl = -1.0, ta = -1.0;
        if (pdl_trainer_should_eval(t, epoch)) {
            if (distributed && H < 0 && epoch < c->epochs)
                evaluate_average(t, &tl, &ta);
            else
                pdl_trainer_evaluate(t, NULL, 1, &tl, &ta);
        }
        pdl_epoch_record *r = pdl_trainer_end_epoch(t, epoch, loss_sum, correct, samples, t->steps_per_epoch, epoch_time);
        r->test_loss = tl;
        r->test_acc = ta;
        if (t->rank == 0) pdl_trainer_report_epoch(t, r);
    }
    if (distributed) pdl_trainer_check_sync(t); /* replicas must agree after the final average */
    return 0;
}
