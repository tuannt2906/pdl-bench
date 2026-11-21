/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * strategy_downpour.c — asynchronous Downpour SGD (Dean et al. 2012; paper
 * §5.2, Algorithm 4) with a parameter server on rank 0.
 *
 * Worker loop (ranks 1..k):
 *   every n_fetch steps : request parameters (non-blocking; applied when they arrive)
 *   every step          : gradient on local mini-batch, local SGD update,
 *                         accumulate gradient
 *   every n_push steps  : send accumulated gradient (non-blocking)
 * Server loop (rank 0): serve messages in arrival order —
 *   GRAD  → w ← w − η_s · Σg ; version++ ; staleness = version − worker's base version
 *   FETCH → send [version, w]
 *   DONE  → retire worker
 * The server evaluates the model at "epoch equivalents" (every
 * workers·⌈steps/n_push⌉ received pushes). Paper §5.2.1: once the number of
 * workers exceeds T/τ (gradient time / server update time) adding workers no
 * longer helps — the server's sync_wait (idle) vs update/comm time exposes this.
 */
#include <math.h>
#include <string.h>

#include "pdl/common.h"
#include "pdl/train.h"

enum { TAG_GRAD = 7201, TAG_FETCH = 7202, TAG_PARAMS = 7203, TAG_DONE = 7204 };

/* ---------------------------------------------------------------------- */
/* Server                                                                  */
/* ---------------------------------------------------------------------- */

static void server_epoch(pdl_trainer *t, int epoch, double *t_start, long updates_before) {
    const double epoch_time = pdl_now() - *t_start;
    t->train_time += epoch_time;
    double tl = -1.0, ta = -1.0;
    if (pdl_trainer_should_eval(t, epoch)) pdl_trainer_evaluate(t, NULL, 0, &tl, &ta);
    pdl_epoch_record *r = pdl_trainer_end_epoch(t, epoch, 0.0, 0, 0, t->server_updates - updates_before, epoch_time);
    r->test_loss = tl;
    r->test_acc = ta;
    pdl_trainer_report_epoch(t, r);
    if (t->cfg->verbose)
        PDL_INFO("          server updates %ld, mean staleness %.2f", t->server_updates,
                 t->server_updates ? t->staleness_sum / t->server_updates : 0.0);
    pdl_trainer_begin_epoch(t, epoch + 1);
    *t_start = pdl_now(); /* evaluation time is excluded from training time */
}

static void run_server(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const size_t P = t->model->num_params;
    const float lr = (float)(c->server_lr > 0.0 ? c->server_lr : c->lr);
    const long pushes_per_epoch =
        (long)t->num_workers * ((t->steps_per_epoch + c->push_every - 1) / c->push_every);
    float *recv = t->host_scratch; /* [steps, base_version, grad...] */
    float *send = t->host_buf;     /* [version, params...] */
    long version = 0, pushes = 0, updates_before = 0;
    int active = t->num_workers, epoch = 0;

    pdl_trainer_begin_epoch(t, 1);
    double t_start = pdl_now();
    while (active > 0) {
        MPI_Status st;
        pdl_phase_begin(&t->phases, PDL_PHASE_SYNC_WAIT); /* idle: waiting for any worker */
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, t->comm, &st);
        switch (st.MPI_TAG) {
        case TAG_GRAD: {
            pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
            MPI_Recv(recv, (int)(P + 2), MPI_FLOAT, st.MPI_SOURCE, TAG_GRAD, t->comm, MPI_STATUS_IGNORE);
            long stale = version - (long)recv[1];
            t->staleness_sum += (double)stale;
            if (stale > t->staleness_max) t->staleness_max = stale;
            pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
            pdl_memcpy_h2d(&t->ctx, t->model->grads, recv + 2, P * sizeof(float));
            pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
            pdl_axpy(&t->ctx, P, -lr, t->model->grads, t->model->params);
            pdl_ctx_timing_sync(&t->ctx);
            pdl_phase_end(&t->phases);
            version++;
            t->server_updates++;
            pushes++;
            if (epoch < c->epochs && pushes >= (long)(epoch + 1) * pushes_per_epoch) {
                server_epoch(t, ++epoch, &t_start, updates_before);
                updates_before = t->server_updates;
            }
            break;
        }
        case TAG_FETCH: {
            pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
            MPI_Recv(NULL, 0, MPI_FLOAT, st.MPI_SOURCE, TAG_FETCH, t->comm, MPI_STATUS_IGNORE);
            pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
            pdl_memcpy_d2h(&t->ctx, send + 1, t->model->params, P * sizeof(float));
            send[0] = (float)version;
            pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
            MPI_Send(send, (int)(P + 1), MPI_FLOAT, st.MPI_SOURCE, TAG_PARAMS, t->comm);
            t->comm_stats.bytes_sent += (double)(P + 1) * sizeof(float);
            t->comm_stats.messages++;
            t->comm_stats.calls++;
            pdl_phase_end(&t->phases);
            break;
        }
        case TAG_DONE:
            pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
            MPI_Recv(NULL, 0, MPI_FLOAT, st.MPI_SOURCE, TAG_DONE, t->comm, MPI_STATUS_IGNORE);
            pdl_phase_end(&t->phases);
            active--;
            break;
        default: PDL_FATAL("server: unexpected tag %d from rank %d", st.MPI_TAG, st.MPI_SOURCE);
        }
    }
    pdl_phase_end(&t->phases);
    while (epoch < c->epochs) { /* workers finished before the last boundary */
        server_epoch(t, ++epoch, &t_start, updates_before);
        updates_before = t->server_updates;
    }
}

/* ---------------------------------------------------------------------- */
/* Worker                                                                  */
/* ---------------------------------------------------------------------- */

typedef struct {
    float *acc;        /* device: accumulated gradients since last push */
    float *fetch_buf;  /* host: [version, params...] */
    MPI_Request push_req, fetch_req;
    long version;      /* server version of the parameters we last received */
    int since_push;
} worker_state;

static void push_gradients(pdl_trainer *t, worker_state *w) {
    const size_t P = t->model->num_params;
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    MPI_Wait(&w->push_req, MPI_STATUS_IGNORE); /* previous push must have left the buffer */
    pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
    pdl_memcpy_d2h(&t->ctx, t->host_buf + 2, w->acc, P * sizeof(float));
    t->host_buf[0] = (float)w->since_push;
    t->host_buf[1] = (float)w->version;
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    MPI_Isend(t->host_buf, (int)(P + 2), MPI_FLOAT, 0, TAG_GRAD, t->comm, &w->push_req);
    t->comm_stats.bytes_sent += (double)(P + 2) * sizeof(float);
    t->comm_stats.messages++;
    t->comm_stats.calls++;
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    pdl_dev_memset0(&t->ctx, w->acc, P * sizeof(float));
    pdl_phase_end(&t->phases);
    w->since_push = 0;
}

static void poll_fetch(pdl_trainer *t, worker_state *w, int blocking) {
    if (w->fetch_req == MPI_REQUEST_NULL) return;
    int done = 0;
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    if (blocking) {
        MPI_Wait(&w->fetch_req, MPI_STATUS_IGNORE);
        done = 1;
    } else {
        MPI_Test(&w->fetch_req, &done, MPI_STATUS_IGNORE);
    }
    if (done) {
        pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
        pdl_memcpy_h2d(&t->ctx, t->model->params, w->fetch_buf + 1, t->model->num_params * sizeof(float));
        w->version = (long)w->fetch_buf[0];
    }
    pdl_phase_end(&t->phases);
}

static void request_fetch(pdl_trainer *t, worker_state *w) {
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    MPI_Send(NULL, 0, MPI_FLOAT, 0, TAG_FETCH, t->comm);
    MPI_Irecv(w->fetch_buf, (int)(t->model->num_params + 1), MPI_FLOAT, 0, TAG_PARAMS, t->comm, &w->fetch_req);
    t->comm_stats.messages++;
    pdl_phase_end(&t->phases);
}

static void run_worker(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const size_t P = t->model->num_params;
    worker_state w;
    memset(&w, 0, sizeof(w));
    w.acc = (float *)pdl_dev_malloc(&t->ctx, P * sizeof(float));
    w.fetch_buf = (float *)pdl_host_malloc(&t->ctx, (P + 1) * sizeof(float));
    w.push_req = MPI_REQUEST_NULL;
    w.fetch_req = MPI_REQUEST_NULL;
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)(t->n_local > 0 ? t->n_local : 1) * sizeof(uint32_t));
    long step = 0;

    for (int epoch = 1; epoch <= c->epochs; ++epoch) {
        pdl_trainer_begin_epoch(t, epoch);
        pdl_trainer_epoch_order(t, epoch, order);
        double loss_sum = 0.0;
        long correct = 0, samples = 0;
        const double t0 = pdl_now();
        for (int s = 0; s < t->steps_per_epoch; ++s, ++step) {
            if (c->fetch_every > 0 && step % c->fetch_every == 0 && w.fetch_req == MPI_REQUEST_NULL)
                request_fetch(t, &w);
            poll_fetch(t, &w, 0);
            int hits = 0;
            double loss = pdl_trainer_compute_gradients(t, pdl_trainer_batch(t, order, s), t->local_batch, 1.0f,
                                                        &hits, NULL, NULL);
            pdl_trainer_apply_update(t); /* local step (Algorithm 4, line 10) */
            pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
            pdl_axpy(&t->ctx, P, 1.0f, t->model->grads, w.acc);
            pdl_phase_end(&t->phases);
            if (++w.since_push == c->push_every) push_gradients(t, &w);
            loss_sum += loss;
            correct += hits;
            samples += t->local_batch;
            if (t->worker_index == 0) pdl_trainer_log_step(t, epoch, loss / t->local_batch, (double)hits / t->local_batch);
            pdl_trainer_check_finite(t, loss);
        }
        if (w.since_push > 0) push_gradients(t, &w);
        const double epoch_time = pdl_now() - t0;
        t->train_time += epoch_time;
        pdl_trainer_end_epoch(t, epoch, loss_sum, correct, samples, t->steps_per_epoch, epoch_time);
    }
    pdl_phase_timer_reset(&t->phases); /* account the shutdown separately, added below */
    poll_fetch(t, &w, 1);
    pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
    MPI_Wait(&w.push_req, MPI_STATUS_IGNORE);
    MPI_Send(NULL, 0, MPI_FLOAT, 0, TAG_DONE, t->comm);
    pdl_phase_end(&t->phases);
    /* the last epoch record keeps the final flush time */
    pdl_phase_accumulate(&t->records[t->num_records - 1].phases, &t->phases);
    free(order);
    pdl_dev_free(&t->ctx, w.acc);
    pdl_host_free(&t->ctx, w.fetch_buf);
}

int pdl_run_downpour(pdl_trainer *t) {
    MPI_Barrier(t->comm);
    if (t->is_server)
        run_server(t);
    else
        run_worker(t);
    MPI_Barrier(t->comm);
    /* broadcast the server's final model so every rank (and --dump-params) sees it */
    const size_t P = t->model->num_params;
    float *h = (float *)pdl_xmalloc(P * sizeof(float));
    pdl_memcpy_d2h(&t->ctx, h, t->model->params, P * sizeof(float));
    pdl_bcast_floats(h, P, 0, t->comm);
    pdl_memcpy_h2d(&t->ctx, t->model->params, h, P * sizeof(float));
    free(h);
    return 0;
}
