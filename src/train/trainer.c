/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * trainer.c — runtime shared by all strategies: setup, data partitioning,
 * gradient computation, communication staging, evaluation, consistency
 * checks, metrics aggregation and the result JSON.
 */
#include <limits.h>
#include <math.h>
#include <string.h>

#include "pdl/common.h"
#include "pdl/env.h"
#include "pdl/json.h"
#include "pdl/rng.h"
#include "pdl/train.h"

const char *pdl_strategy_name(pdl_strategy s) {
    static const char *names[PDL_STRAT_COUNT] = {"sequential", "sync", "psgd", "admm", "downpour"};
    return (s >= 0 && s < PDL_STRAT_COUNT) ? names[s] : "?";
}

/* ------------------------------------------------------------------------ */
/* Setup                                                                     */
/* ------------------------------------------------------------------------ */

static int local_rank(MPI_Comm comm) {
    MPI_Comm node;
    int r = 0;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &r);
    MPI_Comm_free(&node);
    return r;
}

void pdl_trainer_recompute_steps(pdl_trainer *t, MPI_Comm comm) {
    int steps;
    if (t->is_server) {
        steps = INT_MAX;
    } else if (t->global_slicing) {
        steps = t->n_local / (t->local_batch * t->num_workers);
    } else {
        steps = t->n_local / t->local_batch;
    }
    MPI_Allreduce(MPI_IN_PLACE, &steps, 1, MPI_INT, MPI_MIN, comm);
    if (t->cfg->max_steps > 0 && steps > t->cfg->max_steps) steps = t->cfg->max_steps;
    PDL_CHECK(steps > 0,
              "not enough training samples for one step (n_local=%d, local batch=%d, workers=%d)",
              t->n_local, t->local_batch, t->num_workers);
    t->steps_per_epoch = steps;
}

void pdl_trainer_init(pdl_trainer *t, pdl_train_config *cfg, int argc, char **argv) {
    memset(t, 0, sizeof(*t));
    t->cfg = cfg;
    t->argc = argc;
    t->argv = argv;
    t->comm = MPI_COMM_WORLD;
    t->wall_start = pdl_now();
    MPI_Comm_rank(t->comm, &t->rank);
    MPI_Comm_size(t->comm, &t->world);
    t->hosts = pdl_count_hosts(t->comm);
    pdl_log_set_level(cfg->verbose >= 2 ? PDL_LOG_DEBUG : (cfg->verbose == 0 ? PDL_LOG_WARN : PDL_LOG_INFO));

    const pdl_strategy strat = (pdl_strategy)cfg->strategy;
    if (strat == PDL_STRAT_SEQUENTIAL && t->world != 1)
        PDL_FATAL("strategy 'sequential' must run with a single process (got %d ranks)", t->world);
    if (strat == PDL_STRAT_DOWNPOUR && t->world < 2)
        PDL_FATAL("strategy 'downpour' needs >= 2 ranks (rank 0 is the parameter server)");
    t->is_server = (strat == PDL_STRAT_DOWNPOUR && t->rank == 0);
    t->num_workers = strat == PDL_STRAT_DOWNPOUR ? t->world - 1 : t->world;
    t->worker_index = strat == PDL_STRAT_DOWNPOUR ? t->rank - 1 : t->rank;

    /* backend */
    pdl_device dev = (pdl_device)cfg->device;
    int impl = pdl_impl_parse(cfg->impl, dev);
    PDL_CHECK(impl >= 0, "unknown --impl '%s' for device %s", cfg->impl, pdl_device_name(dev));
    int gpu = cfg->gpu_id;
    if (dev == PDL_DEV_CUDA && gpu < 0) {
        int lr = local_rank(t->comm);
        int ndev = 0;
#ifdef PDL_WITH_CUDA
        extern int pdl_cuda_device_count(void);
        ndev = pdl_cuda_device_count();
#endif
        gpu = ndev > 0 ? lr % ndev : 0;
    }
    pdl_ctx_init(&t->ctx, dev, (pdl_impl)impl, cfg->threads, gpu);
    t->ctx.timing_sync = cfg->timing_sync;

    /* data */
    pdl_dataset *full = pdl_dataset_load(cfg->train_data, cfg->train_limit, (uint64_t)cfg->seed);
    if (strcmp(cfg->test_data, "none") != 0 && cfg->test_data[0]) {
        pdl_dataset *tfull = pdl_dataset_load(cfg->test_data, cfg->test_limit, (uint64_t)cfg->seed + 7919);
        PDL_CHECK(tfull->c == full->c && tfull->h == full->h && tfull->w == full->w &&
                      tfull->num_classes == full->num_classes,
                  "train/test datasets have different shapes");
        uint32_t *all = NULL;
        int nt = pdl_partition_indices(tfull, PDL_PART_GLOBAL, 0, 1, 0, &all);
        t->test = pdl_dataset_subset(tfull, all, nt);
        free(all);
        pdl_dataset_free(tfull);
    }
    if (cfg->partition < 0)
        cfg->partition = (strat == PDL_STRAT_SEQUENTIAL || strat == PDL_STRAT_SYNC) ? PDL_PART_GLOBAL
                                                                                      : PDL_PART_IID;
    t->global_slicing = (cfg->partition == PDL_PART_GLOBAL) &&
                        (strat == PDL_STRAT_SEQUENTIAL || strat == PDL_STRAT_SYNC);
    if (t->is_server) {
        t->train = pdl_dataset_alloc(0, full->c, full->h, full->w, full->num_classes);
        t->n_local = 0;
    } else {
        uint32_t *mine = NULL;
        int cnt = pdl_partition_indices(full, (pdl_partition)cfg->partition, t->worker_index,
                                        t->num_workers, (uint64_t)cfg->seed, &mine);
        t->train = pdl_dataset_subset(full, mine, cnt);
        t->n_local = cnt;
        free(mine);
    }
    snprintf(t->train->source, sizeof(t->train->source), "%s", cfg->train_data);
    const int C = full->c, H = full->h, W = full->w, K = full->num_classes;
    const int n_train_total = full->n;
    t->n_train_total = n_train_total;
    pdl_dataset_free(full);
    t->train->n = t->n_local; /* (subset already sized) */

    /* batch geometry */
    if (cfg->global_batch > 0) {
        PDL_CHECK(cfg->global_batch >= t->num_workers, "global batch %d < workers %d", cfg->global_batch,
                  t->num_workers);
        if (cfg->global_batch % t->num_workers && t->rank == 0)
            PDL_WARN("global batch %d not divisible by %d workers; using %d per worker", cfg->global_batch,
                     t->num_workers, cfg->global_batch / t->num_workers);
        t->local_batch = cfg->global_batch / t->num_workers;
    } else {
        t->local_batch = cfg->batch_size;
    }
    pdl_trainer_recompute_steps(t, t->comm);

    /* model */
    char *spec = pdl_model_preset_spec(cfg->model, cfg->width, K);
    t->model = pdl_model_create(&t->ctx, spec, C, H, W, K, t->local_batch);
    free(spec);
    pdl_model_init_params(t->model, (uint64_t)cfg->seed);
    const size_t P = t->model->num_params;
    /* identical replicas from the start (init is deterministic, the broadcast makes it explicit) */
    float *tmp = (float *)pdl_xmalloc(P * sizeof(float));
    pdl_memcpy_d2h(&t->ctx, tmp, t->model->params, P * sizeof(float));
    pdl_bcast_floats(tmp, P, 0, t->comm);
    pdl_memcpy_h2d(&t->ctx, t->model->params, tmp, P * sizeof(float));
    free(tmp);

    /* buffers */
    size_t sample = (size_t)C * H * W;
    t->xbuf = (float *)pdl_host_malloc(&t->ctx, (size_t)t->local_batch * sample * sizeof(float));
    t->ybuf = (int32_t *)pdl_host_malloc(&t->ctx, (size_t)t->local_batch * sizeof(int32_t));
    t->host_buf = (float *)pdl_host_malloc(&t->ctx, (P + 2) * sizeof(float));
    t->host_scratch = (float *)pdl_xcalloc(P + 2, sizeof(float));
    t->opt_state = (float *)pdl_dev_malloc(&t->ctx, P * sizeof(float));
    t->lr = cfg->lr * ((cfg->lr_scale && strat == PDL_STRAT_SYNC) ? t->num_workers : 1);
    t->records = (pdl_epoch_record *)pdl_xcalloc((size_t)cfg->epochs, sizeof(pdl_epoch_record));
    pdl_phase_timer_reset(&t->phases);

    if (t->rank == 0 && cfg->log_steps > 0 && cfg->output[0]) {
        char path[600];
        snprintf(path, sizeof(path), "%s.steps.csv", cfg->output);
        t->steps_log = fopen(path, "w");
        if (t->steps_log) fprintf(t->steps_log, "global_step,epoch,loss,acc,elapsed_s\n");
    }

    if (t->rank == 0 && cfg->verbose) {
        pdl_env_print_banner(&t->ctx, t->world);
        PDL_INFO("strategy=%s workers=%d partition=%s local_batch=%d steps/epoch=%d train=%d test=%d",
                 pdl_strategy_name(strat), t->num_workers, pdl_partition_name((pdl_partition)cfg->partition),
                 t->local_batch, t->steps_per_epoch, n_train_total, t->test ? t->test->n : 0);
        pdl_model_print(t->model, stdout);
    }
}

void pdl_trainer_finalize(pdl_trainer *t) {
    if (t->steps_log) fclose(t->steps_log);
    pdl_host_free(&t->ctx, t->xbuf);
    pdl_host_free(&t->ctx, t->ybuf);
    pdl_host_free(&t->ctx, t->host_buf);
    free(t->host_scratch);
    pdl_dev_free(&t->ctx, t->opt_state);
    pdl_model_free(t->model);
    pdl_dataset_free(t->train);
    pdl_dataset_free(t->test);
    free(t->idx);
    free(t->records);
    pdl_ctx_destroy(&t->ctx);
}

/* ------------------------------------------------------------------------ */
/* Step helpers                                                              */
/* ------------------------------------------------------------------------ */

double pdl_trainer_compute_gradients(pdl_trainer *t, const uint32_t *idx, int count, float grad_scale_mul,
                                     int *correct, pdl_grad_ready_fn cb, void *user) {
    pdl_phase_begin(&t->phases, PDL_PHASE_DATA);
    pdl_dataset_gather(t->train, idx, count, t->xbuf, t->ybuf);
    pdl_phase_begin(&t->phases, PDL_PHASE_H2D);
    pdl_model_set_batch(t->model, t->xbuf, t->ybuf, count);
    pdl_ctx_timing_sync(&t->ctx);
    pdl_phase_begin(&t->phases, PDL_PHASE_FORWARD);
    double loss = pdl_model_forward(t->model, count, grad_scale_mul / (float)count, correct);
    pdl_phase_begin(&t->phases, PDL_PHASE_BACKWARD);
    pdl_model_backward(t->model, count, cb, user);
    pdl_ctx_timing_sync(&t->ctx);
    pdl_phase_end(&t->phases);
    return loss;
}

void pdl_trainer_apply_update(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    pdl_sgd_update(&t->ctx, t->model->num_params, (float)t->lr, (float)c->momentum, (float)c->weight_decay,
                   t->model->params, t->model->grads, t->opt_state);
    pdl_ctx_timing_sync(&t->ctx);
    pdl_phase_end(&t->phases);
}

void pdl_trainer_allreduce_device(pdl_trainer *t, float *dev_buf, size_t n, float divisor) {
    if (t->world > 1) {
        if (t->cfg->measure_sync) {
            pdl_phase_begin(&t->phases, PDL_PHASE_SYNC_WAIT);
            MPI_Barrier(t->comm);
        }
        float *hbuf = dev_buf;
        if (t->ctx.device == PDL_DEV_CUDA) {
            pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
            hbuf = t->host_buf;
            pdl_memcpy_d2h(&t->ctx, hbuf, dev_buf, n * sizeof(float));
        }
        pdl_phase_begin(&t->phases, PDL_PHASE_COMM);
        pdl_allreduce_sum(hbuf, n, (pdl_allreduce_algo)t->cfg->allreduce, t->comm, t->host_scratch,
                          &t->comm_stats);
        if (t->ctx.device == PDL_DEV_CUDA) {
            pdl_phase_begin(&t->phases, PDL_PHASE_TRANSFER);
            pdl_memcpy_h2d(&t->ctx, dev_buf, hbuf, n * sizeof(float));
        }
    }
    pdl_phase_begin(&t->phases, PDL_PHASE_UPDATE);
    if (divisor != 1.0f) pdl_scal(&t->ctx, n, 1.0f / divisor, dev_buf);
    pdl_ctx_timing_sync(&t->ctx);
    pdl_phase_end(&t->phases);
}

void pdl_trainer_evaluate(pdl_trainer *t, const float *params_dev, int collective, double *loss, double *acc) {
    *loss = -1.0;
    *acc = -1.0;
    if (!t->test) return;
    pdl_phase_begin(&t->phases, PDL_PHASE_EVAL);
    pdl_model *m = t->model;
    float *orig = m->params;
    if (params_dev) m->params = (float *)params_dev;
    int n = t->test->n, lo = 0, hi = n;
    if (collective && t->world > 1) {
        lo = (int)((long)n * t->rank / t->world);
        hi = (int)((long)n * (t->rank + 1) / t->world);
    }
    double sums[3] = {0.0, 0.0, 0.0}; /* loss, correct, count */
    uint32_t *idx = (uint32_t *)pdl_xmalloc((size_t)m->max_batch * sizeof(uint32_t));
    for (int s = lo; s < hi; s += m->max_batch) {
        int b = PDL_MIN(m->max_batch, hi - s);
        for (int j = 0; j < b; ++j) idx[j] = (uint32_t)(s + j);
        pdl_dataset_gather(t->test, idx, b, t->xbuf, t->ybuf);
        pdl_model_set_batch(m, t->xbuf, t->ybuf, b);
        int hits = 0;
        sums[0] += pdl_model_eval_batch(m, b, &hits);
        sums[1] += hits;
        sums[2] += b;
    }
    free(idx);
    m->params = orig;
    if (collective && t->world > 1) MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, t->comm);
    pdl_phase_end(&t->phases);
    if (sums[2] > 0) {
        *loss = sums[0] / sums[2];
        *acc = sums[1] / sums[2];
    }
}

void pdl_trainer_epoch_order(pdl_trainer *t, int epoch, uint32_t *order) {
    for (int i = 0; i < t->n_local; ++i) order[i] = (uint32_t)i;
    pdl_rng r;
    /* shared order for global slicing, rank-specific order otherwise */
    uint64_t stream = t->global_slicing ? 0 : (uint64_t)t->rank + 1;
    pdl_rng_seed(&r, (uint64_t)t->cfg->seed, 0xE90CULL, (uint64_t)epoch, stream);
    pdl_rng_shuffle_u32(&r, order, (size_t)t->n_local);
}

const uint32_t *pdl_trainer_batch(const pdl_trainer *t, const uint32_t *order, int step) {
    if (t->global_slicing)
        return order + (size_t)step * t->local_batch * t->num_workers + (size_t)t->worker_index * t->local_batch;
    return order + (size_t)step * t->local_batch;
}

void pdl_trainer_update_lr(pdl_trainer *t, int epoch) {
    const pdl_train_config *c = t->cfg;
    if (c->lr_step > 0 && epoch > 1 && (epoch - 1) % c->lr_step == 0) t->lr *= c->lr_decay;
}

void pdl_trainer_begin_epoch(pdl_trainer *t, int epoch) {
    pdl_trainer_update_lr(t, epoch);
    pdl_phase_timer_reset(&t->phases);
}

int pdl_trainer_should_eval(const pdl_trainer *t, int epoch) {
    const pdl_train_config *c = t->cfg;
    if (epoch == c->epochs) return 1;
    return c->eval_every > 0 && epoch % c->eval_every == 0;
}

pdl_epoch_record *pdl_trainer_end_epoch(pdl_trainer *t, int epoch, double loss_sum, long correct, long samples,
                                        long steps, double epoch_time) {
    PDL_CHECK(t->num_records < t->cfg->epochs, "too many epoch records");
    pdl_epoch_record *r = &t->records[t->num_records++];
    r->epoch = epoch;
    r->loss_sum = loss_sum;
    r->correct = correct;
    r->samples = samples;
    r->steps = steps;
    r->time_s = epoch_time;
    r->elapsed_s = t->train_time;
    r->phases = t->phases;
    r->test_loss = -1.0; /* filled in by the strategy after evaluation */
    r->test_acc = -1.0;
    return r;
}

void pdl_trainer_report_epoch(const pdl_trainer *t, const pdl_epoch_record *r) {
    if (!t->cfg->verbose) return;
    const double *ph = r->phases.total;
    double comm = ph[PDL_PHASE_COMM] + ph[PDL_PHASE_TRANSFER] + ph[PDL_PHASE_SYNC_WAIT] + ph[PDL_PHASE_SHUFFLE];
    char test[64] = "";
    if (r->test_acc >= 0.0) snprintf(test, sizeof(test), " | test loss %.4f acc %.4f", r->test_loss, r->test_acc);
    if (r->samples > 0)
        PDL_INFO("epoch %3d | loss %.4f acc %.4f | %6.2fs (comm %4.1f%%) | %.0f samples/s/rank%s", r->epoch,
                 r->loss_sum / (double)r->samples, (double)r->correct / (double)r->samples, r->time_s,
                 r->time_s > 0 ? 100.0 * comm / r->time_s : 0.0, r->time_s > 0 ? r->samples / r->time_s : 0.0, test);
    else
        PDL_INFO("epoch %3d | %6.2fs%s", r->epoch, r->time_s, test);
}

void pdl_trainer_log_step(pdl_trainer *t, int epoch, double loss, double acc) {
    t->global_step++;
    if (t->steps_log && t->cfg->log_steps > 0 && t->global_step % t->cfg->log_steps == 0)
        fprintf(t->steps_log, "%ld,%d,%.6f,%.6f,%.6f\n", t->global_step, epoch, loss, acc,
                pdl_now() - t->wall_start);
}

void pdl_trainer_check_sync(pdl_trainer *t) {
    uint64_t h = pdl_model_param_hash(t->model);
    int ok = pdl_all_equal_u64(h, t->comm);
    t->sync_checks++;
    if (!ok) {
        t->sync_failures++;
        if (t->rank == 0) PDL_WARN("model replicas diverged at step %ld (hash mismatch)", t->global_step);
    }
}

void pdl_trainer_check_finite(pdl_trainer *t, double loss) {
    int bad = !isfinite(loss);
    if (!bad && t->global_step % 100 == 0)
        bad = pdl_count_nonfinite(&t->ctx, t->model->num_params, t->model->grads) > 0;
    if (bad) {
        t->nonfinite_events++;
        if (t->cfg->check_finite)
            PDL_FATAL("non-finite loss/gradient at step %ld (loss=%g) — lower --lr", t->global_step, loss);
    }
}

void pdl_trainer_warmup(pdl_trainer *t, int with_allreduce) {
    const int W = t->cfg->warmup_steps;
    if (W <= 0 || t->is_server) return;
    const size_t P = t->model->num_params;
    float *saved = (float *)pdl_dev_malloc(&t->ctx, P * sizeof(float));
    pdl_copy(&t->ctx, P, t->model->params, saved);
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)t->n_local * sizeof(uint32_t));
    pdl_trainer_epoch_order(t, 0, order);
    for (int s = 0; s < W; ++s) {
        int hits;
        pdl_trainer_compute_gradients(t, pdl_trainer_batch(t, order, s % t->steps_per_epoch), t->local_batch,
                                      1.0f, &hits, NULL, NULL);
        if (with_allreduce)
            pdl_trainer_allreduce_device(t, t->model->grads, P, (float)t->num_workers);
        pdl_trainer_apply_update(t);
    }
    pdl_copy(&t->ctx, P, saved, t->model->params);
    pdl_dev_memset0(&t->ctx, t->opt_state, P * sizeof(float));
    pdl_ctx_sync(&t->ctx);
    pdl_dev_free(&t->ctx, saved);
    free(order);
    pdl_phase_timer_reset(&t->phases);
    memset(&t->comm_stats, 0, sizeof(t->comm_stats));
}

/* ------------------------------------------------------------------------ */
/* Run + results                                                             */
/* ------------------------------------------------------------------------ */

int pdl_trainer_run(pdl_trainer *t) {
    int rc = 0;
    switch ((pdl_strategy)t->cfg->strategy) {
    case PDL_STRAT_SEQUENTIAL: rc = pdl_run_sequential(t); break;
    case PDL_STRAT_SYNC: rc = pdl_run_sync(t); break;
    case PDL_STRAT_PSGD: rc = pdl_run_psgd(t); break;
    case PDL_STRAT_ADMM: rc = pdl_run_admm(t); break;
    case PDL_STRAT_DOWNPOUR: rc = pdl_run_downpour(t); break;
    default: PDL_FATAL("unknown strategy");
    }
    pdl_trainer_write_results(t);
    return rc;
}

/* Per-epoch values exchanged with rank 0: 8 scalars + phase totals. */
#define REC_SCALARS 8
#define REC_STRIDE (REC_SCALARS + PDL_PHASE_COUNT)
/* Per-rank totals: role, train_time, bytes, messages, calls, samples. */
#define RANK_SCALARS 6

static void write_phase_obj(pdl_json *j, const char *key, const double *v) {
    pdl_json_begin_object(j, key);
    for (int p = 0; p < PDL_PHASE_COUNT; ++p) pdl_json_kv_num(j, pdl_phase_name((pdl_phase)p), v[p]);
    pdl_json_end_object(j);
}

static double ph_compute(const double *ph) {
    return ph[PDL_PHASE_DATA] + ph[PDL_PHASE_H2D] + ph[PDL_PHASE_FORWARD] + ph[PDL_PHASE_BACKWARD] +
           ph[PDL_PHASE_UPDATE];
}
static double ph_comm(const double *ph) {
    return ph[PDL_PHASE_COMM] + ph[PDL_PHASE_TRANSFER] + ph[PDL_PHASE_SHUFFLE];
}

void pdl_trainer_write_results(pdl_trainer *t) {
    const pdl_train_config *c = t->cfg;
    const int E = c->epochs, k = t->world;
    PDL_CHECK(t->num_records == E, "rank %d recorded %d of %d epochs", t->rank, t->num_records, E);

    /* ---- gather ------------------------------------------------------ */
    double *mine = (double *)pdl_xcalloc((size_t)E * REC_STRIDE, sizeof(double));
    for (int e = 0; e < E; ++e) {
        const pdl_epoch_record *r = &t->records[e];
        double *o = mine + (size_t)e * REC_STRIDE;
        o[0] = r->loss_sum;
        o[1] = (double)r->correct;
        o[2] = (double)r->samples;
        o[3] = (double)r->steps;
        o[4] = r->time_s;
        o[5] = r->elapsed_s;
        o[6] = r->test_loss;
        o[7] = r->test_acc;
        for (int p = 0; p < PDL_PHASE_COUNT; ++p) o[REC_SCALARS + p] = r->phases.total[p];
    }
    long samples_total = 0;
    for (int e = 0; e < E; ++e) samples_total += t->records[e].samples;
    double rank_tot[RANK_SCALARS] = {(double)t->is_server, t->train_time, t->comm_stats.bytes_sent,
                                     (double)t->comm_stats.messages, (double)t->comm_stats.calls,
                                     (double)samples_total};
    double *all = NULL, *all_tot = NULL;
    if (t->rank == 0) {
        all = (double *)pdl_xmalloc((size_t)k * E * REC_STRIDE * sizeof(double));
        all_tot = (double *)pdl_xmalloc((size_t)k * RANK_SCALARS * sizeof(double));
    }
    MPI_Gather(mine, E * REC_STRIDE, MPI_DOUBLE, all, E * REC_STRIDE, MPI_DOUBLE, 0, t->comm);
    MPI_Gather(rank_tot, RANK_SCALARS, MPI_DOUBLE, all_tot, RANK_SCALARS, MPI_DOUBLE, 0, t->comm);
    free(mine);

    /* final parameters for offline comparison */
    if (t->rank == 0 && c->dump_params[0] && !t->is_server) {
        size_t P = t->model->num_params;
        float *h = (float *)pdl_xmalloc(P * sizeof(float));
        pdl_memcpy_d2h(&t->ctx, h, t->model->params, P * sizeof(float));
        FILE *f = fopen(c->dump_params, "wb");
        if (f) {
            fwrite(h, sizeof(float), P, f);
            fclose(f);
        } else {
            PDL_WARN("cannot write %s", c->dump_params);
        }
        free(h);
    }
    if (t->rank != 0) return;

    /* ---- aggregate on rank 0 ----------------------------------------- */
    const double wall = pdl_now() - t->wall_start;
    double ph_sum[PDL_PHASE_COUNT] = {0}, ph_max[PDL_PHASE_COUNT], ph_min[PDL_PHASE_COUNT];
    for (int p = 0; p < PDL_PHASE_COUNT; ++p) {
        ph_max[p] = 0.0;
        ph_min[p] = 1e300;
    }
    double server_ph[PDL_PHASE_COUNT] = {0};
    int nworkers = 0;
    double train_time_max = 0.0, train_time_sum = 0.0, comp_max = 0.0, comp_sum = 0.0;
    double comm_sum = 0.0, wait_sum = 0.0, bytes_sum = 0.0, calls_sum = 0.0;
    long samples_all = 0;
    for (int r = 0; r < k; ++r) {
        const double *tot = all_tot + (size_t)r * RANK_SCALARS;
        double ph[PDL_PHASE_COUNT] = {0};
        for (int e = 0; e < E; ++e)
            for (int p = 0; p < PDL_PHASE_COUNT; ++p) ph[p] += all[((size_t)r * E + e) * REC_STRIDE + REC_SCALARS + p];
        bytes_sum += tot[2];
        calls_sum += tot[4];
        if (tot[0] != 0.0) { /* server */
            memcpy(server_ph, ph, sizeof(ph));
            continue;
        }
        nworkers++;
        for (int p = 0; p < PDL_PHASE_COUNT; ++p) {
            ph_sum[p] += ph[p];
            if (ph[p] > ph_max[p]) ph_max[p] = ph[p];
            if (ph[p] < ph_min[p]) ph_min[p] = ph[p];
        }
        train_time_max = PDL_MAX(train_time_max, tot[1]);
        train_time_sum += tot[1];
        double cp = ph_compute(ph);
        comp_max = PDL_MAX(comp_max, cp);
        comp_sum += cp;
        comm_sum += ph_comm(ph);
        wait_sum += ph[PDL_PHASE_SYNC_WAIT];
        samples_all += (long)tot[5];
    }
    double ph_mean[PDL_PHASE_COUNT];
    for (int p = 0; p < PDL_PHASE_COUNT; ++p) ph_mean[p] = nworkers ? ph_sum[p] / nworkers : 0.0;
    const double tt_mean = nworkers ? train_time_sum / nworkers : 0.0;

    pdl_json *j = c->output[0] ? pdl_json_open(c->output) : NULL;
    if (!j) {
        free(all);
        free(all_tot);
        return;
    }
    pdl_json_begin_object(j, NULL);
    pdl_json_kv_str(j, "schema", "pdl-bench/train/v1");
    pdl_json_kv_str(j, "version", PDL_VERSION_STRING);
    pdl_json_kv_str(j, "run_id", c->run_id);
    pdl_json_kv_str(j, "tag", c->tag);
    pdl_train_config_write_json(c, j);
    pdl_env_write_json(j, &t->ctx, t->argc, t->argv, t->world, t->hosts);
    pdl_json_kv_str(j, "strategy", pdl_strategy_name((pdl_strategy)c->strategy));
    pdl_json_kv_int(j, "world_size", t->world);
    pdl_json_kv_int(j, "num_workers", t->num_workers);

    pdl_json_begin_object(j, "dataset");
    pdl_json_kv_str(j, "train", c->train_data);
    pdl_json_kv_str(j, "test", c->test_data);
    pdl_json_kv_int(j, "n_train", t->n_train_total);
    pdl_json_kv_int(j, "n_train_local_rank0", t->n_local);
    pdl_json_kv_int(j, "n_test", t->test ? t->test->n : 0);
    pdl_json_begin_array(j, "shape");
    pdl_json_int(j, t->model->in_c);
    pdl_json_int(j, t->model->in_h);
    pdl_json_int(j, t->model->in_w);
    pdl_json_end_array(j);
    pdl_json_kv_int(j, "classes", t->model->num_classes);
    pdl_json_kv_str(j, "partition", pdl_partition_name((pdl_partition)c->partition));
    pdl_json_end_object(j);

    pdl_json_begin_object(j, "model");
    pdl_json_kv_str(j, "spec", t->model->spec);
    pdl_json_kv_int(j, "num_params", (long long)t->model->num_params);
    pdl_json_kv_int(j, "param_bytes", (long long)(t->model->num_params * sizeof(float)));
    pdl_json_kv_num(j, "flops_per_sample", pdl_model_flops_per_sample(t->model));
    pdl_json_begin_array(j, "layers");
    for (int i = 0; i < t->model->num_layers; ++i) {
        const pdl_layer *L = &t->model->layers[i];
        pdl_json_begin_object(j, NULL);
        pdl_json_kv_str(j, "name", L->name);
        pdl_json_kv_str(j, "kind", pdl_layer_kind_name(L->kind));
        pdl_json_begin_array(j, "in");
        pdl_json_int(j, L->in_c);
        pdl_json_int(j, L->in_h);
        pdl_json_int(j, L->in_w);
        pdl_json_end_array(j);
        pdl_json_begin_array(j, "out");
        pdl_json_int(j, L->out_c);
        pdl_json_int(j, L->out_h);
        pdl_json_int(j, L->out_w);
        pdl_json_end_array(j);
        pdl_json_kv_int(j, "params", (long long)(L->w_count + L->b_count));
        pdl_json_kv_num(j, "flops", L->flops_fwd);
        pdl_json_end_object(j);
    }
    pdl_json_end_array(j);
    pdl_json_end_object(j);
    pdl_json_kv_int(j, "local_batch", t->local_batch);
    pdl_json_kv_int(j, "global_batch", (long long)t->local_batch * t->num_workers);
    pdl_json_kv_int(j, "steps_per_epoch", t->steps_per_epoch);

    /* ---- epochs ---- */
    double best_acc = -1.0, last_test_loss = -1.0, last_test_acc = -1.0, last_train_loss = 0, last_train_acc = 0;
    pdl_json_begin_array(j, "epochs");
    for (int e = 0; e < E; ++e) {
        double loss = 0, corr = 0, samp = 0, tmax = 0, elmax = 0, steps = 0, tl = -1, ta = -1;
        double pm[PDL_PHASE_COUNT] = {0}, px[PDL_PHASE_COUNT] = {0};
        int nw = 0;
        for (int r = 0; r < k; ++r) {
            const double *o = all + ((size_t)r * E + e) * REC_STRIDE;
            if (o[6] >= 0.0 && tl < 0.0) {
                tl = o[6];
                ta = o[7];
            }
            if (all_tot[(size_t)r * RANK_SCALARS] != 0.0) continue; /* server */
            nw++;
            loss += o[0];
            corr += o[1];
            samp += o[2];
            if (steps == 0) steps = o[3];
            tmax = PDL_MAX(tmax, o[4]);
            elmax = PDL_MAX(elmax, o[5]);
            for (int p = 0; p < PDL_PHASE_COUNT; ++p) {
                pm[p] += o[REC_SCALARS + p];
                px[p] = PDL_MAX(px[p], o[REC_SCALARS + p]);
            }
        }
        for (int p = 0; p < PDL_PHASE_COUNT; ++p) pm[p] = nw ? pm[p] / nw : 0.0;
        pdl_json_begin_object(j, NULL);
        pdl_json_kv_int(j, "epoch", e + 1);
        last_train_loss = samp > 0 ? loss / samp : 0.0;
        last_train_acc = samp > 0 ? corr / samp : 0.0;
        pdl_json_kv_num(j, "train_loss", last_train_loss);
        pdl_json_kv_num(j, "train_acc", last_train_acc);
        if (tl >= 0.0) {
            pdl_json_kv_num(j, "test_loss", tl);
            pdl_json_kv_num(j, "test_acc", ta);
            last_test_loss = tl;
            last_test_acc = ta;
            best_acc = PDL_MAX(best_acc, ta);
        } else {
            pdl_json_kv_null(j, "test_loss");
            pdl_json_kv_null(j, "test_acc");
        }
        pdl_json_kv_num(j, "time_s", tmax);
        pdl_json_kv_num(j, "elapsed_s", elmax);
        pdl_json_kv_int(j, "samples", (long long)samp);
        pdl_json_kv_int(j, "steps", (long long)steps);
        write_phase_obj(j, "phases_mean", pm);
        write_phase_obj(j, "phases_max", px);
        pdl_json_end_object(j);
    }
    pdl_json_end_array(j);

    /* ---- summary ---- */
    pdl_json_begin_object(j, "summary");
    pdl_json_kv_num(j, "train_time_s", train_time_max);
    pdl_json_kv_num(j, "wall_time_s", wall);
    pdl_json_kv_int(j, "samples_processed", samples_all);
    pdl_json_kv_num(j, "throughput_samples_per_s", train_time_max > 0 ? samples_all / train_time_max : 0.0);
    pdl_json_kv_num(j, "final_train_loss", last_train_loss);
    pdl_json_kv_num(j, "final_train_acc", last_train_acc);
    if (last_test_acc >= 0) {
        pdl_json_kv_num(j, "final_test_loss", last_test_loss);
        pdl_json_kv_num(j, "final_test_acc", last_test_acc);
        pdl_json_kv_num(j, "best_test_acc", best_acc);
    } else {
        pdl_json_kv_null(j, "final_test_loss");
        pdl_json_kv_null(j, "final_test_acc");
        pdl_json_kv_null(j, "best_test_acc");
    }
    write_phase_obj(j, "phases_mean", ph_mean);
    write_phase_obj(j, "phases_max", ph_max);
    for (int p = 0; p < PDL_PHASE_COUNT; ++p)
        if (ph_min[p] > 1e299) ph_min[p] = 0.0;
    write_phase_obj(j, "phases_min", ph_min);
    const double comp_mean = nworkers ? comp_sum / nworkers : 0.0;
    pdl_json_kv_num(j, "compute_time_mean_s", comp_mean);
    pdl_json_kv_num(j, "comm_time_mean_s", nworkers ? comm_sum / nworkers : 0.0);
    pdl_json_kv_num(j, "sync_wait_mean_s", nworkers ? wait_sum / nworkers : 0.0);
    pdl_json_kv_num(j, "comm_fraction", tt_mean > 0 ? (comm_sum + wait_sum) / nworkers / tt_mean : 0.0);
    pdl_json_kv_num(j, "load_imbalance", comp_mean > 0 ? comp_max / comp_mean : 1.0);
    pdl_json_kv_num(j, "comm_bytes_per_rank", bytes_sum / k);
    pdl_json_kv_num(j, "comm_calls_per_rank", calls_sum / k);
    pdl_json_kv_int(j, "sync_checks", t->sync_checks);
    pdl_json_kv_int(j, "sync_failures", t->sync_failures);
    pdl_json_kv_int(j, "nonfinite_events", t->nonfinite_events);
    if (c->strategy == PDL_STRAT_DOWNPOUR) {
        pdl_json_begin_object(j, "server");
        pdl_json_kv_int(j, "updates", t->server_updates);
        pdl_json_kv_num(j, "mean_staleness", t->server_updates ? t->staleness_sum / t->server_updates : 0.0);
        pdl_json_kv_int(j, "max_staleness", t->staleness_max);
        write_phase_obj(j, "phases", server_ph);
        pdl_json_end_object(j);
    } else {
        pdl_json_kv_null(j, "server");
    }
    if (c->strategy == PDL_STRAT_ADMM) {
        pdl_json_begin_object(j, "admm");
        pdl_json_kv_int(j, "rounds", t->admm_rounds);
        pdl_json_kv_num(j, "primal_residual", t->admm_primal_residual);
        pdl_json_end_object(j);
    }
    pdl_json_end_object(j);

    /* ---- per rank ---- */
    pdl_json_begin_array(j, "per_rank");
    for (int r = 0; r < k; ++r) {
        const double *tot = all_tot + (size_t)r * RANK_SCALARS;
        double ph[PDL_PHASE_COUNT] = {0};
        for (int e = 0; e < E; ++e)
            for (int p = 0; p < PDL_PHASE_COUNT; ++p) ph[p] += all[((size_t)r * E + e) * REC_STRIDE + REC_SCALARS + p];
        pdl_json_begin_object(j, NULL);
        pdl_json_kv_int(j, "rank", r);
        pdl_json_kv_str(j, "role", tot[0] != 0.0 ? "server" : "worker");
        pdl_json_kv_num(j, "train_time_s", tot[1]);
        pdl_json_kv_int(j, "samples", (long long)tot[5]);
        write_phase_obj(j, "phases", ph);
        pdl_json_kv_num(j, "comm_bytes", tot[2]);
        pdl_json_kv_int(j, "comm_messages", (long long)tot[3]);
        pdl_json_kv_int(j, "comm_calls", (long long)tot[4]);
        pdl_json_end_object(j);
    }
    pdl_json_end_array(j);
    pdl_json_end_object(j);
    pdl_json_close(j);
    free(all);
    free(all_tot);
    if (c->verbose) PDL_INFO("results written to %s", c->output);
}
