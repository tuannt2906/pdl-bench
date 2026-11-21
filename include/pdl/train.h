/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * train.h — training configuration, shared trainer runtime and the
 * distributed optimization strategies (paper §4–5).
 *
 *   sequential : single-process mini-batch SGD (baseline)
 *   sync       : synchronous data-parallel SGD, gradients allreduced every step
 *   psgd       : Parallel SGD (Zinkevich et al.) — local SGD + model averaging,
 *                optional ShuffleSGD data re-distribution between rounds
 *   admm       : consensus ADMM with local SGD sub-problem solves
 *   downpour   : asynchronous Downpour SGD with a parameter server on rank 0
 */
#ifndef PDL_TRAIN_H
#define PDL_TRAIN_H

#include <stdio.h>

#include "pdl/backend.h"
#include "pdl/comm.h"
#include "pdl/dataset.h"
#include "pdl/json.h"
#include "pdl/model.h"
#include "pdl/timer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PDL_STRAT_SEQUENTIAL = 0,
    PDL_STRAT_SYNC,
    PDL_STRAT_PSGD,
    PDL_STRAT_ADMM,
    PDL_STRAT_DOWNPOUR,
    PDL_STRAT_COUNT
} pdl_strategy;

const char *pdl_strategy_name(pdl_strategy s);

typedef struct {
    /* strategy & backend */
    int strategy;      /* pdl_strategy */
    int device;        /* pdl_device */
    char impl[32];     /* "auto" | naive | blocked | omp | tiled | cublas */
    int threads;       /* OpenMP threads per rank (0 = OMP default) */
    int gpu_id;        /* -1 = local_rank % num_gpus */
    int timing_sync;   /* sync device at phase boundaries for exact phase times */
    /* data */
    char train_data[512];
    char test_data[512];
    int train_limit;
    int test_limit;
    int partition;     /* pdl_partition */
    /* model */
    char model[512];
    int width;
    /* optimization */
    int epochs;
    int batch_size;    /* per-rank (local) mini-batch */
    int global_batch;  /* if > 0: batch_size = global_batch / workers */
    double lr;
    double momentum;
    double weight_decay;
    double lr_decay;   /* multiply lr by this every lr_step epochs */
    int lr_step;
    int lr_scale;      /* sync: multiply lr by #workers (linear scaling rule) */
    int max_steps;     /* cap of steps per epoch (0 = full epoch) */
    int warmup_steps;  /* untimed steps before epoch 1 */
    long long seed;
    /* communication */
    int allreduce;     /* pdl_allreduce_algo */
    int fusion_layer;  /* 1 = one allreduce per layer, 0 = single fused buffer */
    int overlap;       /* overlap per-layer MPI_Iallreduce with backward */
    int measure_sync;  /* barrier before comm to separate wait time from comm time */
    /* parallel SGD */
    int avg_period;    /* local steps between averaging; 0 = once per epoch; -1 = one-shot at end */
    int reshuffle;     /* ShuffleSGD all-to-all data exchange after each epoch */
    /* ADMM */
    double admm_rho;
    int admm_local_steps;
    /* Downpour */
    int push_every;    /* n_push: local steps between gradient pushes */
    int fetch_every;   /* n_fetch: local steps between parameter fetches */
    double server_lr;  /* 0 → use lr */
    /* correctness */
    int check_sync;    /* verify replica consistency every N steps (0 = off) */
    int check_finite;  /* abort on NaN/Inf in loss/gradients */
    /* output */
    char output[512];  /* result JSON path ("" = none) */
    char run_id[128];
    char tag[128];
    int log_steps;     /* write <output>.steps.csv every N steps (0 = off) */
    int eval_every;    /* evaluate every N epochs (0 = only last) */
    char dump_params[512]; /* write final params (rank 0, raw float32) */
    int verbose;
} pdl_train_config;

void pdl_train_config_defaults(pdl_train_config *c);
/* Parses argv; prints help and exits on --help. Returns 0 on success. */
int pdl_train_config_parse(pdl_train_config *c, int argc, char **argv);
void pdl_train_config_print_help(FILE *f, const char *prog);
void pdl_train_config_write_json(const pdl_train_config *c, pdl_json *j);

/* ------------------------------------------------------------------------ */

typedef struct {
    int epoch;
    double loss_sum;    /* local summed training loss */
    long correct;       /* local correct predictions */
    long samples;       /* local samples processed */
    long steps;         /* local optimizer steps */
    double time_s;      /* epoch wall time (training only, local) */
    double elapsed_s;   /* cumulative training time at epoch end (local) */
    double test_loss;   /* -1 when not evaluated on this rank */
    double test_acc;
    pdl_phase_timer phases;
} pdl_epoch_record;

typedef struct pdl_trainer {
    pdl_train_config *cfg;
    int rank, world;
    int num_workers;        /* world, or world-1 for downpour */
    int worker_index;       /* rank among workers (-1 for the downpour server) */
    int is_server;          /* downpour rank 0 */
    int global_slicing;     /* 1: all ranks share one epoch order and take slices of a global batch */
    int hosts;              /* distinct hosts in the MPI job */
    MPI_Comm comm;
    pdl_ctx ctx;
    pdl_model *model;
    pdl_dataset *train;     /* local training data */
    pdl_dataset *test;
    uint32_t *idx;          /* sample indices owned by this rank (into train) */
    int n_local;
    int n_train_total;      /* training samples before partitioning */
    int local_batch;
    int steps_per_epoch;
    float *xbuf;            /* host batch staging */
    int32_t *ybuf;
    float *host_buf;        /* host staging for gradients/params (num_params) */
    float *host_scratch;    /* allreduce scratch (num_params) */
    float *opt_state;       /* device momentum buffer (num_params) */
    double lr;              /* current learning rate */
    pdl_phase_timer phases; /* current epoch */
    pdl_comm_stats comm_stats;
    pdl_epoch_record *records;
    int num_records;
    double train_time;      /* training time excluding evaluation */
    double wall_start;
    long sync_checks, sync_failures, nonfinite_events;
    /* downpour statistics (server) */
    long server_updates;
    double staleness_sum;
    long staleness_max;
    /* ADMM statistics */
    long admm_rounds;
    double admm_primal_residual; /* ||x_i - z|| aggregated over workers, last round */
    FILE *steps_log;
    long global_step;
    int argc;
    char **argv;
} pdl_trainer;

/* Lifecycle (collective over MPI_COMM_WORLD). */
void pdl_trainer_init(pdl_trainer *t, pdl_train_config *cfg, int argc, char **argv);
void pdl_trainer_finalize(pdl_trainer *t);
int pdl_trainer_run(pdl_trainer *t);

/* ---- helpers used by strategies -------------------------------------- */
/* Loads samples idx[0..count) into the model, runs forward + backward.
 * Gradients of the mean loss over the local batch, scaled by grad_scale_mul,
 * end in model->grads. Returns summed loss; *correct receives hits. */
double pdl_trainer_compute_gradients(pdl_trainer *t, const uint32_t *idx, int count,
                                     float grad_scale_mul, int *correct,
                                     pdl_grad_ready_fn cb, void *user);
/* Applies SGD (momentum / weight decay) with gradients in model->grads. */
void pdl_trainer_apply_update(pdl_trainer *t);
/* Allreduces a device buffer (staging through host when needed); divides by `divisor`. */
void pdl_trainer_allreduce_device(pdl_trainer *t, float *dev_buf, size_t n, float divisor);
/* Evaluates params `params_dev` (NULL = model params) on the test set.
 * If `collective`, the test set is split across ranks and results reduced. */
void pdl_trainer_evaluate(pdl_trainer *t, const float *params_dev, int collective,
                          double *loss, double *acc);
/* Shuffled sample order for an epoch (n_local indices into t->train). */
void pdl_trainer_epoch_order(pdl_trainer *t, int epoch, uint32_t *order);
/* Indices of this rank's mini-batch `step` within an epoch order. */
const uint32_t *pdl_trainer_batch(const pdl_trainer *t, const uint32_t *order, int step);
/* Recomputes steps_per_epoch (minimum over workers, so collective strategies stay in lock-step). */
void pdl_trainer_recompute_steps(pdl_trainer *t, MPI_Comm workers);
/* Untimed warm-up steps; parameters and optimizer state are restored afterwards. */
void pdl_trainer_warmup(pdl_trainer *t, int with_allreduce);
/* Epoch bookkeeping. */
void pdl_trainer_begin_epoch(pdl_trainer *t, int epoch);
pdl_epoch_record *pdl_trainer_end_epoch(pdl_trainer *t, int epoch, double loss_sum, long correct,
                                        long samples, long steps, double epoch_time);
int pdl_trainer_should_eval(const pdl_trainer *t, int epoch);
/* Prints one progress line (rank-local training stats) when verbose. */
void pdl_trainer_report_epoch(const pdl_trainer *t, const pdl_epoch_record *r);
void pdl_trainer_log_step(pdl_trainer *t, int epoch, double loss, double acc);
/* Verifies all replicas hold identical params (collective). */
void pdl_trainer_check_sync(pdl_trainer *t);
void pdl_trainer_check_finite(pdl_trainer *t, double loss);
void pdl_trainer_update_lr(pdl_trainer *t, int epoch);
/* Writes the result JSON on rank 0 (collective). */
void pdl_trainer_write_results(pdl_trainer *t);

/* ---- strategies (all collective over t->comm) ------------------------ */
int pdl_run_sequential(pdl_trainer *t);
int pdl_run_sync(pdl_trainer *t);
int pdl_run_psgd(pdl_trainer *t);
int pdl_run_admm(pdl_trainer *t);
int pdl_run_downpour(pdl_trainer *t);

#ifdef __cplusplus
}
#endif

#endif /* PDL_TRAIN_H */
