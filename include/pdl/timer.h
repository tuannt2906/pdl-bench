/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * timer.h — monotonic wall clock and per-phase timing accumulators.
 *
 * A training step is decomposed into phases (see pdl_phase). Each rank keeps a
 * pdl_phase_timer; the trainer gathers them on rank 0 to report the
 * computation / communication / synchronization breakdown.
 */
#ifndef PDL_TIMER_H
#define PDL_TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Seconds since an arbitrary fixed point (CLOCK_MONOTONIC). */
double pdl_now(void);

typedef enum {
    PDL_PHASE_DATA = 0,  /* host-side batch assembly (gather + normalize)          */
    PDL_PHASE_H2D,       /* input batch host->device copy                          */
    PDL_PHASE_FORWARD,   /* forward propagation + loss                             */
    PDL_PHASE_BACKWARD,  /* backward propagation (gradient computation)            */
    PDL_PHASE_TRANSFER,  /* gradient/param staging device<->host for MPI           */
    PDL_PHASE_SYNC_WAIT, /* time blocked in barrier before communication (imbalance) */
    PDL_PHASE_COMM,      /* MPI communication (allreduce / send / recv / alltoall) */
    PDL_PHASE_UPDATE,    /* optimizer / ADMM / server update                       */
    PDL_PHASE_EVAL,      /* evaluation on the test set                             */
    PDL_PHASE_SHUFFLE,   /* data re-distribution between ranks (ShuffleSGD)        */
    PDL_PHASE_COUNT
} pdl_phase;

const char *pdl_phase_name(pdl_phase p);

typedef struct {
    double total[PDL_PHASE_COUNT]; /* accumulated seconds per phase */
    long count[PDL_PHASE_COUNT];   /* number of intervals recorded  */
    double t_start;                /* start of the currently open interval */
    int open;                      /* currently open phase or -1 */
} pdl_phase_timer;

void pdl_phase_timer_reset(pdl_phase_timer *t);
/* Starts a phase interval (closing any open one). Device synchronization, when
 * required for accurate GPU timing, is the caller's job (see pdl_ctx_timing_sync). */
void pdl_phase_begin(pdl_phase_timer *t, pdl_phase p);
void pdl_phase_end(pdl_phase_timer *t);
void pdl_phase_add(pdl_phase_timer *t, pdl_phase p, double seconds);
double pdl_phase_sum(const pdl_phase_timer *t);
void pdl_phase_accumulate(pdl_phase_timer *dst, const pdl_phase_timer *src);

#ifdef __cplusplus
}
#endif

#endif /* PDL_TIMER_H */
