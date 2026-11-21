/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * timer.c — monotonic clock and phase accumulators.
 */
#define _POSIX_C_SOURCE 200809L
#include "pdl/timer.h"

#include <string.h>
#include <time.h>

double pdl_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *k_phase_names[PDL_PHASE_COUNT] = {
    "data", "h2d", "forward", "backward", "transfer", "sync_wait", "comm", "update", "eval", "shuffle",
};

const char *pdl_phase_name(pdl_phase p) {
    return (p >= 0 && p < PDL_PHASE_COUNT) ? k_phase_names[p] : "?";
}

void pdl_phase_timer_reset(pdl_phase_timer *t) {
    memset(t, 0, sizeof(*t));
    t->open = -1;
}

void pdl_phase_end(pdl_phase_timer *t) {
    if (t->open < 0) return;
    t->total[t->open] += pdl_now() - t->t_start;
    t->count[t->open] += 1;
    t->open = -1;
}

void pdl_phase_begin(pdl_phase_timer *t, pdl_phase p) {
    double now = pdl_now();
    if (t->open >= 0) {
        t->total[t->open] += now - t->t_start;
        t->count[t->open] += 1;
    }
    t->open = (int)p;
    t->t_start = now;
}

void pdl_phase_add(pdl_phase_timer *t, pdl_phase p, double seconds) {
    t->total[p] += seconds;
    t->count[p] += 1;
}

double pdl_phase_sum(const pdl_phase_timer *t) {
    double s = 0.0;
    for (int i = 0; i < PDL_PHASE_COUNT; ++i) s += t->total[i];
    return s;
}

void pdl_phase_accumulate(pdl_phase_timer *dst, const pdl_phase_timer *src) {
    for (int i = 0; i < PDL_PHASE_COUNT; ++i) {
        dst->total[i] += src->total[i];
        dst->count[i] += src->count[i];
    }
}
