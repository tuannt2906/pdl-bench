/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * rng.h — deterministic, platform-independent random numbers.
 *
 * All randomness (weight init, shuffling, synthetic data) flows through this
 * module so that a (seed, rank, epoch) triple fully determines an experiment.
 * Generator: xoshiro256** seeded through splitmix64.
 */
#ifndef PDL_RNG_H
#define PDL_RNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t s[4];
} pdl_rng;

/* Derives an independent stream from a base seed and up to 3 stream ids. */
void pdl_rng_seed(pdl_rng *r, uint64_t seed, uint64_t stream_a, uint64_t stream_b, uint64_t stream_c);
uint64_t pdl_rng_next(pdl_rng *r);
double pdl_rng_uniform(pdl_rng *r);                 /* [0, 1)            */
float pdl_rng_uniformf(pdl_rng *r, float lo, float hi);
float pdl_rng_normal(pdl_rng *r);                   /* N(0, 1)           */
uint64_t pdl_rng_below(pdl_rng *r, uint64_t bound); /* [0, bound)        */
/* Fisher–Yates shuffle of an index array. */
void pdl_rng_shuffle_u32(pdl_rng *r, uint32_t *a, size_t n);
/* Stateless hash (splitmix64 finalizer); used for per-sample synthetic data. */
uint64_t pdl_hash64(uint64_t x);

#ifdef __cplusplus
}
#endif

#endif /* PDL_RNG_H */
