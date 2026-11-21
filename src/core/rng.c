/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * rng.c — xoshiro256** generator seeded with splitmix64.
 */
#include "pdl/rng.h"

#include <math.h>

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t pdl_hash64(uint64_t x) {
    uint64_t s = x;
    return splitmix64(&s);
}

void pdl_rng_seed(pdl_rng *r, uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = seed;
    x = pdl_hash64(x ^ (a * 0xD1B54A32D192ED03ULL));
    x = pdl_hash64(x ^ (b * 0xAEF17502108EF2D9ULL));
    x = pdl_hash64(x ^ (c * 0x94D049BB133111EBULL));
    for (int i = 0; i < 4; ++i) r->s[i] = splitmix64(&x);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t pdl_rng_next(pdl_rng *r) {
    uint64_t *s = r->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

double pdl_rng_uniform(pdl_rng *r) { return (double)(pdl_rng_next(r) >> 11) * 0x1.0p-53; }

float pdl_rng_uniformf(pdl_rng *r, float lo, float hi) {
    return lo + (hi - lo) * (float)pdl_rng_uniform(r);
}

float pdl_rng_normal(pdl_rng *r) {
    /* Box–Muller; one value per call keeps the stream simple and reproducible. */
    double u1 = pdl_rng_uniform(r);
    double u2 = pdl_rng_uniform(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

uint64_t pdl_rng_below(pdl_rng *r, uint64_t bound) {
    if (bound == 0) return 0;
    /* Lemire-style rejection to avoid modulo bias. */
    uint64_t threshold = (0 - bound) % bound;
    for (;;) {
        uint64_t x = pdl_rng_next(r);
        if (x >= threshold) return x % bound;
    }
}

void pdl_rng_shuffle_u32(pdl_rng *r, uint32_t *a, size_t n) {
    if (n < 2) return;
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)pdl_rng_below(r, (uint64_t)i + 1);
        uint32_t tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}
