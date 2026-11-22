/*
 * pdl-bench — SPDX-License-Identifier: MIT
 * dataset.c — .pdlds loader, synthetic generator, partitioning across ranks.
 */
#include "pdl/dataset.h"

#include <math.h>
#include <string.h>

#include "pdl/common.h"

static const char k_magic[8] = {'P', 'D', 'L', 'D', 'S', '1', 0, 0};

pdl_dataset *pdl_dataset_alloc(int n, int c, int h, int w, int num_classes) {
    pdl_dataset *d = (pdl_dataset *)pdl_xcalloc(1, sizeof(pdl_dataset));
    d->n = n;
    d->c = c;
    d->h = h;
    d->w = w;
    d->num_classes = num_classes;
    d->sample_bytes = (size_t)c * h * w;
    d->pixels = (uint8_t *)pdl_xcalloc((size_t)(n > 0 ? n : 1), d->sample_bytes);
    d->labels = (uint8_t *)pdl_xcalloc((size_t)(n > 0 ? n : 1), 1);
    return d;
}

void pdl_dataset_free(pdl_dataset *d) {
    if (!d) return;
    free(d->pixels);
    free(d->labels);
    free(d->protos);
    free(d);
}

/* ------------------------------------------------------------------------ */
/* .pdlds files                                                              */
/* ------------------------------------------------------------------------ */

static pdl_dataset *load_file(const char *path, int limit) {
    FILE *f = fopen(path, "rb");
    if (!f)
        PDL_FATAL("cannot open dataset '%s' — run `python -m pdlbench data` (see README) or use "
                  "synthetic:N:C:H:W:K",
                  path);
    char magic[8];
    uint32_t hdr[6];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, k_magic, 8) != 0)
        PDL_FATAL("'%s' is not a .pdlds file (bad magic)", path);
    if (fread(hdr, sizeof(uint32_t), 6, f) != 6) PDL_FATAL("'%s': truncated header", path);
    int n = (int)hdr[0], c = (int)hdr[1], h = (int)hdr[2], w = (int)hdr[3], k = (int)hdr[4];
    PDL_CHECK(n > 0 && c > 0 && h > 0 && w > 0 && k > 1 && k <= 256, "'%s': invalid header", path);
    int keep = (limit > 0 && limit < n) ? limit : n;
    pdl_dataset *d = pdl_dataset_alloc(keep, c, h, w, k);
    if (fread(d->pixels, d->sample_bytes, (size_t)keep, f) != (size_t)keep)
        PDL_FATAL("'%s': truncated pixel data", path);
    /* labels follow ALL n images */
    long label_off = 32L + (long)((size_t)n * d->sample_bytes);
    if (fseek(f, label_off, SEEK_SET) != 0) PDL_FATAL("'%s': seek failed", path);
    if (fread(d->labels, 1, (size_t)keep, f) != (size_t)keep) PDL_FATAL("'%s': truncated labels", path);
    fclose(f);
    for (int i = 0; i < keep; ++i)
        PDL_CHECK(d->labels[i] < k, "'%s': label %d out of range at %d", path, d->labels[i], i);
    snprintf(d->source, sizeof(d->source), "%s", path);
    return d;
}

int pdl_dataset_save(const pdl_dataset *d, const char *path) {
    PDL_CHECK(d->pixels, "cannot save a non-materialized dataset");
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t hdr[6] = {(uint32_t)d->n, (uint32_t)d->c, (uint32_t)d->h, (uint32_t)d->w,
                       (uint32_t)d->num_classes, 0};
    int ok = fwrite(k_magic, 1, 8, f) == 8 && fwrite(hdr, sizeof(uint32_t), 6, f) == 6 &&
             fwrite(d->pixels, d->sample_bytes, (size_t)d->n, f) == (size_t)d->n &&
             fwrite(d->labels, 1, (size_t)d->n, f) == (size_t)d->n;
    fclose(f);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------------ */
/* Synthetic data                                                            */
/* ------------------------------------------------------------------------ */
/*
 * Each class has a smooth prototype image (sum of 3 random plane waves per
 * channel). A sample = prototype * contrast + brightness shift + noise, all
 * drawn from an RNG keyed by (seed, sample index): learnable but not trivial,
 * and any subset can be generated independently on any rank.
 */

static void build_protos(pdl_dataset *d) {
    size_t sb = d->sample_bytes;
    d->protos = (float *)pdl_xmalloc((size_t)d->num_classes * sb * sizeof(float));
    for (int k = 0; k < d->num_classes; ++k) {
        for (int c = 0; c < d->c; ++c) {
            pdl_rng r;
            pdl_rng_seed(&r, d->seed, 0x9807ULL, (uint64_t)k, (uint64_t)c);
            double fx[3], fy[3], ph[3];
            for (int t = 0; t < 3; ++t) {
                fx[t] = 6.283185307179586 * pdl_rng_uniformf(&r, -4.0f, 4.0f) / d->w;
                fy[t] = 6.283185307179586 * pdl_rng_uniformf(&r, -4.0f, 4.0f) / d->h;
                ph[t] = 6.283185307179586 * pdl_rng_uniform(&r);
            }
            float *p = d->protos + (size_t)k * sb + (size_t)c * d->h * d->w;
            for (int y = 0; y < d->h; ++y)
                for (int x = 0; x < d->w; ++x) {
                    double v = 0.0;
                    for (int t = 0; t < 3; ++t) v += sin(fx[t] * x + fy[t] * y + ph[t]);
                    p[y * d->w + x] = (float)(v / 3.0);
                }
        }
    }
}

static uint8_t synth_label(uint64_t seed, uint64_t split, uint32_t i, int k) {
    uint64_t key = (split << 32) | i;
    return (uint8_t)(pdl_hash64(seed ^ (0xC1A55ULL + key * 0x9E3779B97F4A7C15ULL)) % (uint64_t)k);
}

static void synth_sample(const pdl_dataset *d, uint32_t i, uint8_t label, uint8_t *out) {
    pdl_rng r;
    pdl_rng_seed(&r, d->seed, 0x5A3F1EULL, (uint64_t)i, d->split);
    const float *p = d->protos + (size_t)label * d->sample_bytes;
    float contrast = 50.0f + 40.0f * (float)pdl_rng_uniform(&r);
    float shift = 128.0f + pdl_rng_uniformf(&r, -20.0f, 20.0f);
    size_t j = 0;
    while (j < d->sample_bytes) {
        uint64_t bits = pdl_rng_next(&r);
        for (int q = 0; q < 4 && j < d->sample_bytes; ++q, ++j) {
            /* triangular noise in [-60, 60] from two 8-bit uniforms */
            float u = (float)((bits >> (16 * q)) & 0xFF) + (float)((bits >> (16 * q + 8)) & 0xFF);
            float v = shift + contrast * p[j] + (u - 255.0f) * (60.0f / 255.0f);
            out[j] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
    }
}

static pdl_dataset *load_synthetic(const char *desc, int limit, uint64_t seed, int split) {
    char *f[8];
    int nf = pdl_str_split(desc, ':', f, 8);
    PDL_CHECK(nf == 6, "synthetic descriptor must be synthetic:N:C:H:W:K (got '%s')", desc);
    int n = atoi(f[1]), c = atoi(f[2]), h = atoi(f[3]), w = atoi(f[4]), k = atoi(f[5]);
    pdl_str_free_parts(f, nf);
    PDL_CHECK(n > 0 && c > 0 && h > 0 && w > 0 && k > 1 && k <= 256, "invalid synthetic descriptor '%s'", desc);
    if (limit > 0 && limit < n) n = limit;
    pdl_dataset *d = (pdl_dataset *)pdl_xcalloc(1, sizeof(pdl_dataset));
    d->n = n;
    d->c = c;
    d->h = h;
    d->w = w;
    d->num_classes = k;
    d->sample_bytes = (size_t)c * h * w;
    d->synthetic = 1;
    d->seed = seed;
    d->split = (uint64_t)split;
    d->labels = (uint8_t *)pdl_xmalloc((size_t)n);
    for (int i = 0; i < n; ++i) d->labels[i] = synth_label(seed, d->split, (uint32_t)i, k);
    build_protos(d);
    snprintf(d->source, sizeof(d->source), "%s", desc);
    return d;
}

pdl_dataset *pdl_dataset_load_split(const char *path_or_desc, int limit, uint64_t seed, int split) {
    if (pdl_str_starts_with(path_or_desc, "synthetic:")) return load_synthetic(path_or_desc, limit, seed, split);
    return load_file(path_or_desc, limit);
}

pdl_dataset *pdl_dataset_load(const char *path_or_desc, int limit, uint64_t seed) {
    return pdl_dataset_load_split(path_or_desc, limit, seed, 0);
}

/* Pointer to the bytes of sample i; synthetic samples are generated into tmp. */
static const uint8_t *sample_ptr(const pdl_dataset *d, uint32_t i, uint8_t *tmp) {
    if (d->pixels) return d->pixels + (size_t)i * d->sample_bytes;
    synth_sample(d, i, d->labels[i], tmp);
    return tmp;
}

pdl_dataset *pdl_dataset_subset(const pdl_dataset *d, const uint32_t *idx, int count) {
    pdl_dataset *s = pdl_dataset_alloc(count, d->c, d->h, d->w, d->num_classes);
    uint8_t *tmp = d->pixels ? NULL : (uint8_t *)pdl_xmalloc(d->sample_bytes);
    for (int j = 0; j < count; ++j) {
        PDL_CHECK((int)idx[j] < d->n, "subset index %u out of range", idx[j]);
        const uint8_t *src = sample_ptr(d, idx[j], tmp);
        memcpy(s->pixels + (size_t)j * d->sample_bytes, src, d->sample_bytes);
        s->labels[j] = d->labels[idx[j]];
    }
    free(tmp);
    snprintf(s->source, sizeof(s->source), "%s", d->source);
    return s;
}

void pdl_dataset_gather(const pdl_dataset *d, const uint32_t *idx, int count, float *x, int32_t *y) {
    const size_t sb = d->sample_bytes;
    uint8_t *tmp = d->pixels ? NULL : (uint8_t *)pdl_xmalloc(sb);
    const float scale = 1.0f / 127.5f;
    for (int j = 0; j < count; ++j) {
        const uint8_t *src = sample_ptr(d, idx[j], tmp);
        float *dst = x + (size_t)j * sb;
        for (size_t p = 0; p < sb; ++p) dst[p] = (float)src[p] * scale - 1.0f;
        if (y) y[j] = (int32_t)d->labels[idx[j]];
    }
    free(tmp);
}

/* ------------------------------------------------------------------------ */
/* Partitioning                                                              */
/* ------------------------------------------------------------------------ */

const char *pdl_partition_name(pdl_partition p) {
    switch (p) {
    case PDL_PART_GLOBAL: return "global";
    case PDL_PART_IID: return "iid";
    case PDL_PART_SORTED: return "sorted";
    }
    return "?";
}

int pdl_partition_parse(const char *s) {
    if (strcmp(s, "global") == 0) return PDL_PART_GLOBAL;
    if (strcmp(s, "iid") == 0) return PDL_PART_IID;
    if (strcmp(s, "sorted") == 0 || strcmp(s, "noniid") == 0) return PDL_PART_SORTED;
    return -1;
}

int pdl_partition_indices(const pdl_dataset *d, pdl_partition p, int rank, int world, uint64_t seed,
                          uint32_t **out) {
    const int n = d->n;
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)n * sizeof(uint32_t));
    if (p == PDL_PART_SORTED) {
        /* stable counting sort by label: shards become (mostly) single-class */
        int counts[257] = {0};
        for (int i = 0; i < n; ++i) counts[d->labels[i] + 1]++;
        for (int k = 0; k < 256; ++k) counts[k + 1] += counts[k];
        for (int i = 0; i < n; ++i) order[counts[d->labels[i]]++] = (uint32_t)i;
    } else {
        for (int i = 0; i < n; ++i) order[i] = (uint32_t)i;
        if (p == PDL_PART_IID) {
            pdl_rng r;
            pdl_rng_seed(&r, seed, 0x9A27ULL, 0, 0);
            pdl_rng_shuffle_u32(&r, order, (size_t)n);
        }
    }
    if (p == PDL_PART_GLOBAL || world <= 1) {
        *out = order;
        return n;
    }
    long lo = (long)n * rank / world, hi = (long)n * (rank + 1) / world;
    int cnt = (int)(hi - lo);
    uint32_t *mine = (uint32_t *)pdl_xmalloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(uint32_t));
    memcpy(mine, order + lo, (size_t)cnt * sizeof(uint32_t));
    free(order);
    *out = mine;
    return cnt;
}
