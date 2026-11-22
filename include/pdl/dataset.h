/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * dataset.h — image classification datasets.
 *
 * On-disk format (.pdlds, produced by `python -m pdlbench data`):
 *   offset 0   char[8]  magic "PDLDS1\0\0"
 *   offset 8   uint32   n, c, h, w, num_classes, reserved   (little endian)
 *   offset 32  uint8    pixels[n][c][h][w]
 *   ...        uint8    labels[n]
 *
 * Synthetic datasets are generated on the fly from a descriptor
 *   "synthetic:N:C:H:W:K"  (e.g. synthetic:20000:1:224:224:40, the paper setup)
 * Sample i depends only on (seed, i), so any subset is reproducible on any rank.
 */
#ifndef PDL_DATASET_H
#define PDL_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include "pdl/rng.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n, c, h, w, num_classes;
    size_t sample_bytes; /* c*h*w */
    uint8_t *pixels;     /* [n][sample_bytes]; NULL for a not-yet-materialized synthetic set */
    uint8_t *labels;     /* [n] (always present) */
    char source[256];
    /* synthetic generator state (pixels are produced on demand from (seed, index)) */
    int synthetic;
    uint64_t seed;
    uint64_t split;      /* 0 = train, 1 = test: same classes, disjoint sample keys */
    float *protos;       /* [num_classes][c*h*w] class prototypes in [-1, 1] */
} pdl_dataset;

/* Loads a .pdlds file or a synthetic descriptor. `limit` > 0 keeps only the
 * first `limit` samples. Synthetic datasets only get labels here; pixels are
 * produced by pdl_dataset_subset / pdl_dataset_gather. */
pdl_dataset *pdl_dataset_load(const char *path_or_desc, int limit, uint64_t seed);
/* Same as pdl_dataset_load, for a given split of a synthetic dataset: every split
 * shares the class prototypes of `seed` but draws different samples, so a
 * synthetic test set measures generalization on the same task. Files ignore `split`. */
pdl_dataset *pdl_dataset_load_split(const char *path_or_desc, int limit, uint64_t seed, int split);
/* Creates an empty dataset with the same geometry. */
pdl_dataset *pdl_dataset_alloc(int n, int c, int h, int w, int num_classes);
/* Returns a new, fully materialized dataset with the samples listed in idx
 * (synthetic samples are generated here, so only owned shards cost memory). */
pdl_dataset *pdl_dataset_subset(const pdl_dataset *d, const uint32_t *idx, int count);
void pdl_dataset_free(pdl_dataset *d);
/* Saves to .pdlds (used by tests and tools). */
int pdl_dataset_save(const pdl_dataset *d, const char *path);

/* Gathers samples idx[0..count) into a float batch normalised to [-1, 1] and int labels. */
void pdl_dataset_gather(const pdl_dataset *d, const uint32_t *idx, int count, float *x,
                        int32_t *y);

/* ---- data partitioning across ranks -------------------------------------- */
typedef enum {
    PDL_PART_GLOBAL = 0, /* replicated data; every step uses a slice of a global batch */
    PDL_PART_IID,        /* disjoint shards of a seeded global permutation             */
    PDL_PART_SORTED,     /* disjoint shards of the label-sorted data (non-IID, paper §5.1.1) */
} pdl_partition;

const char *pdl_partition_name(pdl_partition p);
int pdl_partition_parse(const char *s);

/* Index list of the samples owned by `rank` among `world` ranks.
 * For PDL_PART_GLOBAL every rank owns all samples. Caller frees *idx. */
int pdl_partition_indices(const pdl_dataset *d, pdl_partition p, int rank, int world,
                          uint64_t seed, uint32_t **idx);

#ifdef __cplusplus
}
#endif

#endif /* PDL_DATASET_H */
