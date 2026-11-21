/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * comm.h — MPI helpers and hand-written collective algorithms.
 *
 * Gradient synchronization is an allreduce(sum) over a float buffer. Besides
 * the library MPI_Allreduce, several textbook algorithms are implemented on
 * top of point-to-point messages so their cost can be compared with the
 * latency/bandwidth (alpha-beta) model used in the paper's analysis:
 *
 *   algo           messages / rank        bytes sent / rank        rounds
 *   ring           2(k-1)                 2(k-1)/k * n             2(k-1)
 *   rd (rec.dbl.)  log2 k                 n log2 k                 log2 k
 *   tree           <= 2 log2 k            <= 2 n log2 k            2 log2 k   ("BitTorrent aggregate" in paper)
 *   naive          root: 2(k-1)           root: (k-1) n            2(k-1)     (central driver / parameter server)
 *   reduce_bcast   MPI_Reduce + MPI_Bcast (library)
 *   mpi            MPI_Allreduce (library)
 */
#ifndef PDL_COMM_H
#define PDL_COMM_H

#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PDL_AR_MPI = 0,
    PDL_AR_RING,
    PDL_AR_RD,
    PDL_AR_TREE,
    PDL_AR_NAIVE,
    PDL_AR_REDUCE_BCAST,
    PDL_AR_COUNT
} pdl_allreduce_algo;

const char *pdl_allreduce_name(pdl_allreduce_algo a);
int pdl_allreduce_parse(const char *s); /* -1 on failure */

typedef struct {
    double bytes_sent; /* payload bytes sent by this rank */
    long messages;     /* point-to-point messages sent by this rank (estimated for library calls) */
    long calls;        /* number of collective invocations */
} pdl_comm_stats;

/* Initializes MPI (thread level FUNNELED) and installs the fatal-abort hook. */
void pdl_mpi_init(int *argc, char ***argv, int *rank, int *world);
void pdl_mpi_finalize(void);

/*
 * In-place allreduce (sum) of `n` floats with the chosen algorithm.
 * `scratch` must hold at least n floats (may be NULL for PDL_AR_MPI).
 * All algorithms yield bitwise-identical results on every rank.
 */
void pdl_allreduce_sum(float *buf, size_t n, pdl_allreduce_algo algo, MPI_Comm comm,
                       float *scratch, pdl_comm_stats *stats);

/* Broadcast of a float buffer in chunks (handles n > INT_MAX). */
void pdl_bcast_floats(float *buf, size_t n, int root, MPI_Comm comm);

/* Gathers a uint64 from all ranks; returns 1 if all are equal. */
int pdl_all_equal_u64(uint64_t v, MPI_Comm comm);

/* Collects the per-rank processor name; returns number of distinct hosts. */
int pdl_count_hosts(MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif /* PDL_COMM_H */
