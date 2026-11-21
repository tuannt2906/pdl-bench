/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * comm.c — MPI setup and allreduce algorithms built from point-to-point messages.
 *
 * Every algorithm guarantees bitwise-identical results on all ranks (each
 * element's final value is computed once, or computed with the same operand
 * order everywhere, then copied), which the model-synchronization check relies on.
 */
#include "pdl/comm.h"

#include <limits.h>
#include <string.h>

#include "pdl/common.h"

#define TAG_AR 7001

static void mpi_abort_hook(int code) {
    int up = 0, down = 0;
    MPI_Initialized(&up);
    MPI_Finalized(&down);
    if (up && !down) MPI_Abort(MPI_COMM_WORLD, code);
}

void pdl_mpi_init(int *argc, char ***argv, int *rank, int *world) {
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, world);
    pdl_set_abort_hook(mpi_abort_hook);
    if (*world > 1) pdl_log_set_rank(*rank);
    if (provided < MPI_THREAD_FUNNELED && *rank == 0)
        PDL_WARN("MPI library does not provide MPI_THREAD_FUNNELED (OpenMP + MPI may be unsafe)");
}

void pdl_mpi_finalize(void) {
    int down = 0;
    MPI_Finalized(&down);
    if (!down) MPI_Finalize();
}

static const char *k_algo_names[PDL_AR_COUNT] = {"mpi", "ring", "rd", "tree", "naive", "reduce_bcast"};

const char *pdl_allreduce_name(pdl_allreduce_algo a) {
    return (a >= 0 && a < PDL_AR_COUNT) ? k_algo_names[a] : "?";
}

int pdl_allreduce_parse(const char *s) {
    for (int i = 0; i < PDL_AR_COUNT; ++i)
        if (strcmp(s, k_algo_names[i]) == 0) return i;
    if (strcmp(s, "recursive_doubling") == 0) return PDL_AR_RD;
    return -1;
}

static inline void add_into(float *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

static inline void stat_send(pdl_comm_stats *st, size_t floats) {
    if (!st) return;
    st->bytes_sent += (double)floats * sizeof(float);
    st->messages += 1;
}

/* ---- ring: reduce-scatter + allgather ------------------------------------ */
static void ar_ring(float *buf, size_t n, MPI_Comm comm, float *scratch, pdl_comm_stats *st) {
    int rank, k;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &k);
    const int right = (rank + 1) % k, left = (rank - 1 + k) % k;
#define OFF(i) ((size_t)(((unsigned long long)n * (unsigned long long)(i)) / (unsigned long long)k))
    for (int s = 0; s < k - 1; ++s) {
        int sc = ((rank - s) % k + k) % k;
        int rc = ((rank - s - 1) % k + k) % k;
        size_t so = OFF(sc), sl = OFF(sc + 1) - so;
        size_t ro = OFF(rc), rl = OFF(rc + 1) - ro;
        MPI_Sendrecv(buf + so, (int)sl, MPI_FLOAT, right, TAG_AR, scratch, (int)rl, MPI_FLOAT, left,
                     TAG_AR, comm, MPI_STATUS_IGNORE);
        stat_send(st, sl);
        add_into(buf + ro, scratch, rl);
    }
    for (int s = 0; s < k - 1; ++s) {
        int sc = ((rank + 1 - s) % k + k) % k;
        int rc = ((rank - s) % k + k) % k;
        size_t so = OFF(sc), sl = OFF(sc + 1) - so;
        size_t ro = OFF(rc), rl = OFF(rc + 1) - ro;
        MPI_Sendrecv(buf + so, (int)sl, MPI_FLOAT, right, TAG_AR, buf + ro, (int)rl, MPI_FLOAT, left,
                     TAG_AR, comm, MPI_STATUS_IGNORE);
        stat_send(st, sl);
    }
#undef OFF
}

/* ---- recursive doubling (with fold-in for non powers of two) -------------- */
static void ar_rd(float *buf, size_t n, MPI_Comm comm, float *scratch, pdl_comm_stats *st) {
    int rank, k;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &k);
    int p2 = 1;
    while (p2 * 2 <= k) p2 *= 2;
    const int rem = k - p2;
    int newrank;
    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* fold into rank+1, then sit out */
            MPI_Send(buf, (int)n, MPI_FLOAT, rank + 1, TAG_AR, comm);
            stat_send(st, n);
            newrank = -1;
        } else {
            MPI_Recv(scratch, (int)n, MPI_FLOAT, rank - 1, TAG_AR, comm, MPI_STATUS_IGNORE);
            /* keep operand order canonical: lower rank first */
            for (size_t i = 0; i < n; ++i) buf[i] = scratch[i] + buf[i];
            newrank = rank / 2;
        }
    } else {
        newrank = rank - rem;
    }
    if (newrank >= 0) {
        for (int mask = 1; mask < p2; mask <<= 1) {
            int pnew = newrank ^ mask;
            int partner = pnew < rem ? pnew * 2 + 1 : pnew + rem;
            MPI_Sendrecv(buf, (int)n, MPI_FLOAT, partner, TAG_AR, scratch, (int)n, MPI_FLOAT, partner,
                         TAG_AR, comm, MPI_STATUS_IGNORE);
            stat_send(st, n);
            /* IEEE addition is commutative, but fix the order anyway (lower newrank first). */
            if (newrank < pnew)
                for (size_t i = 0; i < n; ++i) buf[i] = buf[i] + scratch[i];
            else
                for (size_t i = 0; i < n; ++i) buf[i] = scratch[i] + buf[i];
        }
    }
    if (rank < 2 * rem) {
        if (rank % 2) {
            MPI_Send(buf, (int)n, MPI_FLOAT, rank - 1, TAG_AR, comm);
            stat_send(st, n);
        } else {
            MPI_Recv(buf, (int)n, MPI_FLOAT, rank + 1, TAG_AR, comm, MPI_STATUS_IGNORE);
        }
    }
}

/* ---- binomial tree reduce to 0 + binomial broadcast ("BitTorrent aggregate") */
static void ar_tree(float *buf, size_t n, MPI_Comm comm, float *scratch, pdl_comm_stats *st) {
    int rank, k;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &k);
    for (int mask = 1; mask < k; mask <<= 1) {
        if (rank & mask) {
            MPI_Send(buf, (int)n, MPI_FLOAT, rank - mask, TAG_AR, comm);
            stat_send(st, n);
            break;
        } else if (rank + mask < k) {
            MPI_Recv(scratch, (int)n, MPI_FLOAT, rank + mask, TAG_AR, comm, MPI_STATUS_IGNORE);
            add_into(buf, scratch, n);
        }
    }
    int top = 1;
    while (top < k) top <<= 1;
    for (int mask = top >> 1; mask > 0; mask >>= 1) {
        if ((rank & (2 * mask - 1)) == 0 && rank + mask < k) {
            MPI_Send(buf, (int)n, MPI_FLOAT, rank + mask, TAG_AR, comm);
            stat_send(st, n);
        } else if ((rank & (2 * mask - 1)) == mask) {
            MPI_Recv(buf, (int)n, MPI_FLOAT, rank - mask, TAG_AR, comm, MPI_STATUS_IGNORE);
        }
    }
}

/* ---- naive: central driver gathers, sums, sends back (parameter-server shape) */
static void ar_naive(float *buf, size_t n, MPI_Comm comm, float *scratch, pdl_comm_stats *st) {
    int rank, k;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &k);
    if (rank == 0) {
        for (int r = 1; r < k; ++r) {
            MPI_Recv(scratch, (int)n, MPI_FLOAT, r, TAG_AR, comm, MPI_STATUS_IGNORE);
            add_into(buf, scratch, n);
        }
        for (int r = 1; r < k; ++r) {
            MPI_Send(buf, (int)n, MPI_FLOAT, r, TAG_AR, comm);
            stat_send(st, n);
        }
    } else {
        MPI_Send(buf, (int)n, MPI_FLOAT, 0, TAG_AR, comm);
        stat_send(st, n);
        MPI_Recv(buf, (int)n, MPI_FLOAT, 0, TAG_AR, comm, MPI_STATUS_IGNORE);
    }
}

void pdl_allreduce_sum(float *buf, size_t n, pdl_allreduce_algo algo, MPI_Comm comm, float *scratch,
                       pdl_comm_stats *st) {
    int k;
    MPI_Comm_size(comm, &k);
    if (st) st->calls += 1;
    if (k == 1 || n == 0) return;
    PDL_CHECK(n <= (size_t)INT_MAX, "allreduce of %zu floats exceeds INT_MAX", n);
    PDL_CHECK(algo == PDL_AR_MPI || algo == PDL_AR_REDUCE_BCAST || scratch,
              "allreduce '%s' needs a scratch buffer", pdl_allreduce_name(algo));
    switch (algo) {
    case PDL_AR_MPI:
        MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, MPI_FLOAT, MPI_SUM, comm);
        if (st) { /* estimate: bandwidth-optimal library algorithm (ring-like) */
            st->bytes_sent += 2.0 * (k - 1) / k * (double)n * sizeof(float);
            st->messages += 2 * (k - 1);
        }
        break;
    case PDL_AR_REDUCE_BCAST: {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0)
            MPI_Reduce(MPI_IN_PLACE, buf, (int)n, MPI_FLOAT, MPI_SUM, 0, comm);
        else
            MPI_Reduce(buf, NULL, (int)n, MPI_FLOAT, MPI_SUM, 0, comm);
        MPI_Bcast(buf, (int)n, MPI_FLOAT, 0, comm);
        if (st) { /* estimate: binomial trees, average per rank */
            st->bytes_sent += 2.0 * (k - 1) / k * (double)n * sizeof(float);
            st->messages += 2;
        }
        break;
    }
    case PDL_AR_RING: ar_ring(buf, n, comm, scratch, st); break;
    case PDL_AR_RD: ar_rd(buf, n, comm, scratch, st); break;
    case PDL_AR_TREE: ar_tree(buf, n, comm, scratch, st); break;
    case PDL_AR_NAIVE: ar_naive(buf, n, comm, scratch, st); break;
    default: PDL_FATAL("unknown allreduce algorithm %d", (int)algo);
    }
}

void pdl_bcast_floats(float *buf, size_t n, int root, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 28;
    for (size_t off = 0; off < n; off += chunk) {
        size_t len = PDL_MIN(chunk, n - off);
        MPI_Bcast(buf + off, (int)len, MPI_FLOAT, root, comm);
    }
}

int pdl_all_equal_u64(uint64_t v, MPI_Comm comm) {
    uint64_t mn = v, mx = v;
    MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_UINT64_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_UINT64_T, MPI_MAX, comm);
    return mn == mx;
}

int pdl_count_hosts(MPI_Comm comm) {
    int rank, k, len = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &k);
    char name[MPI_MAX_PROCESSOR_NAME];
    memset(name, 0, sizeof(name));
    MPI_Get_processor_name(name, &len);
    char *all = NULL;
    if (rank == 0) all = (char *)pdl_xcalloc((size_t)k, MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, comm);
    int hosts = 0;
    if (rank == 0) {
        for (int i = 0; i < k; ++i) {
            int seen = 0;
            for (int j = 0; j < i && !seen; ++j)
                seen = strcmp(all + (size_t)i * MPI_MAX_PROCESSOR_NAME,
                              all + (size_t)j * MPI_MAX_PROCESSOR_NAME) == 0;
            hosts += !seen;
        }
        free(all);
    }
    MPI_Bcast(&hosts, 1, MPI_INT, 0, comm);
    return hosts;
}
