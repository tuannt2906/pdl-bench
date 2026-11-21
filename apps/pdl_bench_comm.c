/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * pdl_bench_comm — MPI communication benchmarks.
 *
 *   pingpong   rank 0 <-> rank 1 one-way time → Hockney α (latency), β (1/bandwidth)
 *   barrier    synchronization cost
 *   allreduce  every algorithm in comm.c over a message-size sweep (gradient sync)
 *   bcast      parameter broadcast (ADMM / parameter-server patterns)
 *   alltoall   data shuffle (ShuffleSGD)
 * Each repetition is timed per rank, the maximum over ranks is the repetition
 * time (a collective is finished when the slowest rank is done).
 */
#include <math.h>
#include <string.h>

#include "pdl/backend.h"
#include "pdl/cli.h"
#include "pdl/comm.h"
#include "pdl/common.h"
#include "pdl/env.h"
#include "pdl/json.h"
#include "pdl/timer.h"

static const pdl_cli_opt k_opts[] = {
    {"algos", "all", "list of mpi, ring, rd, tree, naive, reduce_bcast"},
    {"sizes", "", "explicit message sizes in bytes (overrides the sweep)"},
    {"min-bytes", "4", "smallest message of the sweep"},
    {"max-bytes", "67108864", "largest message of the sweep"},
    {"factor", "4", "geometric step of the sweep"},
    {"reps", "20", "timed repetitions per point"},
    {"warmup", "3", "untimed repetitions per point"},
    {"pingpong", "1", "run the ping-pong benchmark (needs >= 2 ranks)"},
    {"bcast", "1", "run the MPI_Bcast sweep"},
    {"alltoall", "1", "run the MPI_Alltoall sweep"},
    {"device", "cpu", "cpu | cuda (stage buffers through host memory)"},
    {"gpu-id", "-1", "CUDA device (-1 = local rank modulo device count)"},
    {"verify", "1", "check allreduce results"},
    {"seed", "42", "random seed (unused, recorded)"},
    {"output", "", "result JSON path"},
    {"run-id", "", "run identifier"},
    {"tag", "", "free-form tag"},
};

typedef struct {
    double mean, std, min, max;
    int reps;
} stats;

/* Reduces per-rank rep times with MAX, then computes statistics over reps. */
static stats summarize(double *t, int reps, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, t, reps, MPI_DOUBLE, MPI_MAX, comm);
    stats s = {0, 0, 1e300, 0, reps};
    for (int i = 0; i < reps; ++i) {
        s.mean += t[i];
        s.min = PDL_MIN(s.min, t[i]);
        s.max = PDL_MAX(s.max, t[i]);
    }
    s.mean /= reps;
    for (int i = 0; i < reps; ++i) s.std += (t[i] - s.mean) * (t[i] - s.mean);
    s.std = reps > 1 ? sqrt(s.std / (reps - 1)) : 0.0;
    return s;
}

static void write_stats(pdl_json *j, stats s) {
    pdl_json_kv_num(j, "time_mean_s", s.mean);
    pdl_json_kv_num(j, "time_std_s", s.std);
    pdl_json_kv_num(j, "time_min_s", s.min);
    pdl_json_kv_num(j, "time_max_s", s.max);
    pdl_json_kv_int(j, "reps", s.reps);
}

static void fill_pattern(float *b, size_t n, int rank) {
    for (size_t i = 0; i < n; ++i) b[i] = (float)((rank + 1) * (1 + (int)(i % 7)));
}

static int check_pattern(const float *b, size_t n, int k) {
    const float s = (float)k * (k + 1) / 2.0f;
    for (size_t i = 0; i < n; ++i)
        if (b[i] != s * (float)(1 + (int)(i % 7))) return 0;
    return 1;
}

int main(int argc, char **argv) {
    int rank = 0, k = 1;
    pdl_mpi_init(&argc, &argv, &rank, &k);
    pdl_cli cli;
    if (pdl_cli_parse(&cli, k_opts, (int)PDL_ARRAY_LEN(k_opts), argc, argv, rank == 0) != 0) {
        pdl_mpi_finalize();
        return 2;
    }
    if (cli.help) {
        if (rank == 0) pdl_cli_print_help(&cli, argv[0], "MPI communication benchmarks (allreduce algorithms, ping-pong, bcast, alltoall).");
        pdl_mpi_finalize();
        return 0;
    }
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int hosts = pdl_count_hosts(comm);
    const int reps = (int)pdl_cli_int(&cli, "reps"), warmup = (int)pdl_cli_int(&cli, "warmup");
    const int verify = (int)pdl_cli_int(&cli, "verify");
    const char *output = pdl_cli_str(&cli, "output");
    PDL_CHECK(reps > 0, "--reps must be > 0");

    /* message sizes */
    long long sizes[128];
    int nsizes = 0;
    if (pdl_cli_str(&cli, "sizes")[0]) {
        nsizes = pdl_cli_int_list(&cli, "sizes", sizes, 128);
    } else {
        long long lo = pdl_cli_int(&cli, "min-bytes"), hi = pdl_cli_int(&cli, "max-bytes");
        double f = pdl_cli_dbl(&cli, "factor");
        PDL_CHECK(lo >= 4 && hi >= lo && f > 1.0, "invalid size sweep");
        for (double b = (double)lo; b <= (double)hi * 1.0000001 && nsizes < 128; b *= f) sizes[nsizes++] = (long long)b;
    }
    long long max_bytes = 4;
    for (int i = 0; i < nsizes; ++i) max_bytes = PDL_MAX(max_bytes, sizes[i]);
    const size_t max_n = (size_t)(max_bytes / 4) + 1;

    /* algorithms */
    int algos[PDL_AR_COUNT], nalgo = 0;
    char **an;
    int na = pdl_cli_list(&cli, "algos", &an);
    if (na == 1 && strcmp(an[0], "all") == 0) {
        for (int a = 0; a < PDL_AR_COUNT; ++a) algos[nalgo++] = a;
    } else {
        for (int i = 0; i < na; ++i) {
            int a = pdl_allreduce_parse(an[i]);
            PDL_CHECK(a >= 0, "unknown allreduce algorithm '%s'", an[i]);
            algos[nalgo++] = a;
        }
    }
    pdl_str_free_parts(an, na);
    free(an);

    /* device staging */
    pdl_ctx ctx;
    const int use_cuda = strcmp(pdl_cli_str(&cli, "device"), "cuda") == 0;
    int gpu = (int)pdl_cli_int(&cli, "gpu-id");
    if (use_cuda && gpu < 0) {
        MPI_Comm node;
        int lr = 0;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &lr);
        MPI_Comm_free(&node);
        gpu = lr; /* modulo applied by device-count check below */
#ifdef PDL_WITH_CUDA
        extern int pdl_cuda_device_count(void);
        int nd = pdl_cuda_device_count();
        gpu = nd > 0 ? lr % nd : 0;
#endif
    }
    pdl_ctx_init(&ctx, use_cuda ? PDL_DEV_CUDA : PDL_DEV_CPU,
                 use_cuda ? (pdl_has_cublas() ? PDL_IMPL_CUDA_CUBLAS : PDL_IMPL_CUDA_TILED) : PDL_IMPL_CPU_BLOCKED, 1,
                 gpu < 0 ? 0 : gpu);
    float *buf = (float *)pdl_host_malloc(&ctx, max_n * sizeof(float));
    float *scratch = (float *)pdl_xcalloc(max_n, sizeof(float));
    float *dbuf = use_cuda ? (float *)pdl_dev_malloc(&ctx, max_n * sizeof(float)) : NULL;
    double *t = (double *)pdl_xcalloc((size_t)reps, sizeof(double));

    pdl_json *j = NULL;
    if (rank == 0 && output[0]) {
        j = pdl_json_open(output);
        PDL_CHECK(j, "cannot write %s", output);
        pdl_json_begin_object(j, NULL);
        pdl_json_kv_str(j, "schema", "pdl-bench/comm/v1");
        pdl_json_kv_str(j, "version", PDL_VERSION_STRING);
        pdl_json_kv_str(j, "run_id", pdl_cli_str(&cli, "run-id"));
        pdl_json_kv_str(j, "tag", pdl_cli_str(&cli, "tag"));
        pdl_cli_write_json(&cli, j);
        pdl_env_write_json(j, &ctx, argc, argv, k, hosts);
        pdl_json_kv_int(j, "world_size", k);
        pdl_json_kv_str(j, "device", use_cuda ? "cuda" : "cpu");
    }
    if (rank == 0) PDL_INFO("pdl_bench_comm: %d ranks on %d host(s), %d sizes, %d algorithms", k, hosts, nsizes, nalgo);

    /* ---- ping-pong ---------------------------------------------------- */
    if (j) pdl_json_begin_array(j, "pingpong");
    if (pdl_cli_int(&cli, "pingpong") && k >= 2) {
        for (int si = 0; si < nsizes; ++si) {
            int n = (int)PDL_MAX(1, sizes[si] / 4);
            for (int r = -warmup; r < reps; ++r) {
                MPI_Barrier(comm);
                double t0 = pdl_now();
                if (rank == 0) {
                    MPI_Send(buf, n, MPI_FLOAT, 1, 1, comm);
                    MPI_Recv(buf, n, MPI_FLOAT, 1, 1, comm, MPI_STATUS_IGNORE);
                } else if (rank == 1) {
                    MPI_Recv(buf, n, MPI_FLOAT, 0, 1, comm, MPI_STATUS_IGNORE);
                    MPI_Send(buf, n, MPI_FLOAT, 0, 1, comm);
                }
                if (r >= 0) t[r] = rank == 0 ? (pdl_now() - t0) / 2.0 : 0.0;
            }
            stats s = summarize(t, reps, comm);
            if (j) {
                pdl_json_begin_object(j, NULL);
                pdl_json_kv_int(j, "bytes", (long long)n * 4);
                pdl_json_kv_num(j, "one_way_s", s.mean);
                pdl_json_kv_num(j, "one_way_min_s", s.min);
                pdl_json_kv_num(j, "bandwidth_Bps", s.mean > 0 ? n * 4.0 / s.mean : 0.0);
                pdl_json_kv_int(j, "reps", reps);
                pdl_json_end_object(j);
            }
            if (rank == 0) PDL_INFO("pingpong  %10lld B  one-way %10.2f us", (long long)n * 4, s.mean * 1e6);
        }
    }
    if (j) pdl_json_end_array(j);

    /* ---- barrier ------------------------------------------------------- */
    for (int r = -warmup; r < reps; ++r) {
        double t0 = pdl_now();
        MPI_Barrier(comm);
        if (r >= 0) t[r] = pdl_now() - t0;
    }
    stats sb = summarize(t, reps, comm);
    if (j) {
        pdl_json_begin_object(j, "barrier");
        pdl_json_kv_num(j, "mean_s", sb.mean);
        pdl_json_kv_int(j, "reps", reps);
        pdl_json_end_object(j);
    }

    /* ---- allreduce ----------------------------------------------------- */
    int all_ok = 1;
    if (j) pdl_json_begin_array(j, "allreduce");
    for (int ai = 0; ai < nalgo; ++ai) {
        const pdl_allreduce_algo algo = (pdl_allreduce_algo)algos[ai];
        for (int si = 0; si < nsizes; ++si) {
            const size_t n = (size_t)PDL_MAX(1, sizes[si] / 4);
            pdl_comm_stats cs;
            memset(&cs, 0, sizeof(cs));
            int ok = 1;
            for (int r = -warmup; r < reps; ++r) {
                fill_pattern(buf, n, rank);
                if (use_cuda) pdl_memcpy_h2d(&ctx, dbuf, buf, n * sizeof(float));
                MPI_Barrier(comm);
                double t0 = pdl_now();
                if (use_cuda) pdl_memcpy_d2h(&ctx, buf, dbuf, n * sizeof(float));
                pdl_allreduce_sum(buf, n, algo, comm, scratch, r >= 0 ? &cs : NULL);
                if (use_cuda) pdl_memcpy_h2d(&ctx, dbuf, buf, n * sizeof(float));
                if (r >= 0) t[r] = pdl_now() - t0;
                if (verify && r == reps - 1) ok = check_pattern(buf, n, k);
            }
            stats s = summarize(t, reps, comm);
            MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, comm);
            double per_rank[2] = {cs.bytes_sent / reps, (double)cs.messages / reps};
            MPI_Allreduce(MPI_IN_PLACE, per_rank, 2, MPI_DOUBLE, MPI_SUM, comm);
            all_ok &= ok;
            const double bytes = (double)n * 4.0;
            if (j) {
                pdl_json_begin_object(j, NULL);
                pdl_json_kv_str(j, "algo", pdl_allreduce_name(algo));
                pdl_json_kv_int(j, "bytes", (long long)bytes);
                pdl_json_kv_int(j, "count", (long long)n);
                write_stats(j, s);
                double algbw = s.mean > 0 ? bytes / s.mean : 0.0;
                pdl_json_kv_num(j, "algbw_Bps", algbw);
                pdl_json_kv_num(j, "busbw_Bps", k > 1 ? algbw * 2.0 * (k - 1) / k : algbw);
                pdl_json_kv_num(j, "bytes_sent_per_rank", per_rank[0] / k);
                pdl_json_kv_num(j, "messages_per_rank", per_rank[1] / k);
                if (verify)
                    pdl_json_kv_bool(j, "correct", ok);
                else
                    pdl_json_kv_null(j, "correct");
                pdl_json_end_object(j);
            }
            if (rank == 0)
                PDL_INFO("allreduce %-12s %10.0f B  %10.2f us  (min %.2f)%s", pdl_allreduce_name(algo), bytes,
                         s.mean * 1e6, s.min * 1e6, verify ? (ok ? "" : "  WRONG RESULT") : "");
        }
    }
    if (j) pdl_json_end_array(j);

    /* ---- bcast --------------------------------------------------------- */
    if (j) pdl_json_begin_array(j, "bcast");
    if (pdl_cli_int(&cli, "bcast") && k >= 2) {
        for (int si = 0; si < nsizes; ++si) {
            const int n = (int)PDL_MAX(1, sizes[si] / 4);
            for (int r = -warmup; r < reps; ++r) {
                MPI_Barrier(comm);
                double t0 = pdl_now();
                MPI_Bcast(buf, n, MPI_FLOAT, 0, comm);
                if (r >= 0) t[r] = pdl_now() - t0;
            }
            stats s = summarize(t, reps, comm);
            if (j) {
                pdl_json_begin_object(j, NULL);
                pdl_json_kv_int(j, "bytes", (long long)n * 4);
                write_stats(j, s);
                pdl_json_end_object(j);
            }
        }
    }
    if (j) pdl_json_end_array(j);

    /* ---- alltoall ------------------------------------------------------ */
    if (j) pdl_json_begin_array(j, "alltoall");
    if (pdl_cli_int(&cli, "alltoall") && k >= 2) {
        for (int si = 0; si < nsizes; ++si) {
            const int per = (int)(sizes[si] / 4 / k); /* floats per destination */
            if (per < 1) continue;
            for (int r = -warmup; r < reps; ++r) {
                MPI_Barrier(comm);
                double t0 = pdl_now();
                MPI_Alltoall(buf, per, MPI_FLOAT, scratch, per, MPI_FLOAT, comm);
                if (r >= 0) t[r] = pdl_now() - t0;
            }
            stats s = summarize(t, reps, comm);
            if (j) {
                pdl_json_begin_object(j, NULL);
                pdl_json_kv_int(j, "bytes", (long long)per * 4 * k);
                write_stats(j, s);
                pdl_json_end_object(j);
            }
        }
    }
    if (j) pdl_json_end_array(j);

    if (j) {
        pdl_json_kv_bool(j, "all_correct", all_ok);
        pdl_json_end_object(j);
        pdl_json_close(j);
        PDL_INFO("results written to %s", output);
    }
    free(t);
    free(scratch);
    pdl_host_free(&ctx, buf);
    pdl_dev_free(&ctx, dbuf);
    pdl_ctx_destroy(&ctx);
    pdl_cli_free(&cli);
    pdl_mpi_finalize();
    return all_ok ? 0 : 3;
}
