/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * seq_train.c — the SEQUENTIAL version: single-process, single-thread CNN
 * training in plain C (no MPI, no OpenMP, no CUDA).
 *
 * It uses only the shared CNN core (src/core, src/nn, src/data and the CPU
 * kernels in src/backend) and follows exactly the same sample order, weight
 * initialization and update rule as `pdl_train --strategy sequential`, so the
 * two programs print identical losses for identical options. It is the
 * baseline every parallel speedup is measured against.
 *
 *   make && ./seq_train --epochs 2 --output seq.json
 *   ./seq_train --help
 */
#define _DEFAULT_SOURCE
#include <math.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "pdl/backend.h"
#include "pdl/cli.h"
#include "pdl/common.h"
#include "pdl/dataset.h"
#include "pdl/json.h"
#include "pdl/model.h"
#include "pdl/rng.h"
#include "pdl/timer.h"

static const pdl_cli_opt k_opts[] = {
    {"train-data", "../data/mnist-train.pdlds", ".pdlds file or synthetic:N:C:H:W:K"},
    {"test-data", "../data/mnist-test.pdlds", ".pdlds file, synthetic:..., or none"},
    {"train-limit", "0", "use only the first N training samples (0 = all)"},
    {"test-limit", "0", "use only the first N test samples (0 = all)"},
    {"model", "lenet", "preset (mlp, lenet, cnn, cifar, paper) or spec string"},
    {"width", "1", "channel / hidden-unit multiplier for presets"},
    {"epochs", "5", "number of epochs"},
    {"batch-size", "64", "mini-batch size"},
    {"lr", "0.01", "learning rate"},
    {"momentum", "0.9", "SGD momentum"},
    {"weight-decay", "0", "L2 weight decay"},
    {"lr-decay", "1.0", "lr multiplier applied every lr-step epochs"},
    {"lr-step", "0", "epochs between lr decays (0 = never)"},
    {"max-steps", "0", "cap on steps per epoch (0 = full epoch)"},
    {"impl", "blocked", "GEMM variant: naive | blocked"},
    {"eval-every", "1", "evaluate every N epochs (0 = last only)"},
    {"seed", "42", "random seed"},
    {"output", "", "result JSON path"},
    {"run-id", "", "run identifier"},
    {"tag", "", "free-form tag"},
};

/* Phases measured by the sequential version (subset of the parallel ones). */
enum { PH_DATA, PH_FORWARD, PH_BACKWARD, PH_UPDATE, PH_EVAL, PH_N };

typedef struct {
    double train_loss, train_acc, test_loss, test_acc, time_s, elapsed_s;
    double ph[PH_N];
} epoch_rec;

static void evaluate(pdl_model *m, const pdl_dataset *test, float *xbuf, int32_t *ybuf, double *loss, double *acc) {
    double ls = 0.0, hits = 0.0;
    uint32_t *idx = (uint32_t *)pdl_xmalloc((size_t)m->max_batch * sizeof(uint32_t));
    for (int s = 0; s < test->n; s += m->max_batch) {
        int b = PDL_MIN(m->max_batch, test->n - s);
        for (int j = 0; j < b; ++j) idx[j] = (uint32_t)(s + j);
        pdl_dataset_gather(test, idx, b, xbuf, ybuf);
        pdl_model_set_batch(m, xbuf, ybuf, b);
        int h = 0;
        ls += pdl_model_eval_batch(m, b, &h);
        hits += h;
    }
    free(idx);
    *loss = ls / test->n;
    *acc = hits / test->n;
}

static void write_phases(pdl_json *j, const char *key, const double *ph) {
    /* same keys as the parallel version so the Python analysis reads both */
    pdl_json_begin_object(j, key);
    pdl_json_kv_num(j, "data", ph[PH_DATA]);
    pdl_json_kv_num(j, "h2d", 0.0);
    pdl_json_kv_num(j, "forward", ph[PH_FORWARD]);
    pdl_json_kv_num(j, "backward", ph[PH_BACKWARD]);
    pdl_json_kv_num(j, "transfer", 0.0);
    pdl_json_kv_num(j, "sync_wait", 0.0);
    pdl_json_kv_num(j, "comm", 0.0);
    pdl_json_kv_num(j, "update", ph[PH_UPDATE]);
    pdl_json_kv_num(j, "eval", ph[PH_EVAL]);
    pdl_json_kv_num(j, "shuffle", 0.0);
    pdl_json_end_object(j);
}

int main(int argc, char **argv) {
    pdl_cli cli;
    if (pdl_cli_parse(&cli, k_opts, (int)PDL_ARRAY_LEN(k_opts), argc, argv, 1) != 0) return 2;
    if (cli.help) {
        pdl_cli_print_help(&cli, argv[0], "Sequential CNN training in plain C (baseline for pdl-bench).");
        return 0;
    }
    const double wall0 = pdl_now();
    const uint64_t seed = (uint64_t)pdl_cli_int(&cli, "seed");
    const int epochs = (int)pdl_cli_int(&cli, "epochs"), B = (int)pdl_cli_int(&cli, "batch-size");
    const int eval_every = (int)pdl_cli_int(&cli, "eval-every"), lr_step = (int)pdl_cli_int(&cli, "lr-step");
    const int max_steps = (int)pdl_cli_int(&cli, "max-steps");
    double lr = pdl_cli_dbl(&cli, "lr");
    const double lr_decay = pdl_cli_dbl(&cli, "lr-decay");
    const float momentum = (float)pdl_cli_dbl(&cli, "momentum"), wd = (float)pdl_cli_dbl(&cli, "weight-decay");
    PDL_CHECK(epochs > 0 && B > 0 && lr > 0, "epochs, batch-size and lr must be positive");

    pdl_ctx ctx;
    int impl = pdl_impl_parse(pdl_cli_str(&cli, "impl"), PDL_DEV_CPU);
    PDL_CHECK(impl == PDL_IMPL_CPU_NAIVE || impl == PDL_IMPL_CPU_BLOCKED, "--impl must be naive or blocked");
    pdl_ctx_init(&ctx, PDL_DEV_CPU, (pdl_impl)impl, 1, 0);

    /* data: identical loading and ordering as pdl_train (partition "global", 1 rank) */
    pdl_dataset *lazy = pdl_dataset_load(pdl_cli_str(&cli, "train-data"), (int)pdl_cli_int(&cli, "train-limit"), seed);
    uint32_t *all = NULL;
    int n = pdl_partition_indices(lazy, PDL_PART_GLOBAL, 0, 1, seed, &all);
    pdl_dataset *train = pdl_dataset_subset(lazy, all, n);
    free(all);
    pdl_dataset_free(lazy);
    pdl_dataset *test = NULL;
    const char *test_path = pdl_cli_str(&cli, "test-data");
    if (strcmp(test_path, "none") != 0 && test_path[0]) {
        pdl_dataset *tl = pdl_dataset_load_split(test_path, (int)pdl_cli_int(&cli, "test-limit"), seed, 1);
        int nt = pdl_partition_indices(tl, PDL_PART_GLOBAL, 0, 1, 0, &all);
        test = pdl_dataset_subset(tl, all, nt);
        free(all);
        pdl_dataset_free(tl);
    }

    char *spec = pdl_model_preset_spec(pdl_cli_str(&cli, "model"), (int)pdl_cli_int(&cli, "width"), train->num_classes);
    pdl_model *m = pdl_model_create(&ctx, spec, train->c, train->h, train->w, train->num_classes, B);
    pdl_model_init_params(m, seed);
    pdl_model_print(m, stdout);
    const size_t P = m->num_params;
    float *velocity = (float *)pdl_xcalloc(P, sizeof(float));
    float *xbuf = (float *)pdl_xmalloc((size_t)B * train->sample_bytes * sizeof(float));
    int32_t *ybuf = (int32_t *)pdl_xmalloc((size_t)B * sizeof(int32_t));
    uint32_t *order = (uint32_t *)pdl_xmalloc((size_t)n * sizeof(uint32_t));

    int steps = n / B;
    if (max_steps > 0 && steps > max_steps) steps = max_steps;
    PDL_CHECK(steps > 0, "not enough samples (%d) for one batch of %d", n, B);
    printf("sequential: %d training samples, %d test samples, batch %d, %d steps/epoch\n", n, test ? test->n : 0, B,
           steps);

    epoch_rec *rec = (epoch_rec *)pdl_xcalloc((size_t)epochs, sizeof(epoch_rec));
    double train_time = 0.0, best = -1.0;
    for (int e = 1; e <= epochs; ++e) {
        if (lr_step > 0 && e > 1 && (e - 1) % lr_step == 0) lr *= lr_decay;
        /* same shuffle as pdl_trainer_epoch_order with global slicing */
        for (int i = 0; i < n; ++i) order[i] = (uint32_t)i;
        pdl_rng r;
        pdl_rng_seed(&r, seed, 0xE90CULL, (uint64_t)e, 0);
        pdl_rng_shuffle_u32(&r, order, (size_t)n);

        epoch_rec *R = &rec[e - 1];
        double loss_sum = 0.0;
        long hits_sum = 0;
        const double t0 = pdl_now();
        for (int s = 0; s < steps; ++s) {
            double t = pdl_now(), t1;
            pdl_dataset_gather(train, order + (size_t)s * B, B, xbuf, ybuf);
            pdl_model_set_batch(m, xbuf, ybuf, B);
            t1 = pdl_now(), R->ph[PH_DATA] += t1 - t, t = t1;
            int hits = 0;
            double loss = pdl_model_forward(m, B, 1.0f / (float)B, &hits);
            t1 = pdl_now(), R->ph[PH_FORWARD] += t1 - t, t = t1;
            pdl_model_backward(m, B, NULL, NULL);
            t1 = pdl_now(), R->ph[PH_BACKWARD] += t1 - t, t = t1;
            pdl_sgd_update(&ctx, P, (float)lr, momentum, wd, m->params, m->grads, velocity);
            R->ph[PH_UPDATE] += pdl_now() - t;
            PDL_CHECK(isfinite(loss), "non-finite loss at epoch %d step %d — lower --lr", e, s);
            loss_sum += loss;
            hits_sum += hits;
        }
        R->time_s = pdl_now() - t0;
        train_time += R->time_s;
        R->elapsed_s = train_time;
        R->train_loss = loss_sum / ((double)steps * B);
        R->train_acc = (double)hits_sum / ((double)steps * B);
        R->test_loss = R->test_acc = -1.0;
        if (test && (e == epochs || (eval_every > 0 && e % eval_every == 0))) {
            double t = pdl_now();
            evaluate(m, test, xbuf, ybuf, &R->test_loss, &R->test_acc);
            R->ph[PH_EVAL] = pdl_now() - t;
            best = PDL_MAX(best, R->test_acc);
        }
        printf("epoch %3d | loss %.4f acc %.4f | %6.2fs | %.0f samples/s", e, R->train_loss, R->train_acc, R->time_s,
               steps * B / R->time_s);
        if (R->test_acc >= 0) printf(" | test loss %.4f acc %.4f", R->test_loss, R->test_acc);
        printf("\n");
    }

    const char *output = pdl_cli_str(&cli, "output");
    if (output[0]) {
        pdl_json *j = pdl_json_open(output);
        PDL_CHECK(j, "cannot write %s", output);
        pdl_json_begin_object(j, NULL);
        pdl_json_kv_str(j, "schema", "pdl-bench/train/v1");
        pdl_json_kv_str(j, "version", PDL_VERSION_STRING);
        pdl_json_kv_str(j, "run_id", pdl_cli_str(&cli, "run-id"));
        pdl_json_kv_str(j, "tag", pdl_cli_str(&cli, "tag"));
        pdl_cli_write_json(&cli, j);
        char buf[512];
        pdl_json_begin_object(j, "environment");
        time_t now = time(NULL);
        struct tm tmv;
        gmtime_r(&now, &tmv);
        strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
        pdl_json_kv_str(j, "timestamp", buf);
        if (gethostname(buf, sizeof(buf)) == 0) pdl_json_kv_str(j, "hostname", buf);
        struct utsname u;
        if (uname(&u) == 0) {
            snprintf(buf, sizeof(buf), "%s %s %s", u.sysname, u.release, u.machine);
            pdl_json_kv_str(j, "os", buf);
        }
        pdl_json_kv_str(j, "build", "sequential (no MPI, no OpenMP, no CUDA)");
        pdl_json_kv_str(j, "device", "cpu");
        pdl_json_kv_str(j, "impl", pdl_impl_name(ctx.impl));
        pdl_json_kv_int(j, "threads", 1);
        pdl_json_end_object(j);
        pdl_json_kv_str(j, "strategy", "sequential-c");
        pdl_json_kv_int(j, "world_size", 1);
        pdl_json_kv_int(j, "num_workers", 1);
        pdl_json_begin_object(j, "model");
        pdl_json_kv_str(j, "spec", m->spec);
        pdl_json_kv_int(j, "num_params", (long long)P);
        pdl_json_kv_int(j, "param_bytes", (long long)(P * sizeof(float)));
        pdl_json_kv_num(j, "flops_per_sample", pdl_model_flops_per_sample(m));
        pdl_json_end_object(j);
        pdl_json_kv_int(j, "local_batch", B);
        pdl_json_kv_int(j, "global_batch", B);
        pdl_json_kv_int(j, "steps_per_epoch", steps);
        double tot[PH_N] = {0};
        pdl_json_begin_array(j, "epochs");
        for (int e = 0; e < epochs; ++e) {
            pdl_json_begin_object(j, NULL);
            pdl_json_kv_int(j, "epoch", e + 1);
            pdl_json_kv_num(j, "train_loss", rec[e].train_loss);
            pdl_json_kv_num(j, "train_acc", rec[e].train_acc);
            if (rec[e].test_acc >= 0) {
                pdl_json_kv_num(j, "test_loss", rec[e].test_loss);
                pdl_json_kv_num(j, "test_acc", rec[e].test_acc);
            } else {
                pdl_json_kv_null(j, "test_loss");
                pdl_json_kv_null(j, "test_acc");
            }
            pdl_json_kv_num(j, "time_s", rec[e].time_s);
            pdl_json_kv_num(j, "elapsed_s", rec[e].elapsed_s);
            pdl_json_kv_int(j, "samples", (long long)steps * B);
            pdl_json_kv_int(j, "steps", steps);
            write_phases(j, "phases_mean", rec[e].ph);
            pdl_json_end_object(j);
            for (int p = 0; p < PH_N; ++p) tot[p] += rec[e].ph[p];
        }
        pdl_json_end_array(j);
        const epoch_rec *last = &rec[epochs - 1];
        pdl_json_begin_object(j, "summary");
        pdl_json_kv_num(j, "train_time_s", train_time);
        pdl_json_kv_num(j, "wall_time_s", pdl_now() - wall0);
        pdl_json_kv_int(j, "samples_processed", (long long)steps * B * epochs);
        pdl_json_kv_num(j, "throughput_samples_per_s", (double)steps * B * epochs / train_time);
        pdl_json_kv_num(j, "final_train_loss", last->train_loss);
        pdl_json_kv_num(j, "final_train_acc", last->train_acc);
        if (last->test_acc >= 0) {
            pdl_json_kv_num(j, "final_test_loss", last->test_loss);
            pdl_json_kv_num(j, "final_test_acc", last->test_acc);
            pdl_json_kv_num(j, "best_test_acc", best);
        }
        write_phases(j, "phases_mean", tot);
        pdl_json_kv_num(j, "compute_time_mean_s", tot[PH_DATA] + tot[PH_FORWARD] + tot[PH_BACKWARD] + tot[PH_UPDATE]);
        pdl_json_kv_num(j, "comm_time_mean_s", 0.0);
        pdl_json_kv_num(j, "comm_fraction", 0.0);
        pdl_json_kv_num(j, "load_imbalance", 1.0);
        pdl_json_end_object(j);
        pdl_json_end_object(j);
        pdl_json_close(j);
        printf("results written to %s\n", output);
    }
    free(rec), free(order), free(xbuf), free(ybuf), free(velocity), free(spec);
    pdl_model_free(m);
    pdl_dataset_free(train);
    pdl_dataset_free(test);
    pdl_ctx_destroy(&ctx);
    pdl_cli_free(&cli);
    return 0;
}
