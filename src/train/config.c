/*
 * pdl-bench — SPDX-License-Identifier: MIT
 *
 * config.c — table-driven command-line parsing for pdl_train.
 * The same table produces --help output and the "config" object of the
 * result JSON, so every option is automatically recorded for reproducibility.
 */
#include <stddef.h>
#include <string.h>

#include "pdl/common.h"
#include "pdl/json.h"
#include "pdl/train.h"

typedef enum { OPT_INT, OPT_LL, OPT_DBL, OPT_STR, OPT_ENUM } opt_type;

typedef struct {
    const char *name;
    opt_type type;
    size_t offset;
    size_t size;                /* OPT_STR buffer size */
    const char *const *choices; /* OPT_ENUM */
    int nchoices;
    int base;                   /* value stored = choice index + base */
    const char *help;
} opt_def;

static const char *const k_strategies[] = {"sequential", "sync", "psgd", "admm", "downpour"};
static const char *const k_devices[] = {"cpu", "cuda"};
static const char *const k_partitions[] = {"auto", "global", "iid", "sorted"};
static const char *const k_allreduce[] = {"mpi", "ring", "rd", "tree", "naive", "reduce_bcast"};
static const char *const k_fusion[] = {"flat", "layer"};

#define F(field) offsetof(pdl_train_config, field)
#define S(field) sizeof(((pdl_train_config *)0)->field)
#define INT(n, f, h) {n, OPT_INT, F(f), 0, NULL, 0, 0, h}
#define LL(n, f, h) {n, OPT_LL, F(f), 0, NULL, 0, 0, h}
#define DBL(n, f, h) {n, OPT_DBL, F(f), 0, NULL, 0, 0, h}
#define STR(n, f, h) {n, OPT_STR, F(f), S(f), NULL, 0, 0, h}
#define ENUM(n, f, ch, base, h) {n, OPT_ENUM, F(f), 0, ch, (int)PDL_ARRAY_LEN(ch), base, h}

static const opt_def k_opts[] = {
    ENUM("strategy", strategy, k_strategies, 0, "sequential | sync | psgd | admm | downpour"),
    ENUM("device", device, k_devices, 0, "cpu | cuda"),
    STR("impl", impl, "kernel variant: auto | naive | blocked | omp | tiled | cublas"),
    INT("threads", threads, "OpenMP threads per rank (0 = runtime default)"),
    INT("gpu-id", gpu_id, "CUDA device (-1 = local rank modulo device count)"),
    INT("timing-sync", timing_sync, "synchronize GPU at phase boundaries (exact phase times)"),
    STR("train-data", train_data, ".pdlds file or synthetic:N:C:H:W:K"),
    STR("test-data", test_data, ".pdlds file, synthetic:..., or none"),
    INT("train-limit", train_limit, "use only the first N training samples (0 = all)"),
    INT("test-limit", test_limit, "use only the first N test samples (0 = all)"),
    ENUM("partition", partition, k_partitions, -1, "auto | global | iid | sorted"),
    STR("model", model, "preset (mlp, lenet, cnn, cifar, paper) or spec string"),
    INT("width", width, "channel / hidden-unit multiplier for presets"),
    INT("epochs", epochs, "number of epochs"),
    INT("batch-size", batch_size, "per-rank mini-batch size"),
    INT("global-batch", global_batch, "if > 0: per-rank batch = global / workers"),
    DBL("lr", lr, "learning rate"),
    DBL("momentum", momentum, "SGD momentum"),
    DBL("weight-decay", weight_decay, "L2 weight decay"),
    DBL("lr-decay", lr_decay, "lr multiplier applied every lr-step epochs"),
    INT("lr-step", lr_step, "epochs between lr decays (0 = never)"),
    INT("lr-scale", lr_scale, "sync: multiply lr by number of workers"),
    INT("max-steps", max_steps, "cap on steps per epoch (0 = full epoch)"),
    INT("warmup-steps", warmup_steps, "untimed warm-up steps (state restored afterwards)"),
    LL("seed", seed, "global random seed"),
    ENUM("allreduce", allreduce, k_allreduce, 0, "mpi | ring | rd | tree | naive | reduce_bcast"),
    ENUM("fusion", fusion_layer, k_fusion, 0, "flat (one fused allreduce) | layer (per layer)"),
    INT("overlap", overlap, "overlap per-layer MPI_Iallreduce with backward (allreduce=mpi)"),
    INT("measure-sync", measure_sync, "barrier before comm to time load-imbalance waiting"),
    INT("avg-period", avg_period, "psgd: steps between averaging (0 = per epoch, -1 = one-shot)"),
    INT("reshuffle", reshuffle, "psgd: ShuffleSGD all-to-all data exchange every epoch"),
    DBL("admm-rho", admm_rho, "ADMM penalty rho"),
    INT("admm-local-steps", admm_local_steps, "ADMM local SGD steps per round"),
    INT("push-every", push_every, "downpour: steps between gradient pushes"),
    INT("fetch-every", fetch_every, "downpour: steps between parameter fetches"),
    DBL("server-lr", server_lr, "downpour server learning rate (0 = lr)"),
    INT("check-sync", check_sync, "verify replicas identical every N steps (0 = off)"),
    INT("check-finite", check_finite, "abort on NaN/Inf loss or gradients"),
    INT("eval-every", eval_every, "evaluate every N epochs (0 = last only)"),
    STR("output", output, "result JSON path"),
    INT("log-steps", log_steps, "write <output>.steps.csv every N steps"),
    STR("dump-params", dump_params, "write final parameters (float32) to this file"),
    STR("run-id", run_id, "run identifier copied into the result"),
    STR("tag", tag, "free-form tag copied into the result"),
    INT("verbose", verbose, "0 quiet, 1 per epoch, 2 debug"),
};

void pdl_train_config_defaults(pdl_train_config *c) {
    memset(c, 0, sizeof(*c));
    c->strategy = PDL_STRAT_SYNC;
    c->device = PDL_DEV_CPU;
    snprintf(c->impl, sizeof(c->impl), "auto");
    c->gpu_id = -1;
    c->timing_sync = 1;
    snprintf(c->train_data, sizeof(c->train_data), "data/mnist-train.pdlds");
    snprintf(c->test_data, sizeof(c->test_data), "data/mnist-test.pdlds");
    c->partition = -1;
    snprintf(c->model, sizeof(c->model), "lenet");
    c->width = 1;
    c->epochs = 5;
    c->batch_size = 64;
    c->lr = 0.01;
    c->momentum = 0.9;
    c->lr_decay = 1.0;
    c->seed = 42;
    c->allreduce = PDL_AR_MPI;
    c->measure_sync = 1;
    c->admm_rho = 0.05;
    c->admm_local_steps = 20;
    c->push_every = 1;
    c->fetch_every = 1;
    c->check_finite = 1;
    c->eval_every = 1;
    c->verbose = 1;
}

static void format_value(const pdl_train_config *c, const opt_def *o, char *buf, size_t len) {
    const char *base = (const char *)c + o->offset;
    switch (o->type) {
    case OPT_INT: snprintf(buf, len, "%d", *(const int *)base); break;
    case OPT_LL: snprintf(buf, len, "%lld", *(const long long *)base); break;
    case OPT_DBL: snprintf(buf, len, "%g", *(const double *)base); break;
    case OPT_STR: snprintf(buf, len, "%s", base); break;
    case OPT_ENUM: {
        int idx = *(const int *)base - o->base;
        snprintf(buf, len, "%s", (idx >= 0 && idx < o->nchoices) ? o->choices[idx] : "?");
        break;
    }
    }
}

void pdl_train_config_print_help(FILE *f, const char *prog) {
    pdl_train_config d;
    pdl_train_config_defaults(&d);
    fprintf(f, "Usage: [mpirun -np K] %s [--option value]...\n\n", prog);
    fprintf(f, "Parallel & distributed CNN training benchmark (pdl-bench %s)\n\n", PDL_VERSION_STRING);
    for (size_t i = 0; i < PDL_ARRAY_LEN(k_opts); ++i) {
        char def[600];
        format_value(&d, &k_opts[i], def, sizeof(def));
        fprintf(f, "  --%-18s %s [default: %s]\n", k_opts[i].name, k_opts[i].help, def[0] ? def : "\"\"");
    }
    fprintf(f, "\nSee docs/CLI.md for details.\n");
}

static int set_value(pdl_train_config *c, const opt_def *o, const char *v) {
    char *base = (char *)c + o->offset;
    char *end = NULL;
    switch (o->type) {
    case OPT_INT: {
        long x = strtol(v, &end, 10);
        if (!end || *end) return -1;
        *(int *)base = (int)x;
        return 0;
    }
    case OPT_LL: {
        long long x = strtoll(v, &end, 10);
        if (!end || *end) return -1;
        *(long long *)base = x;
        return 0;
    }
    case OPT_DBL: {
        double x = strtod(v, &end);
        if (!end || *end) return -1;
        *(double *)base = x;
        return 0;
    }
    case OPT_STR:
        if (strlen(v) >= o->size) return -1;
        snprintf(base, o->size, "%s", v);
        return 0;
    case OPT_ENUM:
        for (int i = 0; i < o->nchoices; ++i)
            if (strcmp(v, o->choices[i]) == 0) {
                *(int *)base = i + o->base;
                return 0;
            }
        /* accepted aliases */
        if (strcmp(o->name, "allreduce") == 0 && strcmp(v, "recursive_doubling") == 0) {
            *(int *)base = PDL_AR_RD;
            return 0;
        }
        return -1;
    }
    return -1;
}

int pdl_train_config_parse(pdl_train_config *c, int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) return 1;
        if (strncmp(a, "--", 2) != 0) {
            PDL_ERROR("unexpected argument '%s' (options start with --)", a);
            return -1;
        }
        char name[64];
        const char *val = NULL;
        const char *eq = strchr(a + 2, '=');
        if (eq) {
            snprintf(name, sizeof(name), "%.*s", (int)(eq - a - 2), a + 2);
            val = eq + 1;
        } else {
            snprintf(name, sizeof(name), "%s", a + 2);
        }
        const opt_def *o = NULL;
        for (size_t k = 0; k < PDL_ARRAY_LEN(k_opts); ++k)
            if (strcmp(k_opts[k].name, name) == 0) o = &k_opts[k];
        if (!o) {
            PDL_ERROR("unknown option --%s (see --help)", name);
            return -1;
        }
        if (!val) {
            if (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0)
                val = argv[++i];
            else if (o->type == OPT_INT)
                val = "1"; /* bare boolean flag */
            else {
                PDL_ERROR("option --%s needs a value", name);
                return -1;
            }
        }
        if (set_value(c, o, val) != 0) {
            PDL_ERROR("invalid value '%s' for --%s", val, name);
            return -1;
        }
    }
    /* validation */
    if (c->epochs < 1 || c->batch_size < 1 || c->lr <= 0.0 || c->width < 1) {
        PDL_ERROR("epochs, batch-size, width must be >= 1 and lr > 0");
        return -1;
    }
    if (c->strategy == PDL_STRAT_ADMM && (c->admm_local_steps < 1 || c->admm_rho < 0.0)) {
        PDL_ERROR("admm-local-steps must be >= 1 and admm-rho >= 0");
        return -1;
    }
    if (c->strategy == PDL_STRAT_DOWNPOUR && (c->push_every < 1 || c->fetch_every < 0)) {
        PDL_ERROR("push-every must be >= 1 and fetch-every >= 0");
        return -1;
    }
    if (c->overlap && c->allreduce != PDL_AR_MPI) {
        PDL_ERROR("--overlap requires --allreduce mpi (non-blocking MPI_Iallreduce)");
        return -1;
    }
    return 0;
}

void pdl_train_config_write_json(const pdl_train_config *c, pdl_json *j) {
    pdl_json_begin_object(j, "config");
    for (size_t i = 0; i < PDL_ARRAY_LEN(k_opts); ++i) {
        const opt_def *o = &k_opts[i];
        char key[64];
        snprintf(key, sizeof(key), "%s", o->name);
        for (char *p = key; *p; ++p)
            if (*p == '-') *p = '_';
        const char *base = (const char *)c + o->offset;
        switch (o->type) {
        case OPT_INT: pdl_json_kv_int(j, key, *(const int *)base); break;
        case OPT_LL: pdl_json_kv_int(j, key, *(const long long *)base); break;
        case OPT_DBL: pdl_json_kv_num(j, key, *(const double *)base); break;
        case OPT_STR: pdl_json_kv_str(j, key, base); break;
        case OPT_ENUM: {
            char v[64];
            format_value(c, o, v, sizeof(v));
            pdl_json_kv_str(j, key, v);
            break;
        }
        }
    }
    pdl_json_end_object(j);
}
