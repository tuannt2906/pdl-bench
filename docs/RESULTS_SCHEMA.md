# Result file schema

Every binary writes a single JSON document. The `schema` field identifies the
producer and version. All times are **seconds**, sizes are **bytes** unless the
key says otherwise. The Python tooling (`pytools/pdlbench`) depends on these keys.

## Common header (all binaries)

```jsonc
{
  "schema": "pdl-bench/train/v1",        // train | compute | comm | check
  "version": "1.0.0",
  "run_id": "...", "tag": "...",
  "config": { /* every CLI option, keys use '_' instead of '-' */ },
  "environment": {
    "timestamp": "2026-11-21T10:00:00Z", "hostname": "...", "os": "Linux 6.5 x86_64",
    "cpu_model": "12th Gen Intel(R) Core(TM) i5-12600K", "logical_cpus": 16,
    "compiler": "GNU 12.3.0", "build_type": "Release", "build_flags": "-O3 -march=native",
    "git_commit": "abc123", "git_dirty": false,
    "mpi_library": "Open MPI v4.1.6 ...", "mpi_world_size": 4, "mpi_hosts": 1,
    "openmp": true, "omp_max_threads": 16,
    "cuda": true, "cuda_runtime": 12040, "cuda_driver": 12040,
    "gpu_name": "NVIDIA GeForce RTX 2060", "gpu_memory_bytes": 6442450944,
    "gpu_sm_count": 30, "gpu_cc": "7.5", "cublas": true,
    "device": "cpu", "impl": "cpu-omp",
    "command_line": "build/bin/pdl_train --strategy sync ..."
  }
}
```

## `pdl-bench/train/v1` (pdl_train)

```jsonc
{
  ...header,
  "strategy": "sync", "world_size": 4, "num_workers": 4,
  "dataset": { "train": "data/mnist-train.pdlds", "test": "...", "n_train": 60000, "n_test": 10000,
               "shape": [1, 28, 28], "classes": 10, "partition": "global" },
  "model": { "spec": "conv:6:5:1:2,relu,...", "num_params": 61706, "param_bytes": 246824,
             "flops_per_sample": 846000.0,
             "layers": [ { "name": "conv1", "kind": "conv", "in": [1,28,28], "out": [6,28,28],
                           "params": 156, "flops": 235200.0 } ] },
  "local_batch": 64, "global_batch": 256, "steps_per_epoch": 234,
  "summary": {
    "train_time_s": 12.3,           // max over ranks, training only (no eval)
    "wall_time_s": 13.0,            // whole run on rank 0, includes evaluation
    "samples_processed": 300000,    // all workers, all epochs
    "throughput_samples_per_s": 24390.2,
    "final_train_loss": 0.05, "final_train_acc": 0.98,
    "final_test_loss": 0.04, "final_test_acc": 0.987, "best_test_acc": 0.988,
    "phases_mean": { "data": 0.1, "h2d": 0.0, "forward": 3.0, "backward": 5.0, "transfer": 0.0,
                     "sync_wait": 0.3, "comm": 1.2, "update": 0.1, "eval": 0.5, "shuffle": 0.0 },
    "phases_max":  { ... }, "phases_min": { ... },   // statistics across worker ranks
    "compute_time_mean_s": 8.1,     // data + h2d + forward + backward + update (mean over workers)
    "comm_time_mean_s": 1.2,        // comm + transfer
    "sync_wait_mean_s": 0.3,
    "comm_fraction": 0.11,          // (comm + transfer + sync_wait) / train_time (mean over workers)
    "load_imbalance": 1.04,         // max / mean of compute time across workers
    "comm_bytes_per_rank": 1.2e9,   // mean over ranks
    "comm_calls_per_rank": 1170,
    "sync_checks": 0, "sync_failures": 0, "nonfinite_events": 0,
    "server": {                     // downpour only, else null
      "updates": 4000, "mean_staleness": 2.1, "max_staleness": 7,
      "phases": { ... }             // server-side phase totals
    }
  },
  "epochs": [
    { "epoch": 1, "train_loss": 0.3, "train_acc": 0.91,
      "test_loss": 0.1, "test_acc": 0.97,          // null if not evaluated
      "time_s": 2.4,                               // max over workers
      "elapsed_s": 2.4,                            // cumulative training time (max over workers)
      "samples": 60000, "steps": 234,              // global samples, per-worker steps
      "phases_mean": { ... }, "phases_max": { ... } }
  ],
  "per_rank": [
    { "rank": 0, "role": "worker", "train_time_s": 12.2, "samples": 75000,
      "phases": { ... }, "comm_bytes": 1.2e9, "comm_calls": 1170 }
  ]
}
```

`<output>.steps.csv` (when `--log-steps N`): `global_step,epoch,loss,acc,elapsed_s` from rank 0.

## `pdl-bench/compute/v1` (pdl_bench_compute)

```jsonc
{
  ...header,
  "records": [
    { "op": "gemm",       // gemm | fc | conv | conv-bwd | train-step
      "impl": "cuda-tiled", "device": "cuda", "threads": 16,
      "M": 512, "N": 512, "K": 512,   // gemm / fc (fc: M = batch, N = K = n)
      "n": 512,                       // fc matrix size / gemm square size
      "batch": 64,                    // fc / conv / train-step
      "conv": "1:224:224:96:11:4:0",  // conv ops
      "model": "lenet",               // train-step
      "flops": 2.6e8,                 // per repetition
      "time_mean_s": 0.0012, "time_std_s": 1e-5, "time_min_s": 0.0011, "time_median_s": 0.0012,
      "reps": 10, "gflops": 216.0,
      "throughput_samples_per_s": 53000.0,   // ops with a batch
      "include_transfer": false,
      "max_abs_err": 1e-5, "rel_err": 2e-7, "verified": true }   // null when not verified
  ]
}
```

## `pdl-bench/comm/v1` (pdl_bench_comm)

```jsonc
{
  ...header,
  "world_size": 4, "device": "cpu",
  "pingpong": [ { "bytes": 1024, "one_way_s": 1.1e-6, "bandwidth_Bps": 9.3e8, "reps": 20 } ],
  "barrier": { "mean_s": 2e-6, "reps": 20 },
  "allreduce": [
    { "algo": "ring", "bytes": 4194304, "count": 1048576,
      "time_mean_s": 1.0e-3, "time_std_s": 2e-5, "time_min_s": 9.8e-4, "time_max_s": 1.1e-3,
      "reps": 20,
      "algbw_Bps": 4.1e9,      // bytes / time
      "busbw_Bps": 6.2e9,      // algbw * 2(k-1)/k (NCCL convention)
      "bytes_sent_per_rank": 6291456, "messages_per_rank": 6,
      "correct": true }
  ],
  "bcast":    [ { "bytes": 4096, "time_mean_s": ..., "time_std_s": ..., "time_min_s": ..., "reps": 20 } ],
  "alltoall": [ { "bytes": 4096, ... } ]   // bytes = total per rank send buffer
}
```

## `pdl-bench/check/v1` (pdl_check)

```jsonc
{
  ...header,
  "world_size": 4,
  "passed": true,
  "tests": [ { "name": "gradcheck/conv", "device": "cpu", "passed": true,
               "max_err": 3e-4, "tolerance": 1e-2, "detail": "..." } ]
}
```
