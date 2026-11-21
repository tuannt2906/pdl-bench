# Command-line reference

All binaries are built into `build/bin/`. Every option takes a value
(`--overlap 1`, not just `--overlap`) except `--help`; boolean options also accept a
bare flag form. Lists are comma separated. All binaries write one JSON result file
(`--output`) whose format is described in [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md).

MPI programs are launched with `mpirun -np <k> [--bind-to core --map-by core] build/bin/<prog> ...`.
A single-process run (`build/bin/pdl_train ...` without `mpirun`) is valid and is the sequential baseline.

---

## `pdl_train` — training with a distributed strategy

| Option | Default | Description |
|---|---|---|
| `--strategy` | `sync` | `sequential` \| `sync` \| `psgd` \| `admm` \| `downpour` |
| `--device` | `cpu` | `cpu` \| `cuda` |
| `--impl` | `auto` | kernel variant: `naive` \| `blocked` \| `omp` (CPU), `naive` \| `tiled` \| `cublas` (CUDA). `auto` = `omp` on CPU (`blocked` without OpenMP), `cublas` on CUDA (`tiled` without cuBLAS) |
| `--threads` | `0` | OpenMP threads per rank (0 = `OMP_NUM_THREADS` / runtime default) |
| `--gpu-id` | `-1` | CUDA device; `-1` = local rank modulo GPU count |
| `--timing-sync` | `1` | synchronize the GPU at phase boundaries (exact phase times, small overhead) |
| `--train-data` | `data/mnist-train.pdlds` | `.pdlds` file or `synthetic:N:C:H:W:K` |
| `--test-data` | `data/mnist-test.pdlds` | same; `none` disables evaluation |
| `--train-limit` / `--test-limit` | `0` | use only the first N samples (0 = all) |
| `--partition` | `auto` | `global` (replicated, slice of a global batch) \| `iid` \| `sorted` (non-IID shards). `auto` = `global` for sequential/sync, `iid` otherwise |
| `--model` | `lenet` | preset (`mlp`, `lenet`, `cnn`, `cifar`, `paper`) or spec string, e.g. `conv:32:5:1:2,relu,pool:2:2,fc:10` |
| `--width` | `1` | channel/hidden-unit multiplier for presets (model-size scaling) |
| `--epochs` | `5` | training epochs (epoch = one pass over the local shard) |
| `--batch-size` | `64` | per-rank mini-batch |
| `--global-batch` | `0` | if > 0, per-rank batch = global / #workers (strong scaling) |
| `--lr` | `0.01` | learning rate |
| `--momentum` | `0.9` | SGD momentum (0 = plain SGD) |
| `--weight-decay` | `0` | L2 penalty |
| `--lr-decay` / `--lr-step` | `1.0` / `0` | multiply lr by decay every `lr-step` epochs |
| `--lr-scale` | `0` | sync: multiply lr by #workers (linear scaling rule) |
| `--max-steps` | `0` | cap on steps per epoch (0 = full epoch) — for quick benchmarks |
| `--warmup-steps` | `0` | untimed warm-up steps before epoch 1 (parameters are restored afterwards) |
| `--seed` | `42` | global seed (init, shuffling, synthetic data) |
| `--allreduce` | `mpi` | `mpi` \| `ring` \| `rd` \| `tree` \| `naive` \| `reduce_bcast` |
| `--fusion` | `flat` | `flat` = one allreduce over all gradients, `layer` = one per layer |
| `--overlap` | `0` | overlap per-layer non-blocking allreduce with backward (`--allreduce mpi` only) |
| `--measure-sync` | `1` | barrier before communication so waiting (load imbalance) is timed separately |
| `--avg-period` | `0` | psgd: local steps between model averaging; `0` = once per epoch; `-1` = one-shot at end |
| `--reshuffle` | `0` | psgd: ShuffleSGD all-to-all data redistribution after every epoch |
| `--admm-rho` | `0.05` | ADMM penalty ρ |
| `--admm-local-steps` | `20` | ADMM: local SGD steps per consensus round |
| `--push-every` | `1` | downpour: local steps between gradient pushes (n_push) |
| `--fetch-every` | `1` | downpour: local steps between parameter fetches (n_fetch) |
| `--server-lr` | `0` | downpour server learning rate (0 = `--lr`) |
| `--check-sync` | `0` | every N steps verify all replicas are bitwise identical (sync) |
| `--check-finite` | `1` | abort on NaN/Inf in loss or gradients |
| `--eval-every` | `1` | evaluate every N epochs (0 = only after the last epoch) |
| `--output` | `""` | result JSON path |
| `--log-steps` | `0` | also write `<output>.steps.csv` every N steps (rank 0) |
| `--dump-params` | `""` | write final parameters (raw float32) for offline comparison |
| `--run-id` / `--tag` | `""` | free-form identifiers copied into the result |
| `--verbose` | `1` | 0 quiet, 1 per-epoch, 2 debug |

## `pdl_bench_compute` — computation micro-benchmarks (paper §3)

| Option | Default | Description |
|---|---|---|
| `--op` | `gemm` | list of `gemm`, `fc`, `conv`, `conv-bwd`, `train-step` |
| `--impls` | `all` | list of `cpu-naive`, `cpu-blocked`, `cpu-omp`, `cpu-direct`, `cuda-naive`, `cuda-tiled`, `cublas`, `cuda-direct`; `all` = every implementation available in the build (`*-direct` only applies to `conv`) |
| `--sizes` | `64,128,256,512,1024` | `gemm`: square sizes N (N×N×N); `fc`: matrix size n (n×n weights) |
| `--batches` | `1,8,16,32,64,128,256` | batch sizes for `conv`, `conv-bwd`, `train-step` |
| `--fc-batch` | `1` | batch size(s) for `fc` (1 = matrix-vector product, as in the paper's Fig. 3–4) |
| `--conv` | `1:224:224:96:11:4:0` | `C:H:W:OC:K:S:P` — default is the paper's first layer |
| `--model` / `--width` / `--classes` / `--input` | `lenet` / `1` / `10` / `1:28:28` | model for `train-step` (`--input C:H:W`) |
| `--reps` / `--warmup` | `10` / `2` | timed repetitions / untimed warm-up runs |
| `--threads` | `0` | OpenMP threads |
| `--gpu-id` | `0` | CUDA device |
| `--include-transfer` | `0` | include host↔device copies of inputs/outputs in GPU timings |
| `--verify` | `1` | compare every result with the CPU-blocked reference |
| `--max-naive` | `5` | skip `cpu-naive` / `cpu-direct` when one repetition exceeds this many GFLOP (they are very slow) |
| `--seed`, `--output`, `--run-id`, `--tag` | | as above |

## `pdl_bench_comm` — MPI communication benchmarks

| Option | Default | Description |
|---|---|---|
| `--algos` | `all` | list of `mpi`, `ring`, `rd`, `tree`, `naive`, `reduce_bcast` |
| `--sizes` | (sweep) | explicit message sizes in bytes; otherwise `--min-bytes`..`--max-bytes` × `--factor` |
| `--min-bytes` / `--max-bytes` / `--factor` | `4` / `67108864` / `4` | geometric message-size sweep |
| `--reps` / `--warmup` | `20` / `3` | repetitions per point (time = max over ranks, then mean over reps) |
| `--pingpong` | `1` | rank 0 ↔ rank 1 ping-pong (latency α, bandwidth 1/β for the performance model) |
| `--bcast` | `1` | `MPI_Bcast` sweep (parameter broadcast cost) |
| `--alltoall` | `1` | `MPI_Alltoall` sweep (ShuffleSGD cost) |
| `--device` | `cpu` | `cuda`: buffers live on the GPU and are staged through host memory (cost of GPU gradient sync) |
| `--verify` | `1` | check allreduce results against the analytic sum |
| `--seed`, `--output`, `--run-id`, `--tag` | | |

## `pdl_check` — correctness suite

| Option | Default | Description |
|---|---|---|
| `--tests` | `all` | list of `gradcheck`, `kernels`, `determinism`, `allreduce`, `equivalence`, `sync`, `finite` |
| `--device` | `cpu` | `cpu` \| `cuda` \| `both` |
| `--tol` | `1e-3` | relative tolerance for kernel comparison (gradcheck uses its own, see output) |
| `--steps` | `5` | training steps for `equivalence` / `determinism` |
| `--output` / `--verbose` | `""` / `1` | |

MPI tests (`allreduce`, `equivalence`, `sync`) need ≥ 2 ranks: `mpirun -np 4 build/bin/pdl_check --tests allreduce,equivalence,sync`.
The exit code is non-zero if any test fails.
