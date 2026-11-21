# Experiments: methodology and configuration

## 1. Measurement methodology

| Aspect | Rule |
|---|---|
| **Clock** | `clock_gettime(CLOCK_MONOTONIC)`; GPU work is synchronized at phase boundaries when `--timing-sync 1` (default). |
| **Training time** | Sum of epoch times, **excluding evaluation**. A run's `train_time_s` is the maximum over workers, since a synchronous job ends with its slowest rank. |
| **Collectives** | Each repetition is timed on every rank and the **maximum over ranks** is kept. Statistics are taken over repetitions. |
| **Warm-up** | `--warmup` repetitions (benchmarks) or `--warmup-steps` (training, state restored) are discarded. |
| **Repeats & seeds** | Every configuration runs `repeats` times with `seed = seed_base + repeat`. Reported values are mean ± 95% CI (Student t). |
| **Throughput** | Samples processed by all workers / training time. |
| **Speedup** | Strong scaling: `S(k) = T(1) / T(k)` at the **same global batch and sample sequence** (`--global-batch`, `--partition global`). |
| **Efficiency** | Strong: `E(k) = S(k)/k`. Weak (work per rank fixed, dataset scaled with k): `E(k) = T(1)/T(k)`. |
| **Comm overhead** | `comm_fraction = (comm + transfer + sync_wait + shuffle) / train_time`, mean over workers. |
| **Load imbalance** | `max / mean` of per-rank compute time (data + h2d + forward + backward + update). |
| **Convergence** | Test loss/accuracy per epoch (`--eval-every`) vs epoch and vs elapsed training time. Time-to-accuracy is derived from them. |
| **Correctness gate** | `--check-finite 1` aborts on NaN/Inf. `--check-sync N` verifies replica hashes. Each campaign starts with `configs/correctness.yaml`. |

### Before a measurement campaign (target machine)

```bash
sudo cpupower frequency-set -g performance     # or: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo nvidia-smi -pm 1                           # GPU persistence mode (native Linux)
export OMP_PROC_BIND=close OMP_PLACES=cores     # stable OpenMP placement
```
Close other heavy applications. On WSL2 the governor is controlled by Windows: use the
"Best performance" power mode. Record anything unusual in the experiment description.
`pdlbench env` also stores `lscpu`, `nvidia-smi` and tool versions in every manifest.

### Process placement on the i5-12600K

Cores 0–11 are the 6 hyper-threaded P-cores and 12–15 the 4 E-cores (check with `lscpu -e`).
`--bind-to core --map-by core` fills P-cores first with OpenMPI. Runs with np > 6 put ranks on
E-cores and show straggler effects in `sync_wait`. To compare, restrict to P-cores with
`--cpu-set 0,2,4,6,8,10`.

## 2. Experiment configuration (YAML)

```yaml
name: strong_scaling            # results/<name>/<timestamp>/
description: "..."
binary: pdl_train               # pdl_train | pdl_bench_compute | pdl_bench_comm | pdl_check
repeats: 3                      # repetitions of every parameter combination
vary_seed: true                 # seed = seed_base + repeat
seed_base: 42
timeout: 3600                   # seconds per run
mpi:
  launcher: mpirun
  np: [1, 2, 4, 6, 8, 10]       # process counts (matrix key `np`)
  extra_args: [--bind-to, core, --map-by, core]
  allow_oversubscribe: true     # adds --oversubscribe / hwthread binding when np > cores
  launch_single: false          # true: np == 1 still uses mpirun
env: {OMP_NUM_THREADS: 1}
base_args:                      # CLI options of the binary (`--key value`)
  strategy: sync
  global-batch: 256
matrix:                         # cartesian product; may contain np, binary and env.X keys
  allreduce: [mpi, ring]
variants:                       # optional overrides crossed with the matrix
  - {strategy: sequential, np: 1}
exclude:                        # optional partial parameter sets to skip
  - {device: cuda, np: 8}
analysis:
  group_by: [strategy, np]      # columns identifying a configuration
  baseline: {np: 1}             # reference rows for speedup / efficiency
  match_on: [strategy]          # keys shared with the baseline row
  scaling: strong               # strong | weak
  x: np
  plots: [speedup, efficiency, phase_breakdown]
```

Value rules: booleans become `1`/`0`, lists are comma-joined, and `null` drops the option.
A string starting with `=` is a Python expression over `np`, `repeat`, `seed`, `min`, `max`,
`int`, `float`, `round` and all other parameters (dashes → underscores), for example
`train-limit: "=6000*np"` (weak scaling) or `env.OMP_NUM_THREADS: "=12//np"` (hybrid).

The runner always adds `--output <run_dir>/raw/<run>.json` and, except for `pdl_check`,
`--seed`, `--run-id`, `--tag`. It detects OpenMPI and MPICH and translates the binding flags.

### Available plots
`training_time, speedup, efficiency, throughput, phase_breakdown, strong_scaling,
weak_scaling, convergence, model_size, batch_size, comm_allreduce, comm_scaling, pingpong,
compute_cpu_gpu, gemm, perfmodel`. Each one is written as a PNG and a PDF.

## 3. Workflow

```bash
pdlbench run configs/X.yaml --dry-run          # inspect commands
pdlbench run configs/X.yaml                    # execute (resumable: --resume results/X/<ts>)
pdlbench aggregate results/X                   # → analysis/*.csv, *.tex
pdlbench plot results/X                        # → figures/*.png, *.pdf
pdlbench perfmodel results/X --comm-dir results/comm_allreduce
pdlbench report results/X results/Y --out results/report.md
pdlbench all configs/X.yaml                    # all of the above
```

`results/<exp>/latest` points to the newest run directory. To keep results in git, copy
curated run directories to `results/published/` (not ignored).

## 4. Reproducing a result

1. Check out the commit in `environment.git_commit` of the result file.
2. Rebuild with the same `build_type` and flags (`environment.build_flags`).
3. Re-run with the stored `config.yaml` of the run directory (`pdlbench run <run_dir>/config.yaml`).
4. Compare with `pdlbench aggregate`. Expect identical losses on the same device and binary,
   and timing differences within the reported CIs.

## 5. Adding an experiment

Copy the closest config, change `name`, `base_args` and `matrix`, run with `--dry-run`, then
add it to `scripts/run_all.sh` and to the table in the README.
