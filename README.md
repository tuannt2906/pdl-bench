# pdl-bench — Parallel & Distributed Deep Learning Benchmark

A from-scratch C / MPI / OpenMP / CUDA implementation of CNN training, built to **measure,
analyse and compare** parallel and distributed training methods. It reproduces and extends
*Hegde & Usmani, "Parallel and Distributed Deep Learning"* (Stanford, 2016,
[https://hgpu.org/?p=16046](https://hgpu.org/?p=16046)): the CPU-vs-GPU convolution study (§3) and the
distributed optimizers **Synchronous SGD, Parallel SGD, ADMM and Downpour SGD** (§5),
plus their communication / computation cost models.

> **Tóm tắt (Tiếng Việt).** Bộ mã nguồn tự cài đặt mạng CNN bằng C (forward, backward,
> loss, SGD), chạy trên CPU (naive / blocked / OpenMP) và GPU (CUDA naive / tiled / cuBLAS),
> phân tán bằng MPI với 5 chiến lược huấn luyện: tuần tự, Sync SGD, Parallel SGD, ADMM,
> Downpour SGD. Kèm theo: benchmark tính toán và truyền thông (6 thuật toán allreduce),
> bộ kiểm thử tính đúng (gradient check, so khớp tuần tự ↔ phân tán, đồng bộ replica),
> hệ thống chạy thí nghiệm bằng YAML, tổng hợp kết quả, vẽ biểu đồ, mô hình hiệu năng α-β
> và sinh báo cáo. Máy mục tiêu: **i5-12600K + RTX 2060**, Ubuntu hoặc WSL2.
> Bắt đầu nhanh: xem mục [Quick start](#quick-start); đánh giá đặc tả ban đầu:
> [`docs/SPEC_REVIEW.md`](docs/SPEC_REVIEW.md).

---

## Features

| Area | What is implemented |
|---|---|
| **Neural network (C)** | conv (im2col + GEMM, Caffe style), dense, ReLU, max-pool, softmax cross-entropy (paper eq. 2), SGD + momentum + weight decay, He init; all parameters in one flat buffer; model presets `mlp`, `lenet`, `cnn`, `cifar`, `paper` (96×11×11/4 on 224×224, paper §3) or a spec string; width multiplier for model-size scaling |
| **Compute backends** | CPU `naive` / `blocked` / `omp` (OpenMP over rows / images of the batch), direct convolution; CUDA `naive` / `tiled` (shared memory) / `cublas`, deterministic gather-based col2im and max-pool backward |
| **Strategies** | `sequential` baseline · `sync` data-parallel SGD (allreduce every step, flat or per-layer, optional comm/compute overlap) · `psgd` Parallel SGD with periodic / one-shot model averaging and ShuffleSGD all-to-all reshuffling · `admm` consensus ADMM with local SGD solves · `downpour` asynchronous SGD with a parameter server (n_push / n_fetch, staleness tracking) |
| **Communication** | hand-written allreduce: ring, recursive doubling, binomial tree ("BitTorrent aggregate"), naive central driver; plus `MPI_Allreduce` and `MPI_Reduce + MPI_Bcast` — all bitwise identical across ranks |
| **Measurements** | per-phase timers (data, H2D, forward, backward, transfer, sync-wait, comm, update, eval, shuffle) per rank, load imbalance, bytes/messages sent, throughput, speedup, efficiency, convergence per epoch, time-to-accuracy |
| **Benchmarks** | `pdl_bench_compute` (GEMM, FC, conv fwd/bwd, full train step; CPU vs GPU, batch-size and model-size sweeps), `pdl_bench_comm` (ping-pong α/β, barrier, allreduce × 6 algorithms, bcast, alltoall) |
| **Correctness** | `pdl_check`: finite-difference gradient check (kink-aware), GEMM variants × transposes, CUDA vs CPU kernels, determinism, NaN/Inf detection, allreduce exactness, **sync SGD on k ranks == sequential SGD**, replica bitwise consistency |
| **Experiment management** | YAML configs → cartesian runs × repeats × seeds, mpirun flavour detection, resume, manifest with environment, per-run logs and JSON |
| **Analysis** | aggregation (mean/std/95% CI), speedup/efficiency vs baseline, Hockney α-β performance model vs measurements, Downpour saturation k* = T/τ, PNG + PDF figures, Markdown report, LaTeX tables, AI-agent code metrics |

## Repository layout

```
.
├── CMakeLists.txt            # build (C11, MPI required, OpenMP / CUDA / cuBLAS optional)
├── include/pdl/              # public headers (API contracts, documented)
│   ├── backend.h             #   device abstraction + every numerical kernel
│   ├── model.h               #   layers, model spec, forward / backward
│   ├── dataset.h             #   .pdlds format, synthetic data, partitioning
│   ├── comm.h                #   MPI helpers + allreduce algorithms
│   ├── train.h               #   config, trainer runtime, strategies
│   └── common.h timer.h rng.h json.h env.h cli.h
├── src/
│   ├── core/                 # logging, timers, RNG, JSON writer, CLI, environment capture
│   ├── backend/              # backend.c (dispatch), cpu_kernels.c, cuda_kernels.cu
│   ├── nn/model.c            # CNN implementation
│   ├── data/dataset.c        # loader, synthetic generator, sharding
│   ├── comm/comm.c           # ring / recursive-doubling / tree / naive allreduce
│   └── train/                # trainer.c, config.c, strategy_{sync,psgd,admm,downpour}.c
├── apps/                     # parallel programs: pdl_train, pdl_bench_compute, pdl_bench_comm, pdl_check
├── sequential/               # sequential version: seq_train.c + Makefile (plain C)
├── tests/CMakeLists.txt      # CTest: correctness suite + end-to-end runs of every strategy
├── configs/                  # experiment suite (YAML)
├── pytools/pdlbench/         # Python: data, run, aggregate, plot, perfmodel, report, ai-eval
├── scripts/                  # setup_ubuntu.sh, build.sh, prepare_data.sh, quickstart.sh, run_all.sh
├── docs/                     # architecture, algorithms, experiments, CLI, schema, setup, roadmap
├── .claude/skills/           # Claude Code project skills (run experiments, extend the code)
├── CLAUDE.md                 # project context for Claude Code sessions
├── data/   results/          # created locally (git-ignored)
└── LICENSE                   # MIT
```

## Requirements

| | Target machine | Minimum |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04, or Windows 11 + **WSL2** Ubuntu | any Linux / macOS (CPU only) |
| CPU | Intel i5-12600K (6 P-cores + 4 E-cores, 16 threads) | x86-64 / ARM64 |
| GPU | NVIDIA RTX 2060 6 GB (Turing, `sm_75`) | optional |
| Toolchain | GCC ≥ 9, CMake ≥ 3.18, OpenMPI ≥ 4 (or MPICH), CUDA Toolkit ≥ 11.0 | GCC/Clang, CMake, any MPI |
| Python | ≥ 3.9 with numpy, pandas, matplotlib, pyyaml | same |

`scripts/setup_ubuntu.sh` installs the system packages and explains how to install the
CUDA toolkit (native Ubuntu or WSL2). See [`docs/SETUP.md`](docs/SETUP.md) for details.

## Quick start

```bash
git clone <repo-url> pdl-bench && cd pdl-bench
scripts/setup_ubuntu.sh            # apt packages (sudo); prints CUDA instructions if nvcc is missing
scripts/quickstart.sh              # .venv + build + MNIST + smoke test + correctness suite
```

Step by step instead:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .     # provides the `pdlbench` command

scripts/build.sh                   # CMake Release build, CUDA auto-detected (sm_75)
ctest --test-dir build --output-on-failure              # correctness + end-to-end tests
pdlbench data --dataset mnist cifar10                   # -> data/*.pdlds
```

### Sequential version (plain C, no MPI / OpenMP / CUDA)

`sequential/seq_train.c` trains the CNN in a single thread using only the shared core
(`src/core`, `src/nn`, `src/data`, CPU kernels). It follows the same sample order and
initialization as `pdl_train --strategy sequential`, and both print identical losses.

```bash
cd sequential && make            # gcc only; or use build/bin/seq_train from the CMake build
./seq_train --epochs 2 --output seq.json
```

### Run the programs directly

```bash
# sequential baseline (single process)
build/bin/pdl_train --strategy sequential --epochs 3 --output out/seq.json

# synchronous SGD on 4 ranks, ring allreduce, global batch 256
mpirun -np 4 --bind-to core build/bin/pdl_train --strategy sync --allreduce ring \
       --global-batch 256 --lr 0.05 --output out/sync4.json

# the same on the GPU
build/bin/pdl_train --strategy sequential --device cuda --model cifar --width 2 \
       --train-data data/cifar10-train.pdlds --test-data data/cifar10-test.pdlds

# Parallel SGD with ShuffleSGD, ADMM, Downpour (rank 0 = parameter server)
mpirun -np 4 build/bin/pdl_train --strategy psgd --avg-period 10 --reshuffle 1
mpirun -np 4 build/bin/pdl_train --strategy admm --admm-rho 0.05 --admm-local-steps 20
mpirun -np 5 build/bin/pdl_train --strategy downpour --push-every 2 --fetch-every 4

# paper §3: conv layer CPU vs GPU as a function of batch size
build/bin/pdl_bench_compute --op conv --impls cpu-omp,cuda-tiled,cublas --batches 1,16,64,256

# allreduce algorithms vs message size
mpirun -np 8 build/bin/pdl_bench_comm --algos all --max-bytes 67108864

# correctness (MPI tests need >= 2 ranks)
build/bin/pdl_check --device both
mpirun -np 4 build/bin/pdl_check --tests allreduce,equivalence,sync
```

Every option is documented in [`docs/CLI.md`](docs/CLI.md); every program prints `--help`.

### Run experiments

```bash
pdlbench run configs/strong_scaling.yaml --dry-run     # show the expanded commands
pdlbench all configs/strong_scaling.yaml               # run + aggregate + plot + perfmodel + report
pdlbench perfmodel results/strong_scaling --comm-dir results/comm_allreduce
scripts/run_all.sh                                     # whole suite (several hours)
REPEATS=1 scripts/run_all.sh --cpu-only                # quick pass without the GPU
```

Results land in `results/<experiment>/<timestamp>/` (`latest` symlink):

```
config.yaml  manifest.json  raw/*.json  logs/*.log
analysis/*.csv *.tex        figures/*.png *.pdf     report.md
```

## Experiment suite

| Config | Research question (spec §11) | ≈ Runtime |
|---|---|---|
| `smoke.yaml` | toolchain sanity: every binary and strategy | 1–3 min |
| `correctness.yaml` | correctness never traded for speed | < 5 min |
| `compute_paper_cpu_gpu.yaml` | paper §3: conv / FC time vs batch / matrix size, GPU speedup | 15–25 min |
| `compute_gemm.yaml` | GEMM implementations (naive → blocked → OpenMP → CUDA → cuBLAS) | ~10 min |
| `compute_batch_model.yaml` | effect of batch size and model width on compute throughput | ~20 min |
| `comm_allreduce.yaml` | MPI allreduce algorithms × message size × process count; α-β fit | 10–15 min |
| `strong_scaling.yaml` | speedup / efficiency vs #processes at fixed global batch | 20–30 min |
| `weak_scaling.yaml` | fixed work per process | 15–20 min |
| `hybrid_mpi_omp.yaml` | MPI ranks × OpenMP threads at a fixed core budget | ~20 min |
| `batch_scaling.yaml` | batch-size sensitivity, CPU vs GPU | ~15 min |
| `model_scaling.yaml` | model size vs communication fraction and scalability | 30–40 min |
| `strategies_convergence.yaml` | trade-offs of Sync / Parallel SGD / ADMM / Downpour, IID vs label-sorted shards | 1.5–2.5 h |
| `comm_optimization.yaml` | do communication optimizations (algorithm, fusion, overlap) help? | 30–40 min |
| `gpu_mpi.yaml` | several ranks sharing one GPU vs CPU ranks | ~20 min |
| `paper_conv_model.yaml` | paper network (224×224, 40 classes) on CPU vs GPU | 20–30 min |

Configuration schema and methodology: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Correctness

`ctest` runs the full suite (≈ 10 s on CPU). Highlights:

* **Gradient check** of every layer type against central finite differences; probes that
  cross a ReLU / max-pool kink are detected (activation-pattern signature) and skipped.
* **Sequential ↔ distributed equivalence**: sync SGD on k ranks with local batch b must match
  sequential SGD with batch k·b to 1e-4 (only floating-point summation order differs).
* **Replica consistency**: after every allreduce algorithm the k model replicas are **bitwise**
  identical (`--check-sync N` does this during real training runs).
* **CUDA vs CPU**: every CUDA kernel and GEMM variant is compared with the CPU reference.
* Runs abort on NaN/Inf (`--check-finite`), and every result file embeds the full config,
  git commit, compiler, MPI library, CPU/GPU model and CUDA versions.

## Hardware notes (i5-12600K + RTX 2060)

* **Hybrid CPU**: P-cores and E-cores run at different speeds, so with more than 6 ranks a
  synchronous step waits for E-core stragglers (visible as `sync_wait` / `load_imbalance`).
  Compare `--bind-to core` placements and np ≤ 6 vs np > 6; this is a finding, not a bug.
* **Single node**: MPI uses shared memory; α/β describe intra-node copies, and scaling is
  ultimately limited by memory bandwidth (the α-β model under-predicts at high np).
* **One GPU**: `gpu_mpi.yaml` places several ranks on the same RTX 2060 (time-sliced);
  6 GB limits batch × width — the `paper` model at batch 256 needs ≈ 3.5 GB (im2col buffers
  dominate: conv2 alone holds 2400 × 676 floats per image), so use batch ≤ 64 when several
  ranks share the GPU.
* **WSL2** works (CUDA via the Windows driver, OpenMPI inside Ubuntu); keep the repository
  on the Linux filesystem (`~/…`), not `/mnt/c`, for I/O speed.

## Documentation

| Document | Content |
|---|---|
| [`docs/SPEC_REVIEW.md`](docs/SPEC_REVIEW.md) | review of the original specification: what was implemented, adjusted or skipped (Vietnamese) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | modules, data flow, design decisions, risks |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | the strategies and collectives, mapped to the paper, with cost models |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | benchmark methodology, YAML schema, reproducibility checklist |
| [`docs/CLI.md`](docs/CLI.md) | every command-line option |
| [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md) | JSON result format |
| [`docs/SETUP.md`](docs/SETUP.md) | installing the toolchain on Ubuntu / WSL2, troubleshooting |
| [`docs/AI_AGENT_EVALUATION.md`](docs/AI_AGENT_EVALUATION.md) | protocol for evaluating AI-generated HPC code (this repo is the subject) |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | milestones, verification status, next steps |

## Project status

Version 1.0.0. CPU + MPI code paths are built and tested (CTest suite, MNIST training with
all strategies, the full Python pipeline). The CUDA backend passes a full CUDA syntax and
semantic compile check but has **not yet been run on a GPU** — the first session on the
target machine should run `ctest` with CUDA enabled (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).

## Citation

If you use this benchmark, cite it via [`CITATION.cff`](CITATION.cff) and the original report:
V. Hegde, S. Usmani, *Parallel and Distributed Deep Learning*, Stanford University.

## License

[MIT](LICENSE). The reference paper is not redistributed here (copyright of its authors);
see the link above.
