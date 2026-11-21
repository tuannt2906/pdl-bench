---
name: pdl-bench-experiments
description: Build, validate and run pdl-bench on the target machine (i5-12600K + RTX 2060, Ubuntu/WSL2), run the experiment campaign, interpret benchmark results and draft the research report. Use when the user asks to build, test, run experiments/benchmarks, analyse results, fix build/runtime problems on the target, or write up findings.
---

# Running pdl-bench experiments

## 0. Orient
Read `CLAUDE.md` and `docs/ROADMAP.md` (status table + first-session checklist). Check the
machine before running anything:
```bash
nproc; lscpu | grep -E "Model name|Thread|Core"; nvidia-smi -L; nvcc --version | tail -1
mpirun --version | head -1; cmake --version | head -1; python3 --version
```

## 1. Build and validate (always before measuring)
```bash
scripts/setup_ubuntu.sh            # only if packages / .venv are missing
scripts/build.sh                   # check the summary: OpenMP ON, CUDA ON, arch 75
ctest --test-dir build --output-on-failure
build/bin/pdl_check --device both
mpirun -np 4 build/bin/pdl_check --tests allreduce,equivalence,sync --device both
```
* A failing CUDA test means a real bug, because the CUDA backend has never run before the
  first target session. Find the kernel from the test name (`kernels/gemm-cuda-tiled-NT`,
  `kernels/cnn-pipeline-cuda-vs-cpu`, …), compare with the CPU version in `cpu_kernels.c`, fix
  it in `cuda_kernels.cu`, rebuild, rerun. **Never** loosen tolerances to pass.
* Build problems: see the troubleshooting table in `docs/SETUP.md` (host compiler too new,
  CUDA path, OpenMPI slots/root).
* Record every fix in the effort log of `docs/AI_AGENT_EVALUATION.md` §4 (date, symptom, cause,
  fix, minutes spent). This is research data.

## 2. Data
`pdlbench data --dataset mnist cifar10` writes `data/{mnist,cifar10}-{train,test}.pdlds`.
Synthetic data needs no download: `--train-data synthetic:N:C:H:W:K`.

## 3. Campaign
Order (cheap → expensive). Each step is `pdlbench all configs/<name>.yaml`:
1. `smoke`, `correctness`: must be all green.
2. `comm_allreduce`: needed by `perfmodel` for α/β (`--comm-dir results/comm_allreduce`).
3. `compute_gemm`, `compute_paper_cpu_gpu`, `compute_batch_model`, `paper_conv_model`.
4. `strong_scaling`, `weak_scaling`, `hybrid_mpi_omp`, `batch_scaling`, `model_scaling`.
5. `comm_optimization`, `gpu_mpi`, `strategies_convergence` (longest, about 2 h).
Or run everything with `scripts/run_all.sh` (several hours; `REPEATS=1` for a quick pass).
Prepare the machine first (governor, persistence mode, closed apps): `docs/EXPERIMENTS.md` §1.
Use `--dry-run` to inspect commands, and `--resume results/<exp>/<ts>` after interruptions.

## 4. Interpreting results
Each run directory has `analysis/*.csv`, `figures/*.png|pdf` and `report.md`. Key columns:
`train_time_s`, `throughput_samples_per_s`, `speedup`, `efficiency`, `comm_fraction`,
`load_imbalance`, `phases_mean.*`, `final_test_acc`, `time_to_target`.

Things to look for (expected findings, verify against the data before claiming them):
* Paper §3: GPU/CPU ratio for conv grows with batch size. Paper ≈ 40× on a GRID K520;
  RTX 2060 vs 12600K will differ. Report the measured ratio and the trend.
* Strong scaling of `sync`: efficiency drops when compute per rank shrinks (small LeNet), and
  at np > 6 E-core stragglers show up as `sync_wait` and `load_imbalance`. Memory bandwidth
  makes perfmodel under-predict time at high np.
* Allreduce: `rd`/`tree` win for small messages (latency), `ring`/`mpi` for large ones
  (bandwidth), `naive` scales worst. Compare with the α-β predictions.
* Strategies: `psgd`/`admm` communicate less per sample than `sync`. With `--partition sorted`
  (non-IID), psgd without reshuffle converges worse and `--reshuffle 1` recovers it.
  Downpour: staleness grows with workers; the server saturates near k* = T/τ (perfmodel).
* `comm_optimization`: fusion `flat` vs `layer` (latency terms), overlap gains depend on MPI
  progress. Report negative results as they are.
Every claim must come from a number in the CSVs. Quote the CSV and figure.

## 5. Report
`pdlbench report results/<a> results/<b> ... --out results/report.md` builds a combined report.
For the research write-up, answer each research question of `docs/reference/overview.md` §11
with: setup (config + environment), figure, numbers with 95% CI, explanation, limitation
(single node, hybrid cores, one GPU). Curated runs can be copied to `results/published/`
(tracked by git).
