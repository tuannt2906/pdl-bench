# Roadmap and verification status

## Milestones

| # | Milestone | Done when | Status |
|---|---|---|---|
| M0 | Spec review & architecture | `SPEC_REVIEW.md`, `ARCHITECTURE.md` | ✅ |
| M1 | Sequential CNN in C (CPU) | gradcheck passes; LeNet ≥ 98% on MNIST | ✅ verified (98.5% after 2 epochs) |
| M2 | Compute backends | CPU naive/blocked/omp + CUDA naive/tiled/cuBLAS agree with the reference | ✅ CPU, OpenMP and CUDA verified on the target (35/35 `pdl_check`, 13/13 `ctest`) |
| M3 | MPI data parallelism | sync SGD == sequential (1e-4); replicas bitwise identical; 6 allreduce algorithms exact | ✅ verified (2–4 ranks, MPICH) |
| M4 | Distributed optimizers | psgd / admm / downpour converge on MNIST | ✅ verified (4 ranks) |
| M5 | Benchmarks + experiment pipeline | compute & comm benchmarks, YAML runner, aggregate, plots, perfmodel, report | ✅ verified end-to-end (CPU) |
| M6 | Measurement campaign on the target | results + report committed | ✅ 2026-11-23 on i5-12600K + RTX 2060 under WSL2: `experiments/results/dot2_*`, `SUMMARY_dot2.md` (149 runs, 0 failures). E-core/P-core effects could not be tested under WSL2 |
| M7 | Research report | answers to the research questions (spec §11) with figures | ⏳ |
| M8 | AI-agent evaluation | effort log, rubric scores, optional comparison with a team implementation | ⏳ (protocol ready: `AI_AGENT_EVALUATION.md`) |

## Verification performed during development (macOS dev machine, no GPU)

| Check | Result |
|---|---|
| CPU + MPI build (CMake, MPICH 5, Clang, `-Wall -Wextra`) | builds with 0 warnings |
| `ctest` (correctness suite, MPI 2/4 ranks, every strategy end-to-end, pure-C sequential) | 12/12 pass |
| `sequential/seq_train` vs `pdl_train --strategy sequential` (MNIST, same options) | identical loss and accuracy every epoch |
| `pdl_check` gradcheck (4 nets incl. strided conv + overlapping pool) | max rel. error ≈ 1e-4 |
| MNIST LeNet sequential, 2 epochs | test acc 98.53% |
| MNIST 4 ranks: sync (mpi / ring+layer / overlap), psgd+reshuffle, admm, downpour | all converge (95.5–96.4% after 2 short epochs; downpour 90.6% with 3 workers) |
| Python pipeline on real runs (smoke, strong scaling, comm + perfmodel) | OK. Strong scaling 3.2× at np = 4; α-β model within 2.4% at np = 2 |
| CUDA backend | compile check on the dev machine (clang, sm_75); **executed and verified on the target** (RTX 2060): GEMM naive/tiled/cuBLAS, CNN kernels and the SGD update agree with the CPU reference |
| OpenMP kernels | not compiled on the dev machine (Apple Clang); **compiled and verified on the target** (GCC 13). Fixed there: a process-global `omp_set_num_threads` made later `cpu-omp` contexts single-threaded after `--verify` |

## First session on the target machine (checklist) — done 2026-11-23

On WSL2 export `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` first (apt's `nvidia-cuda-toolkit`
installs a `libcuda` that shadows the WSL driver library). `pdlbench` now stops with this hint when a
config needs the GPU and CUDA is not usable.

```bash
scripts/setup_ubuntu.sh && scripts/build.sh 2>&1 | tail     # expect CUDA: ON, OpenMP: ON in the summary
ctest --test-dir build --output-on-failure                  # must be all green, incl. check_cuda
build/bin/pdl_check --device both                           # CUDA kernels vs CPU
pdlbench data --dataset mnist cifar10
pdlbench all configs/smoke.yaml
pdlbench all configs/correctness.yaml
```
If anything fails, fix it before any measurement. Log the fix in the effort log
(`docs/AI_AGENT_EVALUATION.md` §4); it is research data for objective 7.

## Possible extensions

* CUDA-aware MPI (pass device pointers directly; skip the `transfer` phase).
* NCCL allreduce for multi-GPU nodes.
* Mixed precision (FP16 with loss scaling) and gradient compression (top-k, 1-bit SGD) as
  communication optimizations.
* Adagrad on the Downpour server and sharded parameter servers (Dean et al. 2012).
* Model parallelism (layer-wise pipeline across ranks).
* Multi-node runs (the code is plain MPI; only configs and the α-β calibration change).
