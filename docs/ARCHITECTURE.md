# Architecture

## Layers

```
 ┌──────────────────────────── apps/ ─────────────────────────────┐
 │ pdl_train   pdl_bench_compute   pdl_bench_comm   pdl_check      │  CLI programs, JSON results
 └───────┬───────────────┬──────────────────┬──────────────┬──────┘
         │               │                  │              │
 ┌───────▼──────── src/train/ ─────────┐    │              │
 │ config.c   table-driven options      │    │              │
 │ trainer.c  setup, step helpers,      │    │              │
 │            eval, checks, results     │    │              │
 │ strategy_sync.c     (sequential,sync)│    │              │
 │ strategy_psgd.c     (Parallel SGD)   │    │              │
 │ strategy_admm.c     (consensus ADMM) │    │              │
 │ strategy_downpour.c (param server)   │    │              │
 └──┬──────────┬───────────┬────────────┘    │              │
    │          │           │                 │              │
 ┌──▼───────┐ ┌▼─────────┐ ┌▼───────────────▼┐             │
 │ src/nn/  │ │ src/data/│ │ src/comm/       │             │
 │ model.c  │ │dataset.c │ │ comm.c          │             │
 │ layers,  │ │ .pdlds,  │ │ MPI init,       │             │
 │ fwd/bwd  │ │synthetic,│ │ ring / rd /     │             │
 └──┬───────┘ │partition │ │ tree / naive    │             │
    │         └──────────┘ └─────────────────┘             │
 ┌──▼──────────────────── src/backend/ ─────────────────────▼──┐
 │ backend.c      dispatch on ctx->device, memory, capabilities │
 │ cpu_kernels.c  naive / blocked / OpenMP                       │
 │ cuda_kernels.cu naive / tiled / cuBLAS + all CNN kernels      │
 └──────────────────────────────────────────────────────────────┘
 ┌──────────────────────── src/core/ ───────────────────────────┐
 │ common (log, fatal → MPI_Abort), timer (phases), rng          │
 │ (xoshiro256**), json writer, cli parser, env capture          │
 └──────────────────────────────────────────────────────────────┘

 pytools/pdlbench  (Python, outside the measured path)
   data → run → aggregate → plot → perfmodel → report ; ai-eval
```

The C code is the measured system; Python only prepares data, launches runs and analyses
the JSON files, so it never perturbs timings.

## A synchronous training step

```
rank r                                                         phase timer
──────                                                         ───────────
gather samples of slice r of the global batch (uint8 → float)  data
copy batch to device                                            h2d
forward: im2col → batched GEMM → bias → ReLU → pool → … → loss  forward
backward: dW = Σ dY·colᵀ, dcol = Wᵀ·dY, col2im, …               backward
[MPI_Barrier]   (wait time = load imbalance)                    sync_wait
[device → host copy of gradients]  (CUDA only)                  transfer
allreduce(grads, algo)                                          comm
[host → device copy]                                            transfer
grads /= k ; SGD update                                         update
```

## Key design decisions

| Decision | Rationale |
|---|---|
| **One flat parameter buffer** (and one gradient buffer); layers keep offsets | Gradient sync becomes a single collective over a contiguous array ("fused"); per-layer communication is a slice of it. Hashing and averaging models stay trivial. |
| **im2col + batched GEMM** for convolution | The paper (§3.1) analyses Caffe's approach; GEMM dominates, so GEMM variants show the CPU/GPU gap directly. `cublasSgemmStridedBatched` and the tiled kernel use `gridDim.z` for the batch. A direct convolution is kept as a baseline. |
| **Device abstraction by dispatch**, not by function-pointer tables | `pdl_ctx` selects device + variant; `backend.c` routes every kernel. Adding a backend means adding one file plus one `#ifdef` branch. |
| **MPI on host buffers** (GPU data staged through pinned memory) | Works with any MPI on WSL2/Ubuntu, and lets the `transfer` phase be measured separately. CUDA-aware MPI is a possible extension. |
| **Bitwise-identical collectives** | Every allreduce computes each element once (ring, tree, naive) or with a fixed operand order (recursive doubling), so replicas can be checked by hash. |
| **Deterministic kernels** | Gather-based `col2im` / max-pool backward (no atomics), fixed reduction trees: re-running with the same seed reproduces bits on the same device. |
| **Phase timers with optional device sync** | `--timing-sync 1` (default) synchronizes the GPU at phase boundaries, giving exact attribution at a small cost. Set it to 0 to measure pure throughput. |
| **Self-describing results** | Each JSON embeds the full config, environment (CPU, GPU, compiler, MPI, CUDA, git commit) and per-rank data. Analysis never depends on file names. |
| **Synthetic data keyed by (seed, index)** | Any shard can be generated independently on any rank (weak scaling, paper geometry) without a large download. |
| **Table-driven CLI** | One table produces parsing, `--help` and the `config` object of the results, so options can't go unrecorded. |
| **Python for orchestration only** | YAML experiments, statistics and plotting are easier to change than C, and they run outside the timed region. |

## Extension points

* **New layer**: add a `pdl_layer_kind`, parse it in `pdl_model_create`, implement the forward and
  backward cases in `model.c` (kernels in `backend.h` if needed), add a `gradcheck` case.
* **New kernel variant**: extend `pdl_impl`, implement it in `cpu_kernels.c` / `cuda_kernels.cu`,
  register its name in `backend.c` and `pdl_bench_compute.c`.
* **New collective**: add a `pdl_allreduce_algo` and its function in `comm.c`. It is automatically
  available in `--allreduce`, `pdl_bench_comm` and the `allreduce` / `sync` checks.
* **New strategy**: add `strategy_<name>.c` with `pdl_run_<name>(pdl_trainer*)` built on the
  trainer helpers, then register it in `train.h`, `trainer.c` (`pdl_trainer_run`) and `config.c`.

The project skill `.claude/skills/pdl-bench-develop` walks through these steps.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CUDA backend not executed during development (no GPU on the dev machine) | Full clang CUDA (sm_75) syntax/semantic check; `pdl_check --device cuda` compares every kernel with the CPU on first run on the target. |
| Hybrid P/E cores create stragglers | `sync_wait` and `load_imbalance` are measured; the docs recommend np ≤ 6 on P-cores as a comparison. |
| MPI + OpenMP oversubscription | Configs set `OMP_NUM_THREADS` and `--threads` explicitly; `hybrid_mpi_omp.yaml` keeps the core budget fixed. |
| Floating-point non-associativity between algorithms / rank counts | Equivalence tested with a 1e-4 tolerance. Replica identity within a run is exact. |
| Non-blocking allreduce does not progress without an MPI progress thread | `--overlap` is a measured option, not an assumption. Its benefit is a research question. |
| GPU memory (6 GB) shared by several ranks | Allocation failures print free/total memory and advice. Configs keep batch × width small for multi-rank GPU runs. |
| Timer resolution / noise | Warm-up steps, repeats with different seeds, 95% CIs, max-over-ranks for collectives. |
