# Changelog

All notable changes to this project are documented here (format: Keep a Changelog, SemVer).

## [1.0.0] — 2026-11-21
Initial version, generated with Claude Code from the specification (`docs/reference/overview.md`)
and the reference paper. This is the `ai-baseline` subject of `docs/AI_AGENT_EVALUATION.md`.

### Added
- CNN implementation in C: conv (im2col + GEMM), dense, ReLU, max-pool, softmax cross-entropy,
  SGD with momentum / weight decay; model presets and spec strings.
- Backends: CPU naive / blocked / OpenMP, CUDA naive / tiled / cuBLAS, direct convolution.
- Strategies: sequential, synchronous SGD (flat / per-layer / overlapped), Parallel SGD with
  ShuffleSGD, consensus ADMM, Downpour SGD with a parameter server.
- Allreduce algorithms: MPI, ring, recursive doubling, binomial tree, naive, reduce+bcast.
- Programs: `pdl_train`, `pdl_bench_compute`, `pdl_bench_comm`, `pdl_check`; CTest suite.
- Python tooling `pdlbench`: datasets, YAML experiment runner, aggregation, plots, α-β
  performance model, reports, AI-agent code metrics; 15 experiment configs; setup scripts.
- Documentation, `CLAUDE.md` and Claude Code project skills.
