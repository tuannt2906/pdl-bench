# Evaluating an AI coding agent on HPC code

This document is the protocol for research objective 7 of the project specification
(`docs/reference/overview.md` §6): *how well can an AI coding agent (Claude Code)
build a correct, efficient MPI/CUDA deep-learning benchmark?*

**This repository is itself the AI-generated subject.** It was produced by Claude Code
from the paper (https://hgpu.org/?p=16046) and the
specification, then built and validated on the target machine. Every human change made
afterwards must be logged (see [effort log](#4-human-effort-log)) so that the effort needed
to reach a working implementation can be measured.

---

## 1. Subjects

| ID | Implementation | Origin |
|---|---|---|
| **A** | this repository at the tag/commit produced by the agent (`ai-baseline`) | Claude Code |
| **A+** | this repository after human fixes (`main`) | Claude Code + human fixes |
| **T** | team implementation of the same specification (optional, separate tree) | human team |

Tag the untouched agent output before changing anything:

```bash
git tag ai-baseline <first-commit>
```

## 2. Unified specification

All subjects implement the same contract, which makes them directly comparable:

* CLI options — [docs/CLI.md](CLI.md)
* Result files — [docs/RESULTS_SCHEMA.md](RESULTS_SCHEMA.md)
* Experiments — `configs/*.yaml` (run unchanged against each subject's `build/bin`)

A team implementation only needs to produce binaries with the same names and options and
emit JSON following the schema; the Python tooling then benchmarks every subject identically:

```bash
python -m pdlbench run configs/strong_scaling.yaml --build-dir /path/to/team/build --results-dir results/team
python -m pdlbench run configs/strong_scaling.yaml --build-dir build             --results-dir results/ai
```

## 3. Evaluation dimensions and scoring rubric

Each dimension is scored 0–4. Evidence column = the artefact that justifies the score.

| # | Dimension | 0 | 1 | 2 | 3 | 4 | Evidence |
|---|---|---|---|---|---|---|---|
| D1 | **Build** | does not build | builds after major manual fixes | builds after minor fixes | builds, warnings | builds cleanly with `-Wall -Wextra` | `build/build.log`, `pdlbench ai-eval --build-log` |
| D2 | **Correctness (numerics)** | gradients wrong | gradcheck fails for some layers | gradcheck passes, CPU/GPU disagree | all `pdl_check` tests pass | passes + trains to expected accuracy (MNIST ≥ 98%) | `configs/correctness.yaml`, `strategies_convergence` |
| D3 | **MPI capability** | no MPI | only library collectives, deadlocks | works for some `np` | all strategies run for np 1–10 | + custom collectives correct for non-power-of-2 `np` | `pdl_check --tests allreduce`, `comm_allreduce` |
| D4 | **Synchronization & communication** | replicas diverge | sync only with barriers everywhere | correct sync SGD | + correct async Downpour (no deadlock, bounded staleness) | + overlap / fusion options with measurable effect | `--check-sync`, server staleness stats, `comm_optimization` |
| D5 | **Runtime performance** | > 10× slower than reference | 3–10× | 1.5–3× | within 1.5× | at or better than reference | `compute_gemm`, `batch_scaling` vs. team / library baseline |
| D6 | **Scalability** | slowdown with np | speedup < 2 at np = 8 | efficiency ≥ 0.4 at np = 8 | ≥ 0.6 | ≥ 0.75 and matches perf model within 25% | `strong_scaling`, `weak_scaling`, `perfmodel` |
| D7 | **Code quality & maintainability** | unreadable | monolithic, no structure | modular but inconsistent | modular, documented, consistent naming | + low complexity, no dead code, clear extension points | `pdlbench ai-eval`, manual review checklist (§5) |
| D8 | **Reproducibility** | no seeds/config capture | seeds only | seeds + config | + environment capture | + bitwise-repeatable runs (`determinism` test) | result JSON `environment`, `pdl_check --tests determinism` |
| D9 | **Human effort to completion** | rewrite needed | > 3 days | 1–3 days | < 1 day | < 2 hours | effort log (§4) |

Report the scores per subject in a table and the total (max 36). Always attach the evidence.

## 4. Human effort log

Record **every** intervention after the agent's output, one row per issue, in
`docs/ai_effort_log.csv` (create it on first use):

```csv
date,subject,category,severity,component,symptom,root_cause,fix_summary,time_min,prompts,lines_changed,found_by
2026-11-25,A,build,minor,src/backend/cuda_kernels.cu,"nvcc error: ...","missing cast","added (float) cast",10,1,2,compiler
```

| Field | Values |
|---|---|
| `category` | `build`, `numerics`, `mpi-deadlock`, `mpi-semantics`, `sync`, `cuda`, `performance`, `memory`, `ux/docs`, `tooling` |
| `severity` | `blocker` (no result possible), `major` (wrong results), `minor` (cosmetic / perf) |
| `found_by` | `compiler`, `pdl_check`, `smoke`, `experiment`, `review`, `user` |
| `prompts` | number of agent prompts used for the fix (0 = fixed by hand) |

Summary metrics derived from the log: total time, prompts, lines changed, issues per
category, *first-time-right* rate = components with zero issues / total components.

## 5. Manual review checklist (D7)

- [ ] Module boundaries match `docs/` (backend / nn / data / comm / train / apps).
- [ ] No hidden global state besides logging; every buffer has an owner and is freed.
- [ ] MPI: every send has a matching receive on all code paths; tags documented; no
      reliance on eager-protocol buffering for large messages.
- [ ] Collectives are called by all ranks in the same order (no rank-dependent branches around them).
- [ ] CUDA: every API call checked; no implicit synchronisation inside timed regions except
      where documented (`--timing-sync`).
- [ ] Error paths abort the whole MPI job instead of hanging.
- [ ] Hot loops are free of allocations; memory reused across steps.
- [ ] Naming, formatting (`.clang-format`) and comments consistent.

## 6. Procedure

1. **Freeze** the agent output (`git tag ai-baseline`).
2. **Build** each subject: `scripts/build.sh 2>&1` (log kept in `build/build.log`).
3. **Static metrics**:
   ```bash
   python -m pdlbench ai-eval <src_A> <src_T> --names AI Team \
       --build-log <A>/build/build.log <T>/build/build.log --out results/ai_eval
   ```
   (`pip install lizard` adds cyclomatic complexity.)
4. **Correctness**: `python -m pdlbench all configs/correctness.yaml` for each subject.
5. **Performance & scalability**: run the same configs against every subject
   (`--build-dir`), at least 3 repetitions, same machine, idle system, fixed CPU governor
   (`sudo cpupower frequency-set -g performance` on native Linux).
6. **Compare** with `python -m pdlbench report results/ai/<exp>/latest results/team/<exp>/latest`.
7. **Score** D1–D9 with evidence; fill the effort summary.
8. **Discuss** failure patterns (which categories the agent got wrong first time, which it
   fixed itself when shown the error, which needed human insight).

## 7. Threats to validity

* Single machine: MPI runs use shared memory; network effects of real clusters are absent.
* Hybrid CPU (P-cores + E-cores) introduces load imbalance unrelated to the implementation;
  pin ranks (`--bind-to core`) and report the core mapping.
* Agent output depends on prompt, model version and session; record them in the effort log
  header (model ID, date, prompt text or link).
* The team implementation must not copy agent code (and vice versa) to keep the comparison fair.
