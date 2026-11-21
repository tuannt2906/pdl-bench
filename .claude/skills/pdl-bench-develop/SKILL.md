---
name: pdl-bench-develop
description: Modify or extend the pdl-bench C/CUDA/MPI code base — add a layer, kernel variant, allreduce algorithm, training strategy, CLI option, result field or experiment config — while keeping the correctness invariants and the C↔Python result contract. Use for any code change in include/, src/, apps/, pytools/ or configs/.
---

# Developing pdl-bench

Read `CLAUDE.md` (invariants, gotchas) and the header of the module you touch
(`include/pdl/*.h`). Keep changes small and verify with the loop at the end.

## Recipes

### New CLI option for `pdl_train`
1. Add a field to `pdl_train_config` (`include/pdl/train.h`) and a default in
   `pdl_train_config_defaults` (`src/train/config.c`).
2. Add one row to `k_opts[]` (INT/LL/DBL/STR/ENUM). Parsing, `--help` and the JSON `config`
   object update automatically.
3. Document it in `docs/CLI.md`. If experiments use it, add it to `configs/*.yaml`.

### New layer type
1. `pdl_layer_kind` in `model.h`; parse the token in `pdl_model_create` (shapes, `w_count`,
   `b_count`, `flops_fwd`, name); allocate buffers.
2. Forward case in `forward_layers`, backward case in `pdl_model_backward` (write param grads at
   `m->grads + w_off`; write `dx` only when `i > 0`).
3. New kernels go in `backend.h` + `PDL_DECLARE_KERNELS` (`internal.h`) + `backend.c` dispatch +
   `cpu_kernels.c` + `cuda_kernels.cu`.
4. Add a `gc_case` to `test_gradcheck` and CUDA-vs-CPU coverage in `layer_pipeline`
   (`apps/pdl_check.c`). Update `pdl_layer_kind_name`.

### New kernel implementation variant (e.g. a faster GEMM)
Add to `pdl_impl` + `k_impl_names` + `pdl_impl_parse` (`backend.c`), implement it in the
device file, add it to the `resolve_impl` table in `apps/pdl_bench_compute.c`, and loop it in
`test_kernels`. CUDA: keep the `(ta, tb)` row-major semantics. cuBLAS is column-major, so
swap operands as `launch_gemm` does.

### New allreduce algorithm
Add an enum value + name in `comm.h`/`comm.c`, write `static void ar_<name>(...)`, and
dispatch it in `pdl_allreduce_sum`. Requirements: in-place, any `k` (also non powers of
two), `n` not divisible by `k`, **bitwise identical result on all ranks**, update
`pdl_comm_stats`. `pdl_check --tests allreduce,sync` and `pdl_bench_comm --algos <name>`
pick it up automatically. Add its cost formula to `docs/ALGORITHMS.md` and `predict_allreduce`
in `pytools/pdlbench/perfmodel.py`.

### New training strategy
1. `src/train/strategy_<name>.c` with `int pdl_run_<name>(pdl_trainer *t)`, built on the trainer
   helpers: `pdl_trainer_begin_epoch`, `pdl_trainer_epoch_order`, `pdl_trainer_batch`,
   `pdl_trainer_compute_gradients`, `pdl_trainer_apply_update`,
   `pdl_trainer_allreduce_device`, `pdl_trainer_evaluate`, `pdl_trainer_end_epoch`
   (exactly `epochs` records per rank), `pdl_trainer_report_epoch`.
2. Register it in `pdl_strategy` (`train.h`), `pdl_strategy_name`, `pdl_trainer_run`, the
   `k_strategies` choices (`config.c`), `CMakeLists.txt` sources, and `tests/CMakeLists.txt`
   (end-to-end test).
3. Time every region with phases. Exclude evaluation from `t->train_time`. Collectives must be
   reached by all ranks.
4. Document it in `docs/ALGORITHMS.md` and add it to `configs/strategies_convergence.yaml`.

### New result field
Write it in `pdl_trainer_write_results` (or the bench app), document it in
`docs/RESULTS_SCHEMA.md`, and read it in `pytools/pdlbench/aggregate.py` (tolerate absence:
old result files lack it).

### New experiment
Copy a config in `configs/`, adjust it, `pdlbench run <cfg> --dry-run`, then add it to
`scripts/run_all.sh` and the README table.

## Verification loop (run all of it before declaring done)
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
build/bin/pdl_check --device both            # if CUDA is available
mpirun -np 4 build/bin/pdl_check --tests allreduce,equivalence,sync
pdlbench all configs/smoke.yaml              # C ↔ Python contract still intact
python -m py_compile pytools/pdlbench/*.py
```
Expect 0 compiler warnings (`-Wall -Wextra`). If you change performance-critical code, rerun
the relevant `pdl_bench_compute`/`pdl_bench_comm` sweep before and after, and report both numbers.
