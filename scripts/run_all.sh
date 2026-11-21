#!/usr/bin/env bash
# Runs the experiment suite: run + aggregate + plot + perfmodel + report per config,
# then a combined report in results/report.md.
#
#   scripts/run_all.sh                       # full suite (several hours)
#   scripts/run_all.sh --cpu-only            # skip / filter GPU parts
#   scripts/run_all.sh configs/strong_scaling.yaml configs/comm_allreduce.yaml
#   REPEATS=1 scripts/run_all.sh             # override repetitions (quick pass)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY=python3; fi
fi
export PYTHONPATH="$ROOT/pytools${PYTHONPATH:+:$PYTHONPATH}"

CPU_ONLY=0
CONFIGS=()
for a in "$@"; do
    case "$a" in
        --cpu-only) CPU_ONLY=1 ;;
        *) CONFIGS+=("$a") ;;
    esac
done
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=(
        configs/correctness.yaml
        configs/compute_gemm.yaml
        configs/compute_paper_cpu_gpu.yaml
        configs/compute_batch_model.yaml
        configs/comm_allreduce.yaml
        configs/strong_scaling.yaml
        configs/weak_scaling.yaml
        configs/hybrid_mpi_omp.yaml
        configs/batch_scaling.yaml
        configs/model_scaling.yaml
        configs/comm_optimization.yaml
        configs/strategies_convergence.yaml
        configs/gpu_mpi.yaml
        configs/paper_conv_model.yaml
    )
fi

# Per-config filters for machines without a GPU ("skip" = do not run).
cpu_filter() {
    case "$(basename "$1" .yaml)" in
        correctness|batch_scaling|gpu_mpi|paper_conv_model) echo "--filter device=cpu" ;;
        compute_batch_model) echo "--filter impls=cpu-omp" ;;
        compute_paper_cpu_gpu) echo "skip" ;;
        *) echo "" ;;
    esac
}

EXTRA=()
[[ -n "${REPEATS:-}" ]] && EXTRA+=(--repeats "$REPEATS")

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"
    args=()
    if [[ $CPU_ONLY -eq 1 ]]; then
        f="$(cpu_filter "$cfg")"
        if [[ "$f" == "skip" ]]; then echo "==> skipping $name (needs GPU)"; continue; fi
        [[ -n "$f" ]] && read -r -a args <<< "$f"
    fi
    # Training experiments use the communication benchmark for the alpha-beta model.
    if [[ -e results/comm_allreduce/latest && "$name" != "comm_allreduce" ]]; then
        args+=(--comm-dir results/comm_allreduce/latest)
    fi
    echo "==> $name"
    "$PY" -m pdlbench all "$cfg" "${args[@]}" "${EXTRA[@]}" || echo "!! $name finished with errors"
done

reports=()
for d in results/*/latest; do [[ -e "$d" ]] && reports+=("$d"); done
if [[ ${#reports[@]} -gt 0 ]]; then
    "$PY" -m pdlbench report "${reports[@]}" --out results/report.md
    echo "==> Combined report: results/report.md"
fi
