#!/usr/bin/env bash
# One-shot setup on the target machine: Python env, build, MNIST, smoke + correctness tests.
#
#   scripts/quickstart.sh            # assumes system packages are installed (scripts/setup_ubuntu.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "==> Creating .venv"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e .
fi
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/pytools${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Building"
scripts/build.sh

echo "==> Preparing MNIST"
PYTHON="$PY" scripts/prepare_data.sh mnist

echo "==> Smoke test"
"$PY" -m pdlbench run configs/smoke.yaml

echo "==> Correctness suite"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    "$PY" -m pdlbench all configs/correctness.yaml
else
    "$PY" -m pdlbench all configs/correctness.yaml --filter device=cpu
fi
echo "==> Quickstart finished. See results/smoke/latest and results/correctness/latest/report.md"
