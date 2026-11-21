#!/usr/bin/env bash
# Downloads MNIST and CIFAR-10 and converts them to data/*.pdlds.
#
#   scripts/prepare_data.sh            # mnist + cifar10
#   scripts/prepare_data.sh mnist      # only MNIST
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY=python3; fi
fi
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(mnist cifar10)

PYTHONPATH="$ROOT/pytools${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pdlbench data --dataset "${DATASETS[@]}" --out data
PYTHONPATH="$ROOT/pytools${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pdlbench data --info data/*.pdlds
