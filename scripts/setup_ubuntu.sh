#!/usr/bin/env bash
# Installs the system dependencies of pdl-bench on Ubuntu 22.04 / 24.04 (native or WSL2).
#
#   scripts/setup_ubuntu.sh            # apt packages + Python virtualenv (.venv)
#   scripts/setup_ubuntu.sh --no-venv  # apt packages only
#
# CUDA is NOT installed automatically (driver/toolkit choice is machine specific);
# the script checks for nvcc and prints instructions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAKE_VENV=1
[[ "${1:-}" == "--no-venv" ]] && MAKE_VENV=0

SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi

echo "==> Installing build tools, OpenMPI and Python"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libopenmpi-dev openmpi-bin \
    python3 python3-venv python3-pip \
    util-linux pciutils

if [[ $MAKE_VENV -eq 1 ]]; then
    echo "==> Creating Python virtualenv in $ROOT/.venv"
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install --upgrade pip
    "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
    "$ROOT/.venv/bin/pip" install -e "$ROOT"
fi

echo "==> Checking CUDA"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    echo "nvcc found in /usr/local/cuda/bin — add it to PATH:"
    echo '    export PATH=/usr/local/cuda/bin:$PATH'
    echo '    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
else
    cat <<'MSG'
nvcc not found. To build the CUDA backend (RTX 2060 = compute capability 7.5):
  * Quick (Ubuntu repository, older toolkit):  sudo apt-get install nvidia-cuda-toolkit
  * Recommended (current toolkit): follow https://developer.nvidia.com/cuda-downloads
    (Linux > x86_64 > Ubuntu > deb (network)), then: sudo apt-get install cuda-toolkit
  * WSL2: install the NVIDIA driver on WINDOWS only (never inside WSL), then install the
    "WSL-Ubuntu" cuda-toolkit package from the same page. Check with: nvidia-smi
Without CUDA, build with: scripts/build.sh --cuda off
MSG
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
    cat <<'MSG'
WSL2 detected:
  * Keep the repository inside the Linux filesystem (e.g. ~/pdl-bench), not /mnt/c — I/O is much faster.
  * OpenMPI may warn about the network interface; the runner adds no flags for that. If needed:
        export OMPI_MCA_btl_tcp_if_include=lo
  * Give WSL enough memory/cores in %UserProfile%\.wslconfig ([wsl2] memory=24GB processors=16).
MSG
fi
echo "==> Done. Next: scripts/build.sh && scripts/prepare_data.sh"
