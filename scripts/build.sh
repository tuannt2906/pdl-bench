#!/usr/bin/env bash
# Configures and builds pdl-bench with CMake.
#
#   scripts/build.sh [--cuda on|off|auto] [--type Release|Debug|RelWithDebInfo]
#                    [--arch 75] [--build-dir build] [-j N] [--clean] [-- <extra cmake args>]
#
# Defaults: --cuda auto (enabled when nvcc is found), --type Release, --arch 75 (RTX 2060).
# The full build log is written to <build-dir>/build.log (used by `pdlbench ai-eval`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA=auto
TYPE=Release
ARCH=75
BUILD_DIR="$ROOT/build"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CLEAN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA="$2"; shift 2 ;;
        --type) TYPE="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -j) JOBS="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$CUDA" == "auto" ]]; then
    if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then CUDA=on; else CUDA=off; fi
fi
case "$CUDA" in on|ON) CUDA=ON ;; off|OFF) CUDA=OFF ;; *) echo "--cuda must be on|off|auto" >&2; exit 2 ;; esac
if [[ "$CUDA" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
echo "==> Configuring (type=$TYPE, cuda=$CUDA, arch=sm_$ARCH) in $BUILD_DIR"
{
    cmake -S "$ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$TYPE" \
        -DPDL_ENABLE_CUDA="$CUDA" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        "${EXTRA[@]}"
    echo "==> Building with $JOBS jobs"
    cmake --build "$BUILD_DIR" -j "$JOBS"
} 2>&1 | tee "$BUILD_DIR/build.log"

echo "==> Binaries:"
ls -1 "$BUILD_DIR/bin" 2>/dev/null || true
