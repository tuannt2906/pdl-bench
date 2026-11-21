# Setup on the target machine (i5-12600K + RTX 2060)

## Option A — native Ubuntu 22.04 / 24.04

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libopenmpi-dev openmpi-bin python3-venv
# NVIDIA driver (if not installed):   sudo ubuntu-drivers install   → reboot → nvidia-smi
# CUDA toolkit (recommended: NVIDIA repository, https://developer.nvidia.com/cuda-downloads)
sudo apt-get install -y cuda-toolkit            # after adding NVIDIA's apt repository
export PATH=/usr/local/cuda/bin:$PATH           # add to ~/.bashrc
```
Ubuntu's own `nvidia-cuda-toolkit` package also works (older CUDA, still supports sm_75).

## Option B — Windows 11 + WSL2

1. Install the latest NVIDIA Game Ready / Studio driver **on Windows**. Never install a Linux
   driver inside WSL.
2. `wsl --install -d Ubuntu-24.04`, then in Ubuntu: `nvidia-smi` must list the RTX 2060.
3. Install the toolkit from NVIDIA's "WSL-Ubuntu" repository (`cuda-toolkit` package, which
   ships no driver), then the packages of option A without the driver.
4. Clone the repository under `~/` (Linux filesystem), not `/mnt/c/...`.
5. Optional `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=24GB
   processors=16
   ```

## Build

```bash
scripts/setup_ubuntu.sh        # apt packages + .venv (asks for sudo)
scripts/build.sh               # = cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPDL_ENABLE_CUDA=auto ...
ctest --test-dir build --output-on-failure
```

| CMake option | Default | Meaning |
|---|---|---|
| `PDL_ENABLE_CUDA` | `AUTO` | `ON` requires nvcc, `OFF` builds CPU-only |
| `CMAKE_CUDA_ARCHITECTURES` | `75` | RTX 2060 = Turing sm_75 (`86` for RTX 30xx, `89` for 40xx) |
| `PDL_ENABLE_CUBLAS` | `ON` | adds the `cublas` GEMM variant |
| `PDL_ENABLE_OPENMP` | `ON` | `cpu-omp` kernels |
| `PDL_NATIVE` | `ON` | `-march=native` (AVX2 on Alder Lake) |
| `PDL_BUILD_TESTS` | `ON` | registers the CTest suite |

Expected `ctest` result: all tests pass. `check_cuda` appears only in CUDA builds.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `There are not enough slots available` (OpenMPI) | add `--oversubscribe` or `--use-hwthread-cpus`; the runner does this when `allow_oversubscribe: true` |
| `mpirun has detected an attempt to run as root` | add `--allow-run-as-root` (the runner adds it automatically as root) |
| nvcc: `unsupported GNU version` | `scripts/build.sh -- -DCMAKE_CUDA_HOST_COMPILER=g++-12` (install an older g++) |
| CMake can't find CUDA | `export PATH=/usr/local/cuda/bin:$PATH` or `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |
| `no CUDA device found` on WSL2 | update the Windows driver; check `ls /usr/lib/wsl/lib/libcuda.so*` and `nvidia-smi` |
| `cudaMalloc(...) failed` | reduce `--batch-size`, `--width` or the number of ranks sharing the GPU |
| Very slow multi-rank runs | `OMP_NUM_THREADS` × ranks > cores. Set `OMP_NUM_THREADS=1` or use `hybrid_mpi_omp.yaml` |
| OpenMPI TCP warnings on WSL2 | `export OMPI_MCA_btl_tcp_if_include=lo` or `export OMPI_MCA_btl=self,vader` |
| `cannot open dataset 'data/mnist-train.pdlds'` | `pdlbench data --dataset mnist` (or use `--train-data synthetic:...`) |
