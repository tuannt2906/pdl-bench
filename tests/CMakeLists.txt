# pdl-bench — CTest registration (included from the top-level CMakeLists.txt).
# SPDX-License-Identifier: MIT
#
# The correctness logic lives in apps/pdl_check.c; these tests run it on the
# CPU (and CUDA when built), under MPI with 2 and 4 ranks, and run every
# training strategy end-to-end on a tiny synthetic dataset.
#
#   ctest --test-dir build --output-on-failure            # everything
#   ctest --test-dir build -L quick                        # no MPI launches

set(_bin ${CMAKE_BINARY_DIR}/bin)

# OpenMPI refuses to start more ranks than cores (and as root) without flags.
set(_mpi_pre ${MPIEXEC_PREFLAGS})
if(MPI_C_LIBRARY_VERSION_STRING MATCHES "Open MPI")
  list(APPEND _mpi_pre --oversubscribe)
endif()

function(pdl_mpi_test name np)
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_mpi_pre} ${ARGN} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${name} PROPERTIES LABELS "mpi" TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction()

# --- correctness suite -------------------------------------------------------
add_test(NAME check_cpu COMMAND ${_bin}/pdl_check --tests gradcheck,kernels,determinism,finite --device cpu --verbose 0)
set_tests_properties(check_cpu PROPERTIES LABELS "quick" TIMEOUT 300)
if(PDL_HAVE_CUDA)
  add_test(NAME check_cuda COMMAND ${_bin}/pdl_check --tests gradcheck,kernels,determinism,finite --device cuda --verbose 0)
  set_tests_properties(check_cuda PROPERTIES LABELS "quick;cuda" TIMEOUT 300)
endif()
pdl_mpi_test(check_mpi_2 2 ${_bin}/pdl_check --tests allreduce,equivalence,sync --verbose 0)
pdl_mpi_test(check_mpi_4 4 ${_bin}/pdl_check --tests allreduce,equivalence,sync --verbose 0)

# --- end-to-end training on synthetic data ------------------------------------
set(_tiny --train-data synthetic:1024:1:16:16:4 --test-data synthetic:256:1:16:16:4
          --model conv:4:3:1:1,relu,pool:2:2,fc:16,relu,fc:4 --epochs 2 --batch-size 16 --lr 0.02 --verbose 0)
set(_tiny_seq --train-data synthetic:1024:1:16:16:4 --test-data synthetic:256:1:16:16:4
              --model conv:4:3:1:1,relu,pool:2:2,fc:16,relu,fc:4 --epochs 2 --batch-size 16 --lr 0.02)
add_test(NAME seq_train_pure_c COMMAND ${_bin}/seq_train ${_tiny_seq})
set_tests_properties(seq_train_pure_c PROPERTIES LABELS "quick" TIMEOUT 120)
add_test(NAME train_sequential COMMAND ${_bin}/pdl_train --strategy sequential ${_tiny})
set_tests_properties(train_sequential PROPERTIES LABELS "quick" TIMEOUT 120)
pdl_mpi_test(train_sync_ring 3 ${_bin}/pdl_train --strategy sync --allreduce ring --check-sync 5 ${_tiny})
pdl_mpi_test(train_sync_overlap 2 ${_bin}/pdl_train --strategy sync --overlap 1 --fusion layer --check-sync 5 ${_tiny})
pdl_mpi_test(train_psgd_reshuffle 2 ${_bin}/pdl_train --strategy psgd --avg-period 4 --reshuffle 1 ${_tiny})
pdl_mpi_test(train_admm 2 ${_bin}/pdl_train --strategy admm --admm-local-steps 4 ${_tiny})
pdl_mpi_test(train_downpour 3 ${_bin}/pdl_train --strategy downpour --push-every 2 --fetch-every 2 ${_tiny})
pdl_mpi_test(bench_comm_small 3 ${_bin}/pdl_bench_comm --max-bytes 65536 --reps 3 --warmup 1)
add_test(NAME bench_compute_small COMMAND ${_bin}/pdl_bench_compute --op gemm,conv,train-step
         --impls cpu-blocked --sizes 32,64 --batches 2 --conv 1:32:32:8:5:2:0 --reps 2 --warmup 1)
set_tests_properties(bench_compute_small PROPERTIES LABELS "quick" TIMEOUT 120)
