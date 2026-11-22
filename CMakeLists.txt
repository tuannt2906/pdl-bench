# pdl-bench — Parallel & Distributed Deep Learning Benchmark
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)

project(pdl_bench VERSION 1.0.0 LANGUAGES C
        DESCRIPTION "Benchmark suite for parallel and distributed CNN training (C / MPI / OpenMP / CUDA)")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(PDL_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: ON, OFF or AUTO (if nvcc is found)")
set_property(CACHE PDL_ENABLE_CUDA PROPERTY STRINGS ON OFF AUTO)
option(PDL_ENABLE_CUBLAS "Use cuBLAS as an additional GEMM implementation (CUDA builds)" ON)
option(PDL_ENABLE_OPENMP "Build the OpenMP CPU kernels" ON)
option(PDL_NATIVE "Optimize for the build machine (-march=native)" ON)
option(PDL_BUILD_TESTS "Register CTest correctness tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS C)

set(PDL_HAVE_OPENMP OFF)
if(PDL_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    set(PDL_HAVE_OPENMP ON)
  else()
    message(WARNING "OpenMP not found: cpu-omp kernels will run single-threaded")
  endif()
endif()

set(PDL_HAVE_CUDA OFF)
if(PDL_ENABLE_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(PDL_HAVE_CUDA ON)
  else()
    message(STATUS "nvcc not found: building CPU-only (set -DPDL_ENABLE_CUDA=ON to require CUDA)")
  endif()
elseif(PDL_ENABLE_CUDA)
  set(PDL_HAVE_CUDA ON)
endif()

if(PDL_HAVE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)   # Turing — GeForce RTX 2060
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Build metadata (recorded in every result file)
# ---------------------------------------------------------------------------
set(PDL_GIT_COMMIT "unknown")
set(PDL_GIT_DIRTY 0)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE _sha
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND _sha)
    set(PDL_GIT_COMMIT "${_sha}")
  endif()
  execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE _st ERROR_QUIET)
  if(_st)
    set(PDL_GIT_DIRTY 1)
  endif()
endif()

set(PDL_C_FLAGS -Wall -Wextra -Wno-unused-parameter)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND PDL_C_FLAGS -O3)
  if(PDL_NATIVE)
    include(CheckCCompilerFlag)
    check_c_compiler_flag(-march=native PDL_HAS_MARCH_NATIVE)
    if(PDL_HAS_MARCH_NATIVE)
      list(APPEND PDL_C_FLAGS -march=native)
    endif()
  endif()
endif()
string(REPLACE ";" " " PDL_FLAGS_STRING "${PDL_C_FLAGS}")

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(PDL_SOURCES
  src/core/common.c
  src/core/timer.c
  src/core/rng.c
  src/core/json.c
  src/core/cli.c
  src/core/env.c
  src/backend/backend.c
  src/backend/cpu_kernels.c
  src/nn/model.c
  src/data/dataset.c
  src/comm/comm.c
  src/train/config.c
  src/train/trainer.c
  src/train/strategy_sync.c
  src/train/strategy_psgd.c
  src/train/strategy_admm.c
  src/train/strategy_downpour.c
)
if(PDL_HAVE_CUDA)
  list(APPEND PDL_SOURCES src/backend/cuda_kernels.cu)
endif()

add_library(pdl_core STATIC ${PDL_SOURCES})
target_include_directories(pdl_core PUBLIC ${CMAKE_SOURCE_DIR}/include PRIVATE ${CMAKE_SOURCE_DIR}/src/backend)
target_compile_options(pdl_core PRIVATE $<$<COMPILE_LANGUAGE:C>:${PDL_C_FLAGS}>)
target_compile_definitions(pdl_core PUBLIC
  PDL_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  PDL_BUILD_FLAGS="${PDL_FLAGS_STRING}"
  PDL_GIT_COMMIT="${PDL_GIT_COMMIT}"
  PDL_GIT_DIRTY=${PDL_GIT_DIRTY})
target_link_libraries(pdl_core PUBLIC MPI::MPI_C m)
if(PDL_HAVE_OPENMP)
  target_link_libraries(pdl_core PUBLIC OpenMP::OpenMP_C)
endif()
if(PDL_HAVE_CUDA)
  target_compile_definitions(pdl_core PUBLIC PDL_WITH_CUDA)
  target_compile_options(pdl_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
  # Programs are linked by the C compiler (MPI link flags are not nvcc-friendly);
  # the nvcc-compiled objects only need the CUDA runtime and the C++ runtime.
  target_link_libraries(pdl_core PUBLIC CUDA::cudart stdc++)
  if(PDL_ENABLE_CUBLAS)
    target_compile_definitions(pdl_core PUBLIC PDL_WITH_CUBLAS)
    target_link_libraries(pdl_core PUBLIC CUDA::cublas)
  endif()
  set_target_properties(pdl_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------
foreach(app pdl_train pdl_bench_compute pdl_bench_comm pdl_check)
  add_executable(${app} apps/${app}.c)
  target_compile_options(${app} PRIVATE ${PDL_C_FLAGS})
  target_link_libraries(${app} PRIVATE pdl_core)
endforeach()

# ---------------------------------------------------------------------------
# Sequential version: plain C, built from the shared CNN core only
# (no MPI, no OpenMP, no CUDA). Also buildable alone with sequential/Makefile.
# ---------------------------------------------------------------------------
add_executable(seq_train
  sequential/seq_train.c
  src/core/common.c src/core/timer.c src/core/rng.c src/core/json.c src/core/cli.c
  src/backend/backend.c src/backend/cpu_kernels.c
  src/nn/model.c src/data/dataset.c)
target_include_directories(seq_train PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/backend)
target_compile_options(seq_train PRIVATE ${PDL_C_FLAGS})
target_compile_definitions(seq_train PRIVATE PDL_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_link_libraries(seq_train PRIVATE m)

# ---------------------------------------------------------------------------
# Tests (CTest) — correctness suite + tiny end-to-end runs on synthetic data
# ---------------------------------------------------------------------------
if(PDL_BUILD_TESTS)
  enable_testing()
  include(${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
endif()

# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "pdl-bench ${PROJECT_VERSION} configuration")
message(STATUS "  build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C flags         : ${PDL_FLAGS_STRING}")
message(STATUS "  MPI             : ${MPI_C_LIBRARY_VERSION_STRING}")
message(STATUS "  OpenMP          : ${PDL_HAVE_OPENMP}")
message(STATUS "  CUDA            : ${PDL_HAVE_CUDA} ${CMAKE_CUDA_COMPILER_VERSION} (arch ${CMAKE_CUDA_ARCHITECTURES})")
message(STATUS "  cuBLAS          : ${PDL_ENABLE_CUBLAS}")
message(STATUS "  git commit      : ${PDL_GIT_COMMIT} (dirty=${PDL_GIT_DIRTY})")
message(STATUS "")
