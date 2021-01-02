cmake_minimum_required (VERSION 3.8)

project (vector CXX CUDA)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR})

find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--fmad=true;--default-stream per-thread;-std=c++14;--expt-relaxed-constexpr)

include(ClangFormat)
clangformat_setup(
  sandbox/main.cu
  memory/vector.h
)

add_subdirectory(sandbox)
