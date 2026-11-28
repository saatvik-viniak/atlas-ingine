# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
cmake_minimum_required(VERSION 3.28)

set(INGINE_BACKEND "CUDA" CACHE STRING "Compute backend (CUDA, CPU)")
set_property(CACHE INGINE_BACKEND PROPERTY STRINGS CUDA CPU)
if(NOT INGINE_BACKEND STREQUAL "CUDA" AND NOT INGINE_BACKEND STREQUAL "CPU")
  message(FATAL_ERROR "Unsupported INGINE_BACKEND='${INGINE_BACKEND}' (First Slice supports CUDA or CPU)")
endif()

if(INGINE_BACKEND STREQUAL "CUDA")
  # Multi-arch build: supports Ampere (86), Ada (89), Hopper (90a), Blackwell (120a), and DGX Spark GB10 (121a).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86;89;120a" CACHE STRING "CUDA architectures to build")
  endif()

  # Prefer the installer's private toolkit, then /opt/cuda, when nvcc is not chosen explicitly.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    foreach(_nvcc ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/nvidia/cu13/bin/nvcc
                  ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/bin/nvcc /opt/cuda/bin/nvcc)
      if(EXISTS ${_nvcc})
        set(CMAKE_CUDA_COMPILER ${_nvcc} CACHE FILEPATH "CUDA compiler")
        break()
      endif()
    endforeach()
  endif()

  project(ingine LANGUAGES C CXX CUDA)
else()
  project(ingine LANGUAGES C CXX)
endif()

option(INGINE_BUILD_TESTS "Build ingine tests" OFF)
option(INGINE_BUILD_APPS "Build the ingine executables (serve, bench, validate)" ON)
option(INGINE_ACT_FP16 "Use FP16 activations/weights instead of BF16 (required for pre-Ampere sm_60/70/75)" OFF)
option(INGINE_FORCE_SIMT_MMA "Force FP32-accumulated SIMT MMA fallback on all archs (to oracle-verify Pascal sm_60/61 path on V100)" OFF)
set(INGINE_TP_SIZE "1" CACHE STRING "Compile-time Tensor+Expert Parallelism world size (1, 2, or 4)")
if(NOT INGINE_TP_SIZE MATCHES "^(1|2|4)$")
  message(FATAL_ERROR "INGINE_TP_SIZE must be 1, 2, or 4 (got '${INGINE_TP_SIZE}')")
endif()
if(INGINE_TP_SIZE GREATER 1 AND NOT INGINE_BACKEND STREQUAL "CUDA")
  message(FATAL_ERROR "INGINE_TP_SIZE=${INGINE_TP_SIZE} requires INGINE_BACKEND=CUDA")
endif()
add_compile_definitions(INGINE_TP_SIZE=${INGINE_TP_SIZE})

if(INGINE_BACKEND STREQUAL "CUDA")
  # Detect pre-Ampere (< sm_80) and Ampere+ (>= sm_80) architectures.
  set(_ingine_has_pre_sm80 OFF)
  set(_ingine_has_sm80_plus OFF)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
    if(_arch_num)
      if(_arch_num LESS 80)
        set(_ingine_has_pre_sm80 ON)
      else()
        set(_ingine_has_sm80_plus ON)
      endif()
    endif()
  endforeach()
  if(_ingine_has_pre_sm80 AND _ingine_has_sm80_plus AND NOT INGINE_ACT_FP16)
    message(FATAL_ERROR
      "Mixed pre-Ampere (<80) and Ampere+ (>=80) CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' requires -DINGINE_ACT_FP16=ON explicitly (ActT is compile-time).")
  endif()
  if(_ingine_has_pre_sm80 AND NOT INGINE_ACT_FP16)
    set(INGINE_ACT_FP16 ON CACHE BOOL "Use FP16 activations/weights instead of BF16 (required for pre-Ampere sm_60/70/75)" FORCE)
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(INGINE_BACKEND STREQUAL "CUDA")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # no C++20 modules; skip the dependency scan
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ingine_link=1)
  set(CMAKE_JOB_POOL_LINK ingine_link)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(third_party)
add_subdirectory(src/core)

# Every other component directory owns its CMakeLists.txt and defines the target named here; a
# directory is added only once it has one, so the tree builds while components are in progress.
#   src/text              -> ingine_text
#   src/image             -> ingine_image
#   src/kernels/<area>    -> ingine_kernels_<area>   (aggregated by ingine_kernels)
#   src/model             -> ingine_model
#   src/engine            -> ingine_engine
#   src/serve             -> ingine_serve
#   apps/serve|bench|validate -> ingine-serve | ingine-bench | ingine-validate
function(ingine_add_component dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

ingine_add_component(src/text)
ingine_add_component(src/image)
ingine_add_component(src/persona)
ingine_add_component(src/engine)
ingine_add_component(src/backends/qwen38)
ingine_add_component(src/backends/qwen4exp)
ingine_add_component(src/backends/embedding_gemma2)

if(INGINE_BACKEND STREQUAL "CUDA")
  set(INGINE_KERNEL_AREAS common dense moe attn head prefill draft vision batch)
  foreach(area ${INGINE_KERNEL_AREAS})
    ingine_add_component(src/kernels/${area})
  endforeach()
  add_library(ingine_kernels INTERFACE)
  foreach(area ${INGINE_KERNEL_AREAS})
    if(TARGET ingine_kernels_${area})
      target_link_libraries(ingine_kernels INTERFACE ingine_kernels_${area})
    endif()
  endforeach()
  if(TARGET ingine_ops AND TARGET ingine_kernels_dense)
    target_link_libraries(ingine_ops PUBLIC ingine_kernels_dense CUDA::cudart)
  endif()

  ingine_add_component(src/model)
endif()
ingine_add_component(src/serve)

if(INGINE_BUILD_APPS)
  foreach(app serve bench validate persona_probe pack gen_ids)
    ingine_add_component(apps/${app})
  endforeach()
endif()

if(INGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
