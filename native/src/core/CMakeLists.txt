# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
if(INGINE_BACKEND STREQUAL "CUDA")
  add_library(ingine_core STATIC
    arena.cu
    decode_graph.cpp
    device.cu
    dtype.cpp
    gguf.cpp
    hal_cuda.cpp
    host_memory.cpp
    paged_tier.cpp
    platos_key.cpp
    platos_key_cuda.cu
    safetensors_io.cpp
    safetensors_upload_cuda.cpp
    tensor.cpp
    weight_pack.cpp
  )
  ingine_cuda_target(ingine_core)
  target_compile_definitions(ingine_core PUBLIC INGINE_BACKEND_CUDA=1 INGINE_BACKEND_CPU=0)
  target_link_libraries(ingine_core PUBLIC
    CUDA::cudart CUDA::cublasLt CUDA::nvtx3 Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_cuda.cpp
  )
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
else()
  add_library(ingine_core STATIC
    dtype.cpp
    gguf.cpp
    hal_cpu.cpp
    platos_key.cpp
    safetensors_io.cpp
    safetensors_upload_cpu.cpp
    tensor.cpp
    weight_pack.cpp
  )
  target_compile_definitions(ingine_core PUBLIC INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1)
  target_link_libraries(ingine_core PUBLIC
    Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_cpu.cpp
  )
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
endif()

ingine_internal_includes(ingine_core)
target_include_directories(ingine_core PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
if(INGINE_ACT_FP16)
  target_compile_definitions(ingine_core PUBLIC INGINE_ACT_FP16=1)
else()
  target_compile_definitions(ingine_core PUBLIC INGINE_ACT_FP16=0)
endif()
if(INGINE_FORCE_SIMT_MMA)
  target_compile_definitions(ingine_core PUBLIC INGINE_FORCE_SIMT_MMA=1)
endif()
