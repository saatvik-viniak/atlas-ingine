add_library(ingine_model STATIC
  expert_cache.cpp
  kv_checkpoint.cpp
  paged_kv.cpp
  weights.cpp
  weights_gguf.cpp
  state.cpp
  decoder.cpp
  persona_runtime.cpp
  runtime.cpp
  generator.cpp
  batch_engine.cpp
  tp_engine.cpp
  model_kernels.cu
)
# prefill (package Prefill) and the speculative loop (package Spec)
foreach(src prefill.cpp drafter.cpp batch_drafter.cpp spec.cpp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    target_sources(ingine_model PRIVATE ${src})
  endif()
endforeach()
ingine_cuda_target(ingine_model)
ingine_internal_includes(ingine_model)
target_include_directories(ingine_model PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ingine_model PUBLIC
  ingine_core
  ingine_ops
  ingine_engine
  ingine_persona
  ingine_qwen38_cuda
  ingine_qwen4exp_cuda
  ingine_eg2_cuda
  ingine_kernels_dense
  ingine_kernels_moe
  ingine_kernels_attn
  ingine_kernels_head
  nlohmann_json::nlohmann_json
  CUDA::cudart)
if(INGINE_TP_SIZE GREATER 1)
  find_library(NCCL_LIB nccl REQUIRED)
  target_link_libraries(ingine_model PUBLIC ${NCCL_LIB})
endif()
foreach(area prefill draft batch)
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_model PUBLIC ingine_kernels_${area})
  endif()
endforeach()
# the vision tower (image input); its matrix products use the prefill GEMM
if(TARGET ingine_kernels_vision AND TARGET ingine_kernels_prefill AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vision.cpp)
  target_sources(ingine_model PRIVATE vision.cpp)
  target_link_libraries(ingine_model PUBLIC ingine_kernels_vision)
endif()
