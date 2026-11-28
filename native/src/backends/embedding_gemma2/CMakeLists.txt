add_library(ingine_eg2_cpu STATIC
  eg2_config.cpp
  eg2_manifest.cpp
  eg2_cpu.cpp
)

target_link_libraries(ingine_eg2_cpu PUBLIC
  ingine_core
  ingine_engine
  ingine_text
  nlohmann_json::nlohmann_json
)

ingine_internal_includes(ingine_eg2_cpu)
target_compile_options(ingine_eg2_cpu PRIVATE -Wall -Wextra -Wpedantic)

if(INGINE_BACKEND STREQUAL "CUDA")
  add_library(ingine_eg2_cuda STATIC
    eg2_cuda.cu
  )
  ingine_cuda_target(ingine_eg2_cuda)
  ingine_internal_includes(ingine_eg2_cuda)
  target_link_libraries(ingine_eg2_cuda PUBLIC
    ingine_eg2_cpu
    ingine_core
    CUDA::cudart
    CUDA::cublasLt
  )
endif()
