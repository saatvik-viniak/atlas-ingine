# EmbeddingGemma 2 (#968) Phase 1 tests: GemmaBpe tokenizer, config/manifest, CPU FP32 encoder,
# Matryoshka/prompt_name HTTP serving, and CUDA BF16/FP32 parity + batch invariance.
ingine_add_test(NAME eg2_cpu SOURCES test_eg2_cpu.cpp LIBS ingine_eg2_cpu ingine_serve LABELS eg2 TIMEOUT 600)
if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(NAME eg2_gpu SOURCES test_eg2_gpu.cu LIBS ingine_eg2_cuda ingine_serve ingine_model CUDA::cudart LABELS eg2 GPU MODEL TIMEOUT 600)
endif()
