# Server tests (CPU only). serve_scheduler: the scheduler with a multi-token-per-round generator.
# serve_server: the HTTP API end to end against FakeGenerator (needs the tokenizer and chat template of
# INGINE_MODEL_DIR, skipped otherwise).
ingine_add_test(NAME serve_scheduler SOURCES test_scheduler.cpp LIBS ingine_engine LABELS serve TIMEOUT 60)
ingine_add_test(NAME serve_server SOURCES test_server.cpp LIBS ingine_serve LABELS serve TIMEOUT 300)
ingine_add_test(NAME serve_embeddings SOURCES test_embeddings.cpp LIBS ingine_serve LABELS serve TIMEOUT 120)
ingine_add_test(NAME serve_cache_baseline SOURCES test_cache_baseline.cpp LIBS ingine_serve LABELS serve TIMEOUT 300)
if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(NAME serve_gpu_warm_start SOURCES test_gpu_warm_start.cpp LIBS ingine_serve ingine_model CUDA::cudart LABELS serve GPU MODEL TIMEOUT 600)
  ingine_add_test(NAME serve_gpu_embeddings SOURCES test_gpu_embeddings.cpp LIBS ingine_serve ingine_model CUDA::cudart LABELS serve GPU MODEL TIMEOUT 600)
endif()
