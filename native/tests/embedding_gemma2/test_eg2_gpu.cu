// GPU parity, batch-invariance, sliding-window boundary, and live HTTP smoke test for
// google/embeddinggemma-2 (#968) on both sm_80+ (BF16) and sm_70 V100 (native FP32, zero FP16).
#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_cpu.h"
#include "backends/embedding_gemma2/eg2_cuda.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "core/device.h"
#include "ingine/generator.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using namespace ingine::backends::embedding_gemma2;

int g_failures = 0;

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                 \
      throw std::runtime_error("check failed");                                     \
    }                                                                               \
  } while (0)

struct ParityStats {
  double cosine = 0.0;
  double max_abs = 0.0;
};

ParityStats compare_vecs(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size() && !a.empty());
  double dot = 0.0, na = 0.0, nb = 0.0, ma = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
    ma = std::max(ma, std::abs(x - y));
  }
  return ParityStats{dot / (std::sqrt(na) * std::sqrt(nb)), ma};
}

double dot_f32(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return s;
}

double l2_norm(const std::vector<float>& v) { return std::sqrt(dot_f32(v, v)); }

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir = "/mnt/storage/models/embeddinggemma-2";
  std::filesystem::path out_json;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_json = argv[++i];
  }
  if (const char* env = std::getenv("INGINE_EG2_MODEL_DIR"); env && *env && argc == 1) {
    model_dir = env;
  }
  if (!std::filesystem::is_regular_file(model_dir / "model.safetensors")) {
    std::printf("[SKIP] EmbeddingGemma-2 checkpoint not found at %s\n", model_dir.c_str());
    return 77;
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("[SKIP] no CUDA GPU available\n");
    return 77;
  }

  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sm = prop.major * 10 + prop.minor;

    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 8192;
    opts.vision = false;
    opts.audio = false;

    const auto t_load0 = std::chrono::steady_clock::now();
    auto gen = ingine::model::make_generator(opts);
    const double gpu_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();

    auto* gpu_backend = dynamic_cast<Eg2CudaBackend*>(gen.get());
    CHECK(gpu_backend != nullptr);
    const bool is_bf16 = gpu_backend->uses_bf16();
    const auto residency = gpu_backend->residency_report();
    const double weight_mib = static_cast<double>(residency.weights_device_bytes) / (1024.0 * 1024.0);
    std::printf("[GPU] device=%s sm_%d mode=%s weights=%.2f MiB load=%.3fs\n", prop.name, sm,
                is_bf16 ? "BF16(FP32-acc)" : "FP32(zero-FP16)", weight_mib, gpu_load_s);

    // Selective loading check: text-only must stay ~517 MiB (BF16) or ~1034 MiB (FP32) — never 1420+ MiB!
    if (is_bf16) {
      CHECK(weight_mib > 500.0 && weight_mib < 540.0);
    } else {
      CHECK(weight_mib > 1000.0 && weight_mib < 1080.0);
    }

    const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const auto& prompts = gpu_backend->config().prompts;

    // Reference CPU FP32 backend for exact GPU-vs-CPU parity verification
    Eg2CpuBackend cpu_backend(opts);

    const std::vector<std::pair<std::string, std::string>> test_pairs = {
        {"SearchQuery", "What causes the northern lights?"},
        {"Document", "The northern lights are caused by charged particles from the sun."},
        {"Document", "A recipe for chocolate chip cookies uses butter, sugar, and flour."},
        {"CodeRetrieval", "binary search over sorted array in C++ returning lower_bound"},
        {"Document", "template <class T> size_t lower_bound_idx(const std::vector<T>& a, const T& x) { ... }"},
        {"FactChecking", "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
        {"Classification", "Quarterly revenue grew 18 percent year over year on strong cloud demand."},
        {"SentenceSimilarity", "The quick brown fox jumps over the lazy dog near the river bank."},
    };

    std::vector<std::vector<std::int32_t>> batch_ids;
    for (const auto& [pn, txt] : test_pairs) {
      batch_ids.push_back(tok.encode_with_post_processor(prompts.at(pn) + txt, false));
    }

    // 1) GPU vs CPU FP32 parity across all 4 Matryoshka dims (768, 512, 256, 128)
    const double min_cos = is_bf16 ? 0.9990 : 0.999995;
    Json mrl_report = Json::object();
    for (int dim : {768, 512, 256, 128}) {
      ingine::model::EmbeddingRequest req;
      req.inputs = {batch_ids[0], batch_ids[1], batch_ids[2]};
      req.dimensions = dim;
      req.normalize = true;
      const auto gres = gpu_backend->embed(req);
      const auto cres = cpu_backend.embed(req);
      CHECK(gres.stats.output_dim == dim);
      CHECK(gres.stats.kv_slots_allocated == 0);
      CHECK(gres.stats.lm_head_skipped);

      double worst_cos = 1.0, worst_abs = 0.0;
      for (int i = 0; i < 3; ++i) {
        CHECK(std::abs(l2_norm(gres.embeddings[i]) - 1.0) < 1e-4);
        const auto ps = compare_vecs(gres.embeddings[i], cres.embeddings[i]);
        worst_cos = std::min(worst_cos, ps.cosine);
        worst_abs = std::max(worst_abs, ps.max_abs);
      }
      const double cm = dot_f32(gres.embeddings[0], gres.embeddings[1]);
      const double cu = dot_f32(gres.embeddings[0], gres.embeddings[2]);
      std::printf("[PARITY dim=%d] gpu_vs_cpu_min_cos=%.7f max_abs=%.6f cos(q,m)=%.6f cos(q,u)=%.6f margin=%+.6f\n",
                  dim, worst_cos, worst_abs, cm, cu, cm - cu);
      CHECK(worst_cos >= min_cos);
      CHECK(cm - cu > 0.20);
      mrl_report[std::to_string(dim)] = {
          {"gpu_vs_cpu_min_cos", worst_cos},
          {"gpu_vs_cpu_max_abs", worst_abs},
          {"cos_query_match", cm},
          {"cos_query_unrelated", cu},
          {"margin", cm - cu},
      };
    }

    // 2) Early-exit tap_layer=11 GPU vs CPU parity
    {
      ingine::model::EmbeddingRequest req;
      req.inputs = {batch_ids[0], batch_ids[1]};
      req.tap_layer = 11;
      req.normalize = true;
      const auto gres = gpu_backend->embed(req);
      const auto cres = cpu_backend.embed(req);
      CHECK(gres.stats.output_dim == 512);
      CHECK(gres.stats.effective_tap_layer == 11);
      const auto ps0 = compare_vecs(gres.embeddings[0], cres.embeddings[0]);
      const auto ps1 = compare_vecs(gres.embeddings[1], cres.embeddings[1]);
      std::printf("[PARITY tap_layer=11] cos0=%.7f cos1=%.7f\n", ps0.cosine, ps1.cosine);
      CHECK(std::min(ps0.cosine, ps1.cosine) >= min_cos);
    }

    // 3) Packed ragged batch invariance: B=1 vs B=8
    double batch_min_cos = 1.0, batch_max_abs = 0.0;
    {
      ingine::model::EmbeddingRequest r8;
      r8.inputs = batch_ids;
      r8.normalize = true;
      const auto res8 = gpu_backend->embed(r8);
      for (std::size_t i = 0; i < batch_ids.size(); ++i) {
        ingine::model::EmbeddingRequest r1;
        r1.inputs = {batch_ids[i]};
        r1.normalize = true;
        const auto res1 = gpu_backend->embed(r1);
        const auto ps = compare_vecs(res8.embeddings[i], res1.embeddings[0]);
        batch_min_cos = std::min(batch_min_cos, ps.cosine);
        batch_max_abs = std::max(batch_max_abs, ps.max_abs);
      }
      std::printf("[BATCH INVARIANCE B=1 vs B=8] min_cos=%.8f max_abs=%.7f\n", batch_min_cos, batch_max_abs);
      const double batch_cos_thresh = is_bf16 ? 0.9995 : 0.999995;
      CHECK(batch_min_cos >= batch_cos_thresh);
    }

    // 4) Long-sequence sliding-window boundary parity (T = 640 > sliding_window=512)
    double swa_cos = 0.0, swa_abs = 0.0;
    {
      std::string long_text = prompts.at("Document");
      for (int k = 0; k < 48; ++k) {
        long_text += "The northern lights are caused by charged particles from the sun entering Earth's atmosphere. ";
      }
      auto long_ids = tok.encode_with_post_processor(long_text, false);
      CHECK(long_ids.size() > 600);
      ingine::model::EmbeddingRequest rl;
      rl.inputs = {long_ids};
      rl.normalize = true;
      const auto gres = gpu_backend->embed(rl);
      const auto cres = cpu_backend.embed(rl);
      const auto ps = compare_vecs(gres.embeddings[0], cres.embeddings[0]);
      swa_cos = ps.cosine;
      swa_abs = ps.max_abs;
      std::printf("[SLIDING WINDOW T=%zu > 512] gpu_vs_cpu_cos=%.7f max_abs=%.6f\n", long_ids.size(), swa_cos, swa_abs);
      CHECK(swa_cos >= min_cos);
    }

    // 5) Throughput benchmark (B=1, B=8, B=32 warm iterations)
    auto bench_batch = [&](int B, int iters) -> Json {
      ingine::model::EmbeddingRequest rb;
      rb.normalize = true;
      int toks = 0;
      for (int i = 0; i < B; ++i) {
        rb.inputs.push_back(batch_ids[static_cast<std::size_t>(i) % batch_ids.size()]);
        toks += static_cast<int>(rb.inputs.back().size());
      }
      // Warmup
      (void)gpu_backend->embed(rb);
      const auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) (void)gpu_backend->embed(rb);
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double ms_per_batch = (elapsed / iters) * 1e3;
      const double tok_s = (static_cast<double>(toks) * iters) / elapsed;
      const double seq_s = (static_cast<double>(B) * iters) / elapsed;
      std::printf("[BENCH B=%d toks=%d] %.3f ms/batch | %.1f seq/s | %.1f tok/s\n", B, toks, ms_per_batch, seq_s, tok_s);
      return Json{{"batch", B}, {"tokens", toks}, {"ms_per_batch", ms_per_batch}, {"seq_per_s", seq_s}, {"tok_per_s", tok_s}};
    };
    Json bench_j = Json::array({bench_batch(1, 20), bench_batch(8, 20), bench_batch(32, 10)});

    // 6) Live HTTP ChatServer smoke on GPU
    {
      ingine::serve::ServerOptions sopts;
      sopts.max_model_len = 8192;
      ingine::serve::ChatServer server(*gen, fmt, sopts);
      CHECK(server.bind("127.0.0.1", 0));
      server.start();
      httplib::Client cli("127.0.0.1", server.port());
      cli.set_read_timeout(60, 0);

      auto h = cli.Get("/health");
      CHECK(h && h->status == 200);
      const Json hj = Json::parse(h->body);
      CHECK(hj["model_family"] == "embedding_gemma2" && hj["encoder_only"] == true);

      auto r = cli.Post("/v1/embeddings",
                        R"({"input":["What causes the northern lights?"],"prompt_name":"SearchQuery","dimensions":256,"encoding_format":"float"})",
                        "application/json");
      CHECK(r && r->status == 200);
      const Json rj = Json::parse(r->body);
      CHECK(rj["data"][0]["embedding"].size() == 256);

      auto rb64 = cli.Post("/v1/embeddings",
                           R"({"input":"hello world","dimensions":128,"encoding_format":"base64"})",
                           "application/json");
      CHECK(rb64 && rb64->status == 200);
      const Json bj = Json::parse(rb64->body);
      CHECK(bj["data"][0]["embedding"].is_string());

      auto rchat = cli.Post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})", "application/json");
      CHECK(rchat && rchat->status == 400);
      server.stop();
      std::printf("[PASS] live HTTP ChatServer GPU smoke verified\n");
    }

    if (!out_json.empty()) {
      Json report{
          {"device", prop.name},
          {"sm", sm},
          {"mode", is_bf16 ? "bf16_fp32_acc" : "fp32_zero_fp16"},
          {"weights_device_bytes", residency.weights_device_bytes},
          {"weights_mib", weight_mib},
          {"gpu_load_s", gpu_load_s},
          {"matryoshka_parity", mrl_report},
          {"batch_invariance_b1_vs_b8", {{"min_cos", batch_min_cos}, {"max_abs", batch_max_abs}}},
          {"sliding_window_gt512_parity", {{"cos", swa_cos}, {"max_abs", swa_abs}}},
          {"benchmarks", bench_j},
          {"http_smoke", "PASS"},
          {"status", "PASS"},
      };
      std::filesystem::create_directories(out_json.parent_path());
      std::ofstream f(out_json);
      f << report.dump(2) << "\n";
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eg2_gpu failed: %s\n", e.what());
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
