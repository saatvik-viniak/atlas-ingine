// GPU parity, batch-invariance, sliding-window boundary, and live HTTP smoke test for
// google/embeddinggemma-2 (#968) on both sm_80+ (BF16) and sm_70 V100 (native FP32, zero FP16).
#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_cpu.h"
#include "backends/embedding_gemma2/eg2_cuda.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "core/device.h"
#include "core/sha256.h"
#include "ingine/generator.h"
#include "ingine/platos_key.h"
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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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

// Deterministic splitmix64 pseudo-random float in [-scale, +scale] (avoids sinusoidal dot-product cancellation).
float prng_f32(std::uint64_t idx, std::uint64_t seed, float scale) {
  std::uint64_t z = idx * 0x9e3779b97f4a7c15ull ^ (seed * 0xbf58476d1ce4e5b9ull + 0x94d049bb133111ebull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= (z >> 31);
  const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
  return static_cast<float>((2.0 * u - 1.0) * static_cast<double>(scale));
}

std::string write_f32_platos_key(const std::filesystem::path& path, int d_in, int d_out, int seed,
                                 const std::map<std::string, std::string>& meta = {}) {
  const std::uint64_t s64 = static_cast<std::uint64_t>(seed);
  std::vector<float> R(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_out));
  for (std::size_t i = 0; i < R.size(); ++i) {
    R[i] = prng_f32(i + 1, s64 + 1u, 0.25f);
  }
  std::vector<float> dm(static_cast<std::size_t>(d_in)), tm(static_cast<std::size_t>(d_out));
  for (int i = 0; i < d_in; ++i) dm[i] = prng_f32(static_cast<std::uint64_t>(i + 1), s64 + 2u, 0.05f);
  for (int j = 0; j < d_out; ++j) tm[j] = prng_f32(static_cast<std::uint64_t>(j + 1), s64 + 3u, 0.1f);
  const std::vector<float> sc = {1.15f};

  nlohmann::ordered_json hdr = nlohmann::ordered_json::object();
  if (!meta.empty()) {
    nlohmann::ordered_json m = nlohmann::ordered_json::object();
    for (const auto& [k, v] : meta) m[k] = v;
    hdr["__metadata__"] = std::move(m);
  }
  std::vector<std::uint8_t> payload;
  auto add_tensor = [&](const std::string& name, const std::vector<std::int64_t>& shape, const std::vector<float>& vals) {
    const std::uint64_t b = payload.size();
    const std::size_t nb = vals.size() * sizeof(float);
    payload.resize(payload.size() + nb);
    std::memcpy(payload.data() + b, vals.data(), nb);
    const std::uint64_t e = payload.size();
    hdr[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {b, e}}};
  };
  add_tensor("R", {d_in, d_out}, R);
  add_tensor("donor_mean", {d_in}, dm);
  add_tensor("target_mean", {d_out}, tm);
  add_tensor("scale", {1}, sc);

  const std::string hdr_str = hdr.dump();
  const std::uint64_t hdr_len = static_cast<std::uint64_t>(hdr_str.size());
  std::vector<std::uint8_t> file_bytes(8 + hdr_str.size() + payload.size());
  for (int i = 0; i < 8; ++i) file_bytes[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
  std::memcpy(file_bytes.data() + 8, hdr_str.data(), hdr_str.size());
  std::memcpy(file_bytes.data() + 8 + hdr_str.size(), payload.data(), payload.size());

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  CHECK(static_cast<bool>(out));
  out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
  out.flush();
  CHECK(static_cast<bool>(out));
  return ingine::sha256_hex(ingine::sha256_bytes(file_bytes.data(), file_bytes.size()));
}

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

    // 7) Finding (1) & (2): EG2 GPU Plato's Key precision (zero FP16: BF16 on sm_86, FP32 on sm_70) across BOTH
    //    projection branches (batch <= 8 warp kernel AND batch > 8 cuBLASLt), for BOTH 512-d intermediate (visit_2)
    //    AND 768-d final (final_normed) cartridges + live GPU HTTP + non-FP16 dynamic-range proof (>65504)!
    Json pk_report = Json::object();
    {
      const auto tmp = std::filesystem::temp_directory_path() / "ingine_eg2_gpu_platos_key_968";
      std::filesystem::remove_all(tmp);
      std::filesystem::create_directories(tmp);

      const ingine::PlatosKeyPrecision expected_prec =
          is_bf16 ? ingine::PlatosKeyPrecision::Bf16 : ingine::PlatosKeyPrecision::Fp32;
      const char* prec_name = is_bf16 ? "bf16_fp32_acc" : "fp32_zero_fp16";

      // (a) Direct PlatosKeyGpu large-magnitude (>65504) zero-FP16 proof on BOTH B=2 (<=8) and B=12 (>8)!
      //     FP16 saturates/clamps at 65504 -> would fail this check by >34%; BF16 and FP32 pass cleanly!
      {
        const auto k_range_path = tmp / "eg2_range_512x64.safetensors";
        (void)write_f32_platos_key(k_range_path, 512, 64, 77, {{"format", "platos_key_v1"}});
        auto pk_cpu = ingine::PlatosKey::load(k_range_path, 512);
        auto pk_gpu = ingine::PlatosKeyGpu::create(pk_cpu, nullptr, expected_prec);
        CHECK(pk_gpu->precision() == expected_prec);

        // Also preserve default Fp16 path for Inkling/Qwen backends: verify B=2 (<=8) and B=12 (>8) parity on normal range!
        auto pk_gpu_f16 = ingine::PlatosKeyGpu::create(pk_cpu, nullptr);
        CHECK(pk_gpu_f16->precision() == ingine::PlatosKeyPrecision::Fp16);
        Json f16_preserve_j = Json::object();
        for (int B_f16 : {2, 12}) {
          std::vector<float> hx_n(static_cast<std::size_t>(B_f16) * 512u);
          for (std::size_t idx = 0; idx < hx_n.size(); ++idx) {
            hx_n[idx] = prng_f32(idx + 1, 901u, 1.5f);
          }
          std::vector<float> yc(static_cast<std::size_t>(B_f16) * 64u, 0.0f);
          pk_cpu->project_cpu(hx_n.data(), B_f16, yc.data(), false);
          float *dx = nullptr, *dy = nullptr;
          CUDA_CHECK(cudaMalloc(&dx, hx_n.size() * sizeof(float)));
          CUDA_CHECK(cudaMalloc(&dy, yc.size() * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(dx, hx_n.data(), hx_n.size() * sizeof(float), cudaMemcpyHostToDevice));
          pk_gpu_f16->project(dx, B_f16, dy, false, nullptr);
          CUDA_CHECK(cudaDeviceSynchronize());
          std::vector<float> yg(yc.size(), 0.0f);
          CUDA_CHECK(cudaMemcpy(yg.data(), dy, yg.size() * sizeof(float), cudaMemcpyDeviceToHost));
          cudaFree(dx);
          cudaFree(dy);
          const auto ps = compare_vecs(yg, yc);
          std::printf("[PK PRESERVED FP16 B=%d] cos=%.7f max_abs=%.6e\n", B_f16, ps.cosine, ps.max_abs);
          CHECK(ps.cosine >= 0.9999);
          f16_preserve_j["B=" + std::to_string(B_f16)] = {{"cos", ps.cosine}, {"max_abs", ps.max_abs}};
        }
        pk_report["preserved_default_fp16_path"] = f16_preserve_j;

        Json range_j = Json::object();
        for (int B_test : {2, 12}) {
          std::vector<float> hx(static_cast<std::size_t>(B_test) * 512u);
          for (int b = 0; b < B_test; ++b) {
            for (int d = 0; d < 512; ++d) {
              // Values up to +-120000.0f exceed IEEE FP16 max (65504.0f) by nearly 2x!
              hx[static_cast<std::size_t>(b) * 512u + d] =
                  prng_f32(static_cast<std::uint64_t>(b * 512 + d + 1), 902u, 120000.0f);
            }
          }
          std::vector<float> y_cpu(static_cast<std::size_t>(B_test) * 64u, 0.0f);
          pk_cpu->project_cpu(hx.data(), B_test, y_cpu.data(), false);

          float *dx = nullptr, *dy = nullptr;
          CUDA_CHECK(cudaMalloc(&dx, hx.size() * sizeof(float)));
          CUDA_CHECK(cudaMalloc(&dy, y_cpu.size() * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
          pk_gpu->project(dx, B_test, dy, false, nullptr);
          CUDA_CHECK(cudaDeviceSynchronize());
          std::vector<float> y_gpu(y_cpu.size(), 0.0f);
          CUDA_CHECK(cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
          cudaFree(dx);
          cudaFree(dy);

          double worst_rel = 0.0, worst_cos = 1.0;
          for (int b = 0; b < B_test; ++b) {
            std::vector<float> rc(y_cpu.begin() + b * 64, y_cpu.begin() + (b + 1) * 64);
            std::vector<float> rg(y_gpu.begin() + b * 64, y_gpu.begin() + (b + 1) * 64);
            const auto ps = compare_vecs(rg, rc);
            worst_cos = std::min(worst_cos, ps.cosine);
            const double norm_c = l2_norm(rc);
            worst_rel = std::max(worst_rel, ps.max_abs / std::max(norm_c, 1e-6));
          }
          std::printf("[PK ZERO-FP16 RANGE >65504 prec=%s B=%d] min_cos=%.7f max_rel_err=%.6e\n", prec_name, B_test,
                      worst_cos, worst_rel);
          CHECK(worst_cos >= (is_bf16 ? 0.9999 : 0.9999999));
          CHECK(worst_rel < (is_bf16 ? 5e-3 : 1e-5));
          range_j["B=" + std::to_string(B_test)] = {{"min_cos", worst_cos}, {"max_rel_err", worst_rel}};
        }
        pk_report["zero_fp16_large_range_gt65504"] = range_j;
      }

      // (b) Real Eg2CudaBackend vs Eg2CpuBackend keyed parity on BOTH B=2 (<=8) AND B=12 (>8),
      //     for BOTH 512-d intermediate (visit_2) AND 768-d final (final_normed) cartridges + live GPU HTTP!
      struct KeyCase {
        std::string name;
        int d_in;
        int d_out;
        std::string tap_spec;
        std::optional<int> tap_layer;
        int seed;
      };
      const std::vector<KeyCase> key_cases = {
          {"intermediate_visit2_512x2", 512, 2, "visit_2", 2, 101},
          {"intermediate_visit2_512x64", 512, 64, "visit_2", 2, 103},
          {"final_normed_768x2", 768, 2, "final_normed", std::nullopt, 201},
          {"final_normed_768x64", 768, 64, "final_normed", std::nullopt, 203},
      };

      // Build 12 distinct tokenized inputs for B=2 and B=12 checks
      std::vector<std::vector<std::int32_t>> batch12;
      for (int i = 0; i < 12; ++i) {
        const auto& [pn, txt] = test_pairs[static_cast<std::size_t>(i) % test_pairs.size()];
        batch12.push_back(tok.encode_with_post_processor(prompts.at(pn) + txt + " #" + std::to_string(i), false));
      }

      // Precompute exact CPU FP32 unnormalized embeddings ONCE for all 12 inputs at tap_layer=2 (512d) and final (768d),
      // so we compare GPU keyed outputs against exact CPU reference across all 4 cartridges x B={2, 12} without redundant CPU forward passes.
      ingine::model::EmbeddingRequest ref_req_tap2;
      ref_req_tap2.inputs = batch12;
      ref_req_tap2.tap_layer = 2;
      ref_req_tap2.normalize = false;
      const auto cpu_tap2_12 = cpu_backend.embed(ref_req_tap2).embeddings;
      CHECK(cpu_tap2_12.size() == 12 && cpu_tap2_12[0].size() == 512);

      ingine::model::EmbeddingRequest ref_req_fin;
      ref_req_fin.inputs = batch12;
      ref_req_fin.normalize = false;
      const auto cpu_fin_12 = cpu_backend.embed(ref_req_fin).embeddings;
      CHECK(cpu_fin_12.size() == 12 && cpu_fin_12[0].size() == 768);

      for (const auto& kc : key_cases) {
        const auto kp = tmp / (kc.name + ".safetensors");
        const auto mp = tmp / (kc.name + ".json");
        const std::string sha = write_f32_platos_key(kp, kc.d_in, kc.d_out, kc.seed, {{"format", "platos_key_v1"}});
        {
          std::ofstream mf(mp);
          mf << Json{{"cartridge_id", "eg2-" + kc.name},
                     {"cartridge_sha256", sha},
                     {"donor_model_id", "google/embeddinggemma-2"},
                     {"tap_spec", kc.tap_spec},
                     {"d_donor", kc.d_in},
                     {"d_target", kc.d_out}}
                    .dump();
        }
        ingine::model::RuntimeOptions kopts = opts;
        kopts.platos_key = kp;
        kopts.platos_key_metadata = mp;
        Eg2CudaBackend kgpu(kopts);
        auto pk_ref = kgpu.platos_key();
        CHECK(pk_ref != nullptr);
        CHECK(kgpu.platos_key_precision() == std::optional<ingine::PlatosKeyPrecision>(expected_prec));
        CHECK(kgpu.tap_embedding_dim() == 512 && kgpu.embedding_dim() == 768);

        const auto& cpu_base_12 = kc.tap_layer.has_value() ? cpu_tap2_12 : cpu_fin_12;
        std::vector<float> flat_cpu_in(12u * static_cast<std::size_t>(kc.d_in));
        for (int b = 0; b < 12; ++b) {
          std::memcpy(flat_cpu_in.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(kc.d_in),
                      cpu_base_12[static_cast<std::size_t>(b)].data(),
                      static_cast<std::size_t>(kc.d_in) * sizeof(float));
        }
        std::vector<float> flat_cpu_raw(12u * static_cast<std::size_t>(kc.d_out), 0.0f);
        std::vector<float> flat_cpu_norm(12u * static_cast<std::size_t>(kc.d_out), 0.0f);
        pk_ref->project_cpu(flat_cpu_in.data(), 12, flat_cpu_raw.data(), false);
        pk_ref->project_cpu(flat_cpu_in.data(), 12, flat_cpu_norm.data(), true);

        Json case_j{{"d_in", kc.d_in}, {"d_out", kc.d_out}, {"tap_spec", kc.tap_spec}, {"precision", prec_name}};
        for (int B_test : {2, 12}) {
          ingine::model::EmbeddingRequest kr;
          kr.inputs.assign(batch12.begin(), batch12.begin() + B_test);
          kr.tap_layer = kc.tap_layer;
          kr.use_platos_key = true;
          kr.cartridge_id = "eg2-" + kc.name;
          kr.cartridge_sha256 = sha;
          kr.normalize = false;
          const auto gres = kgpu.embed(kr);
          CHECK(gres.stats.output_dim == kc.d_out);
          CHECK(static_cast<int>(gres.embeddings.size()) == B_test);

          double worst_cos = 1.0, worst_abs = 0.0;
          for (int b = 0; b < B_test; ++b) {
            const float* c_ptr = flat_cpu_raw.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(kc.d_out);
            std::vector<float> cref(c_ptr, c_ptr + kc.d_out);
            const auto ps = compare_vecs(gres.embeddings[b], cref);
            worst_cos = std::min(worst_cos, ps.cosine);
            worst_abs = std::max(worst_abs, ps.max_abs);
          }
          // Also check normalized output unit norm & parity
          kr.normalize = true;
          const auto gres_n = kgpu.embed(kr);
          double worst_cos_n = 1.0, worst_abs_n = 0.0;
          for (int b = 0; b < B_test; ++b) {
            CHECK(std::abs(l2_norm(gres_n.embeddings[b]) - 1.0) < 1e-4);
            const float* c_ptr = flat_cpu_norm.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(kc.d_out);
            std::vector<float> cref(c_ptr, c_ptr + kc.d_out);
            const auto ps = compare_vecs(gres_n.embeddings[b], cref);
            worst_cos_n = std::min(worst_cos_n, ps.cosine);
            worst_abs_n = std::max(worst_abs_n, ps.max_abs);
          }
          std::printf("[PK GPU PARITY %s prec=%s B=%d] raw_cos=%.7f raw_max_abs=%.6f norm_cos=%.7f norm_max_abs=%.6f\n",
                      kc.name.c_str(), prec_name, B_test, worst_cos, worst_abs, worst_cos_n, worst_abs_n);
          const double pk_cos_thresh = is_bf16 ? 0.9990 : 0.999995;
          CHECK(worst_cos >= pk_cos_thresh);
          CHECK(worst_cos_n >= pk_cos_thresh);
          case_j["B=" + std::to_string(B_test)] = {
              {"raw_min_cos", worst_cos},
              {"raw_max_abs", worst_abs},
              {"norm_min_cos", worst_cos_n},
              {"norm_max_abs", worst_abs_n},
          };
        }

        // Live GPU HTTP smoke + mismatched HTTP 400 control for both 512x2 visit_2 and 768x2 final_normed
        if (kc.d_out == 2) {
          ingine::serve::ServerOptions sopts;
          sopts.max_model_len = 8192;
          ingine::serve::ChatServer ksrv(kgpu, fmt, sopts);
          CHECK(ksrv.bind("127.0.0.1", 0));
          ksrv.start();
          httplib::Client kcli("127.0.0.1", ksrv.port());
          kcli.set_read_timeout(60, 0);

          Json req_ok{{"input", Json::array({"What causes the northern lights?", "Solar wind and magnetosphere"})},
                      {"platos_key", true},
                      {"cartridge_id", "eg2-" + kc.name},
                      {"cartridge_sha256", sha},
                      {"normalize", true}};
          if (kc.tap_layer.has_value()) req_ok["tap_layer"] = *kc.tap_layer;
          auto r_ok = kcli.Post("/v1/embeddings", req_ok.dump(), "application/json");
          CHECK(r_ok && r_ok->status == 200);
          const Json rj = Json::parse(r_ok->body);
          CHECK(rj["data"].size() == 2 && rj["data"][0]["embedding"].size() == 2);

          // Mismatched tap control -> HTTP 400
          Json req_bad{{"input", "hi"}, {"platos_key", true}};
          if (kc.tap_layer.has_value()) {
            // omit tap_layer -> 400
          } else {
            req_bad["tap_layer"] = 2;  // intermediate tap on final key -> 400
          }
          auto r_bad = kcli.Post("/v1/embeddings", req_bad.dump(), "application/json");
          CHECK(r_bad && r_bad->status == 400);
          ksrv.stop();
          case_j["http_smoke"] = "PASS";
        }
        pk_report[kc.name] = case_j;
      }

      std::filesystem::remove_all(tmp);
      std::printf("[PASS] EG2 GPU Plato's Key precision (%s) B<=8 & B>8 parity verified for 512d-intermediate & 768d-final\n",
                  prec_name);
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
          {"platos_key_gpu_parity", pk_report},
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
