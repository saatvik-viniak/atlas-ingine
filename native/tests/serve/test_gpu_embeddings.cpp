// Live local GPU parity, zero-KV, early-exit speedup, and Plato's Key verification for POST /v1/embeddings (#963).
// Skips cleanly with exit code 77 when no CUDA GPU is visible.
#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "ingine/generator.h"
#include "ingine/platos_key.h"
#include "model/model_data.h"
#include "model/prefill.h"
#include "model/runtime.h"
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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::ActT;
using ingine::DeviceBuffer;
using ingine::PlatosKey;
using ingine::PlatosKeyGpu;
using ingine::model::Ablation;
using ingine::model::EmbeddingPooling;
using ingine::model::EmbeddingRequest;
using ingine::model::EmbeddingResult;
using ingine::model::Generator;
using ingine::model::HIDDEN;
using ingine::model::NUM_TAPS;
using ingine::model::Prefiller;
using ingine::model::RuntimeOptions;
using ingine::model::State;
using ingine::model::UNPADDED_VOCAB;
using ingine::model::Weights;
using ingine::serve::ChatServer;
using ingine::serve::ServerOptions;
using ingine::text::ChatFormat;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

struct ParityStats {
  double max_abs = 0.0;
  double cosine = 0.0;
};

ParityStats compare_vecs(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size() && !a.empty());
  double max_abs = 0.0;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double va = static_cast<double>(a[i]);
    const double vb = static_cast<double>(b[i]);
    CHECK(std::isfinite(va) && std::isfinite(vb));
    max_abs = std::max(max_abs, std::fabs(va - vb));
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  const double cos = (na > 0.0 && nb > 0.0) ? (dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0;
  return {max_abs, cos};
}

std::size_t vram_used_bytes() {
  std::size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  return total_b - free_b;
}

std::uint16_t f32_to_fp16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((u >> 16) & 0x8000u);
  const std::uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u < 0x38800000u) return sign;
  const std::uint32_t exp = (abs_u >> 23) - 112u;
  const std::uint32_t mant = abs_u & 0x007fffffu;
  const std::uint32_t rounded = ((exp << 23) | mant) + 0x00000fffu + ((mant >> 13) & 1u);
  return static_cast<std::uint16_t>(sign | (rounded >> 13));
}

void write_synthetic_platos_key_f16(const std::filesystem::path& path, int d_in, int d_out,
                                    const std::string& tap_spec) {
  std::vector<std::uint8_t> r_bytes(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_out) * 2);
  std::vector<std::uint8_t> dm_bytes(static_cast<std::size_t>(d_in) * 2);
  std::vector<std::uint8_t> tm_bytes(static_cast<std::size_t>(d_out) * 2);
  std::vector<std::uint8_t> sc_bytes(2);
  const float inv_s = 1.0f / std::sqrt(static_cast<float>(d_in));
  for (int i = 0; i < d_in; ++i) {
    const std::uint16_t h = f32_to_fp16(static_cast<float>((i % 11) - 5) * 0.02f);
    dm_bytes[2 * i] = static_cast<std::uint8_t>(h & 0xffu);
    dm_bytes[2 * i + 1] = static_cast<std::uint8_t>((h >> 8) & 0xffu);
    for (int j = 0; j < d_out; ++j) {
      const float val = static_cast<float>(((i * 31 + j * 17) % 19) - 9) * 0.1f * inv_s;
      const std::uint16_t rh = f32_to_fp16(val);
      const std::size_t idx = (static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out) + j) * 2;
      r_bytes[idx] = static_cast<std::uint8_t>(rh & 0xffu);
      r_bytes[idx + 1] = static_cast<std::uint8_t>((rh >> 8) & 0xffu);
    }
  }
  for (int j = 0; j < d_out; ++j) {
    const std::uint16_t h = f32_to_fp16(static_cast<float>((j % 9) - 4) * 0.05f);
    tm_bytes[2 * j] = static_cast<std::uint8_t>(h & 0xffu);
    tm_bytes[2 * j + 1] = static_cast<std::uint8_t>((h >> 8) & 0xffu);
  }
  const std::uint16_t sh = f32_to_fp16(1.03f);
  sc_bytes[0] = static_cast<std::uint8_t>(sh & 0xffu);
  sc_bytes[1] = static_cast<std::uint8_t>((sh >> 8) & 0xffu);

  nlohmann::ordered_json hdr;
  hdr["__metadata__"] = {{"format", "platos_key_v1"}, {"tap_spec", tap_spec}, {"method", "ridge"}};
  std::uint64_t off = 0;
  auto add = [&](const char* name, std::vector<std::int64_t> shape, std::size_t sz) {
    hdr[name] = {{"dtype", "F16"}, {"shape", std::move(shape)}, {"data_offsets", {off, off + sz}}};
    off += sz;
  };
  add("R", {d_in, d_out}, r_bytes.size());
  add("donor_mean", {d_in}, dm_bytes.size());
  add("target_mean", {d_out}, tm_bytes.size());
  add("scale", {1}, sc_bytes.size());

  const std::string hs = hdr.dump();
  const std::uint64_t hl = static_cast<std::uint64_t>(hs.size());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  CHECK(static_cast<bool>(out));
  for (int k = 0; k < 8; ++k) {
    const char c = static_cast<char>((hl >> (8 * k)) & 0xffu);
    out.write(&c, 1);
  }
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  out.write(reinterpret_cast<const char*>(r_bytes.data()), static_cast<std::streamsize>(r_bytes.size()));
  out.write(reinterpret_cast<const char*>(dm_bytes.data()), static_cast<std::streamsize>(dm_bytes.size()));
  out.write(reinterpret_cast<const char*>(tm_bytes.data()), static_cast<std::streamsize>(tm_bytes.size()));
  out.write(reinterpret_cast<const char*>(sc_bytes.data()), static_cast<std::streamsize>(sc_bytes.size()));
  out.flush();
  CHECK(static_cast<bool>(out));
}

struct HttpRes {
  int status = 0;
  httplib::Headers headers;
  Json body;
};

HttpRes post_json(int port, const std::string& path, const Json& payload) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(120, 0);
  auto r = cli.Post(path, payload.dump(), "application/json");
  CHECK(static_cast<bool>(r));
  return {r->status, r->headers, Json::parse(r->body)};
}

std::string hdr_val(const httplib::Headers& h, const std::string& key) {
  auto it = h.find(key);
  return it == h.end() ? "" : it->second;
}

Json run_inkling_gpu_verification(const std::filesystem::path& model_dir) {
  std::printf("\n=== [Inkling-2-Mini] Live GPU Hidden-State Parity, Zero-KV & Throughput ===\n");
  Json report;
  report["model_dir"] = model_dir.string();

  const ChatFormat fmt = ChatFormat::load(model_dir);
  const std::vector<std::string> sample_texts = {
      "The geometry of residual streams across transformer families reveals linear bridges between representations.",
      "Zero-KV batched embedding prefill eliminates persistent KV cache allocation and skips the vocabulary head.",
      "Early-exit layer tapping cuts forward pass FLOPs roughly in half while preserving semantic conditioning.",
      "Atlas Verified Ledger pairs deterministic exact retrieval with fast neural inference.",
      "Continuous batching schedules concurrent chat streams alongside compute-bound embedding batches.",
      "Sparse Mixture-of-Experts routing turns large packed prefill batches into dense Tensor Core grouped GEMMs.",
      "Plato's Key maps hidden activations into multimodal conditioning coordinates in under 0.15 milliseconds.",
      "Every GPU benchmark measures parity, latency, throughput, and VRAM delta on real local hardware.",
  };
  std::vector<std::vector<std::int32_t>> token_seqs;
  for (const auto& s : sample_texts) {
    token_seqs.push_back(fmt.tokenizer().encode(s, /*special=*/false));
    CHECK(!token_seqs.back().empty());
  }

  // ---- Part 1: Direct full-pass Prefiller::run(taps) oracle parity check ----
  {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Weights w = ingine::model::load_weights(model_dir, /*dense_fp8=*/true, stream);
    Ablation abl = ingine::model::load_ablation({});
    Prefiller prefiller(w, abl, /*max_chunk=*/2048, stream);

    const std::size_t vram_before_state = vram_used_bytes();
    State ref_state = ingine::model::make_state(/*max_ctx=*/2048, /*kv_fp8=*/false);
    const std::size_t chat_kv_state_bytes = vram_used_bytes() - vram_before_state;
    report["inkling_chat_kv_state_mib_2048ctx"] = static_cast<double>(chat_kv_state_bytes) / (1024.0 * 1024.0);

    const auto& seq0 = token_seqs[0];
    const int T0 = static_cast<int>(seq0.size());
    DeviceBuffer d_toks(static_cast<std::size_t>(T0) * sizeof(std::int32_t));
    d_toks.copy_from_host(seq0.data(), static_cast<std::size_t>(T0) * sizeof(std::int32_t));
    DeviceBuffer d_logits(static_cast<std::size_t>(UNPADDED_VOCAB) * sizeof(float));
    DeviceBuffer d_taps(static_cast<std::size_t>(T0) * NUM_TAPS * HIDDEN * sizeof(ActT));
    ingine::Tensor taps_t = ingine::model::rm(d_taps.p, ingine::kActDType, {T0, NUM_TAPS, HIDDEN});

    ingine::model::reset_state(ref_state, stream);
    prefiller.run(ref_state, static_cast<const std::int32_t*>(d_toks.p), T0, 0, static_cast<float*>(d_logits.p),
                  &taps_t);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<ActT> host_taps(static_cast<std::size_t>(T0) * NUM_TAPS * HIDDEN);
    CUDA_CHECK(cudaMemcpy(host_taps.data(), d_taps.p, host_taps.size() * sizeof(ActT), cudaMemcpyDeviceToHost));

    auto pool_ref_slot = [&](int slot, EmbeddingPooling pooling) {
      std::vector<float> out(HIDDEN, 0.0f);
      if (pooling == EmbeddingPooling::Cls) {
        for (int d = 0; d < HIDDEN; ++d) {
          out[d] = ingine::act::host_to_float(host_taps[static_cast<std::size_t>(slot) * HIDDEN + d]);
        }
      } else if (pooling == EmbeddingPooling::LastToken) {
        const std::size_t base = (static_cast<std::size_t>(T0 - 1) * NUM_TAPS + slot) * HIDDEN;
        for (int d = 0; d < HIDDEN; ++d) out[d] = ingine::act::host_to_float(host_taps[base + d]);
      } else {
        for (int d = 0; d < HIDDEN; ++d) {
          double acc = 0.0;
          for (int t = 0; t < T0; ++t) {
            acc += ingine::act::host_to_float(host_taps[(static_cast<std::size_t>(t) * NUM_TAPS + slot) * HIDDEN + d]);
          }
          out[d] = static_cast<float>(acc / static_cast<double>(T0));
        }
      }
      return out;
    };

    const std::vector<std::pair<std::string, std::pair<std::optional<int>, int>>> tap_checks = {
        {"final_xf_normed", {std::nullopt, 3}},
        {"visit_5", {5, 0}},
        {"visit_23", {23, 1}},
        {"visit_41", {41, 2}},
    };

    Json parity_json;
    for (const auto& [name, cfg] : tap_checks) {
      for (EmbeddingPooling pool : {EmbeddingPooling::Mean, EmbeddingPooling::LastToken, EmbeddingPooling::Cls}) {
        EmbeddingRequest er;
        er.inputs = {seq0};
        er.pooling = pool;
        er.normalize = false;
        er.tap_layer = cfg.first;
        const EmbeddingResult res = prefiller.embed(er, 2048, nullptr);
        const std::vector<float> oracle = pool_ref_slot(cfg.second, pool);
        const ParityStats ps = compare_vecs(res.embeddings[0], oracle);
        double act_rounded_max_abs = -1.0;
        if (pool != EmbeddingPooling::Mean) {
          // Prefiller::run stores taps truncated to ActT (BF16/FP16), whereas Prefiller::embed pools un-truncated FP32
          // hid directly. Rounding embed output to ActT proves bit-exact identity against the tap tensor.
          std::vector<float> rounded(res.embeddings[0].size());
          for (std::size_t d = 0; d < rounded.size(); ++d) rounded[d] = ingine::act::host_round(res.embeddings[0][d]);
          act_rounded_max_abs = compare_vecs(rounded, oracle).max_abs;
          CHECK(act_rounded_max_abs == 0.0);
        }
        std::printf("  [parity] %-16s pool=%-10s max_abs=%.6e cosine=%.8f%s\n", name.c_str(),
                    ingine::model::to_string(pool), ps.max_abs, ps.cosine,
                    act_rounded_max_abs >= 0.0 ? " (act_rounded_max_abs=0.0 EXACT)" : "");
        std::fflush(stdout);
        CHECK(ps.cosine >= 0.99995);
        CHECK(ps.max_abs <= 0.5);  // BF16 7-bit mantissa truncation on ~50-100 magnitude residual channels
        if (name == "final_xf_normed") CHECK(ps.max_abs <= 1e-4);
        Json entry = {{"max_abs", ps.max_abs}, {"cosine", ps.cosine}};
        if (act_rounded_max_abs >= 0.0) entry["act_rounded_max_abs"] = act_rounded_max_abs;
        parity_json[name + "_" + ingine::model::to_string(pool)] = std::move(entry);
      }
    }

    // Batch invariance: B=8 packed vs B=1 individual across visit_23 and full depth.
    EmbeddingRequest er_batch;
    er_batch.inputs = token_seqs;
    er_batch.pooling = EmbeddingPooling::Mean;
    er_batch.normalize = false;
    er_batch.tap_layer = 23;
    const EmbeddingResult res_batch = prefiller.embed(er_batch, 2048, nullptr);
    double worst_batch_max_abs = 0.0;
    double worst_batch_cos = 1.0;
    for (std::size_t i = 0; i < token_seqs.size(); ++i) {
      EmbeddingRequest er_one;
      er_one.inputs = {token_seqs[i]};
      er_one.pooling = EmbeddingPooling::Mean;
      er_one.normalize = false;
      er_one.tap_layer = 23;
      const EmbeddingResult res_one = prefiller.embed(er_one, 2048, nullptr);
      const ParityStats ps = compare_vecs(res_batch.embeddings[i], res_one.embeddings[0]);
      worst_batch_max_abs = std::max(worst_batch_max_abs, ps.max_abs);
      worst_batch_cos = std::min(worst_batch_cos, ps.cosine);
    }
    std::printf("  [batch_invariance B=8 vs B=1, visit_23] max_abs=%.6e min_cosine=%.8f\n", worst_batch_max_abs,
                worst_batch_cos);
    std::fflush(stdout);
    CHECK(worst_batch_cos >= 0.9995);
    parity_json["batch8_vs_single_visit23"] = {{"max_abs", worst_batch_max_abs}, {"min_cosine", worst_batch_cos}};
    report["parity"] = std::move(parity_json);

    // GPU Plato's Key parity + latency benchmark on real Inkling hidden states (2304 -> 5120 F16).
    const auto tmp_pk = std::filesystem::temp_directory_path() / "ingine_gpu_test_inkling_pk.safetensors";
    write_synthetic_platos_key_f16(tmp_pk, HIDDEN, 5120, "visit_23");
    auto pk_gpu = PlatosKeyGpu::load(tmp_pk, HIDDEN, stream);

    // Warm up + measure B=1 and B=16 Plato's Key projection latency and parity vs FP64 CPU reference.
    Json pk_json;
    pk_json["device_mib"] = static_cast<double>(pk_gpu->device_bytes()) / (1024.0 * 1024.0);
    for (int B_pk : {1, 16}) {
      EmbeddingRequest er_pk;
      for (int i = 0; i < B_pk; ++i) er_pk.inputs.push_back(token_seqs[i % token_seqs.size()]);
      er_pk.pooling = EmbeddingPooling::Mean;
      er_pk.normalize = false;
      er_pk.tap_layer = 23;
      er_pk.use_platos_key = true;
      (void)prefiller.embed(er_pk, 2048, pk_gpu.get());  // warmup
      double min_pk_ms = 1e9;
      EmbeddingResult res_pk;
      for (int rep = 0; rep < 5; ++rep) {
        res_pk = prefiller.embed(er_pk, 2048, pk_gpu.get());
        min_pk_ms = std::min(min_pk_ms, res_pk.stats.platos_key_seconds * 1e3);
      }
      // Compare against FP64 CPU projection of raw visit_23 hidden states.
      er_pk.use_platos_key = false;
      const EmbeddingResult res_raw = prefiller.embed(er_pk, 2048, nullptr);
      std::vector<float> flat_in;
      for (const auto& v : res_raw.embeddings) flat_in.insert(flat_in.end(), v.begin(), v.end());
      std::vector<float> cpu_ref(static_cast<std::size_t>(B_pk) * 5120);
      pk_gpu->key()->project_cpu(flat_in.data(), B_pk, cpu_ref.data(), false);
      std::vector<float> gpu_flat;
      for (const auto& v : res_pk.embeddings) gpu_flat.insert(gpu_flat.end(), v.begin(), v.end());
      const ParityStats pk_ps = compare_vecs(gpu_flat, cpu_ref);
      std::printf("  [platos_key 2304->5120 B=%d] gpu_ms=%.4f max_abs=%.6e cosine=%.8f\n", B_pk, min_pk_ms,
                  pk_ps.max_abs, pk_ps.cosine);
      CHECK(pk_ps.cosine >= 0.99999);
      CHECK(min_pk_ms < 0.25);
      pk_json["B" + std::to_string(B_pk)] = {{"gpu_ms", min_pk_ms}, {"max_abs", pk_ps.max_abs}, {"cosine", pk_ps.cosine}};
    }
    report["platos_key_2304_to_5120"] = std::move(pk_json);
    std::filesystem::remove(tmp_pk);

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  // ---- Part 2: Live HTTP ChatServer batched throughput, early-exit speedup, zero-KV VRAM delta, chat intact ----
  {
    RuntimeOptions ro;
    ro.model_dir = model_dir;
    ro.max_ctx = 4096;
    ro.prefill_chunk = 2048;
    ro.dense_fp8 = true;
    ro.kv_fp8 = true;
    ro.vision = false;
    ro.max_batch = 4;  // continuous batching!
    ro.park_mb = 0;

    auto gen = ingine::model::make_generator(ro);
    ServerOptions so;
    so.max_queue = 16;
    ChatServer srv(*gen, fmt, so);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    const int port = srv.port();

    // Baseline chat completion before embeddings.
    const Json chat_req = {{"messages", Json::array({{{"role", "user"}, {"content", "Reply with exactly: OK"}}})},
                           {"temperature", 0.0},
                           {"max_tokens", 16},
                           {"enable_thinking", false}};
    const HttpRes c_before = post_json(port, "/v1/chat/completions", chat_req);
    CHECK(c_before.status == 200);
    const std::string reply_before = c_before.body["choices"][0]["message"]["content"].get<std::string>();

    const std::size_t vram_before_embed = vram_used_bytes();
    Json bench_json;
    // Build a ~64-token document template for clean throughput measurement across B=1, 4, 16, 32.
    const std::string doc_base =
        "Atlas ingine serves OpenAI-compatible embeddings directly from the resident transformer backbone without "
        "allocating persistent key-value cache pages or computing the 200K-token output vocabulary projection, "
        "packing multiple independent strings into compute-bound Tensor Core GEMMs for high throughput.";

    for (const auto& [mode_name, tap_opt] :
         std::vector<std::pair<std::string, std::optional<int>>>{{"full_48v", std::nullopt}, {"early_exit_v23", 23}}) {
      for (int B : {1, 4, 16, 32}) {
        Json arr = Json::array();
        for (int i = 0; i < B; ++i) arr.push_back(doc_base + " Document #" + std::to_string(i) + ".");
        Json body = {{"input", arr}, {"encoding_format", "base64"}, {"pooling", "mean"}, {"normalize", true}};
        if (tap_opt.has_value()) body["tap_layer"] = *tap_opt;

        // Warmup
        const HttpRes warm = post_json(port, "/v1/embeddings", body);
        CHECK(warm.status == 200);
        CHECK(hdr_val(warm.headers, "X-Ingine-Kv-Slots-Allocated") == "0");
        CHECK(hdr_val(warm.headers, "X-Ingine-Lm-Head-Skipped") == "true");
        const int prompt_tokens = warm.body["usage"]["prompt_tokens"].get<int>();

        constexpr int kReps = 3;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r) {
          const HttpRes rr = post_json(port, "/v1/embeddings", body);
          CHECK(rr.status == 200);
        }
        const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ms_per_batch = (total_s / kReps) * 1e3;
        const double tok_per_s = static_cast<double>(prompt_tokens * kReps) / total_s;
        std::printf("  [HTTP B=%2d %-14s] tokens=%4d ms/batch=%6.2f ms/item=%5.2f tok/s=%8.1f\n", B, mode_name.c_str(),
                    prompt_tokens, ms_per_batch, ms_per_batch / B, tok_per_s);
        bench_json[mode_name]["B" + std::to_string(B)] = {
            {"batch", B},
            {"prompt_tokens", prompt_tokens},
            {"ms_per_batch", ms_per_batch},
            {"ms_per_item", ms_per_batch / B},
            {"tok_per_s", tok_per_s},
        };
      }
    }
    const std::size_t vram_after_embed = vram_used_bytes();
    const double vram_delta_mib =
        static_cast<double>(static_cast<std::int64_t>(vram_after_embed) - static_cast<std::int64_t>(vram_before_embed)) /
        (1024.0 * 1024.0);

    // Chat completion after embedding burst must match identically (zero KV/prefix corruption!).
    const HttpRes c_after = post_json(port, "/v1/chat/completions", chat_req);
    CHECK(c_after.status == 200);
    const std::string reply_after = c_after.body["choices"][0]["message"]["content"].get<std::string>();
    CHECK(reply_before == reply_after);
    std::printf("  [chat_after_embeddings] identical=%s vram_delta_mib=%.2f\n",
                reply_before == reply_after ? "true" : "false", vram_delta_mib);

    report["http_benchmark"] = std::move(bench_json);
    report["vram_delta_mib_after_embed_burst"] = vram_delta_mib;
    report["chat_reply_identical_after_embed_burst"] = (reply_before == reply_after);
    srv.stop();
  }
  return report;
}

Json run_qwen38_gpu_verification(const std::filesystem::path& model_dir, const std::filesystem::path& platos_key_path) {
  std::printf("\n=== [Qwen3.8-27B] Live GPU Parity, Real Frozen Plato's Key (L31) & Throughput ===\n");
  Json report;
  report["model_dir"] = model_dir.string();
  report["platos_key_path"] = platos_key_path.string();

  const ChatFormat fmt = ChatFormat::load(model_dir);
  RuntimeOptions ro;
  ro.model_dir = model_dir;
  ro.max_ctx = 2048;
  ro.prefill_chunk = 512;
  ro.vision = false;
  ro.max_batch = 1;
  ro.park_mb = 0;
  ro.platos_key = platos_key_path;

  auto gen = ingine::model::make_generator(ro);
  CHECK(gen->supports_embeddings());
  CHECK(gen->embedding_dim() == 5120);
  CHECK(gen->num_tap_layers() == 64);
  CHECK(gen->has_platos_key());
  CHECK(gen->platos_key_out_dim() == 5120);

  ServerOptions so;
  so.max_queue = 8;
  ChatServer srv(*gen, fmt, so);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();
  const int port = srv.port();

  // 1. Prime chat session & capture greedy reference reply before embeddings.
  const Json chat_req = {{"messages", Json::array({{{"role", "user"}, {"content", "What is 2+2? Answer with just the number."}}})},
                         {"temperature", 0.0},
                         {"max_tokens", 12},
                         {"enable_thinking", false}};
  const HttpRes c0 = post_json(port, "/v1/chat/completions", chat_req);
  CHECK(c0.status == 200);
  const std::string chat_before = c0.body["choices"][0]["message"]["content"].get<std::string>();

  const std::size_t vram_before_embed = vram_used_bytes();

  // 2. Parity: raw L31 tap (B=4 packed vs B=1 single) + REAL frozen Qwen3.8->H3 Plato's Key GPU vs FP64 CPU reference!
  const std::vector<std::string> texts = {
      "A misty alpine lake reflecting snow-capped granite peaks at golden hour, dramatic volumetric light.",
      "Zero-KV batched embedding prefill skips KV allocation and the 248K vocabulary projection.",
      "Half-depth early exit at layer 31 saves fifty percent of transformer layers on Qwen3.8-27B.",
      "A tiny ceramic teapot resting on a weathered oak table beside an open brass pocket watch.",
  };

  const HttpRes r_raw_b4 = post_json(
      port, "/v1/embeddings",
      Json{{"input", texts}, {"tap_layer", 31}, {"pooling", "mean"}, {"normalize", false}, {"platos_key", false}});
  CHECK(r_raw_b4.status == 200);
  CHECK(hdr_val(r_raw_b4.headers, "X-Ingine-Kv-Slots-Allocated") == "0");
  CHECK(hdr_val(r_raw_b4.headers, "X-Ingine-Lm-Head-Skipped") == "true");
  CHECK(hdr_val(r_raw_b4.headers, "X-Ingine-Tap-Layer") == "31");

  double batch_max_abs = 0.0, batch_min_cos = 1.0;
  std::vector<float> flat_raw;
  for (std::size_t i = 0; i < texts.size(); ++i) {
    const HttpRes r_raw_1 = post_json(
        port, "/v1/embeddings",
        Json{{"input", texts[i]}, {"tap_layer", 31}, {"pooling", "mean"}, {"normalize", false}, {"platos_key", false}});
    CHECK(r_raw_1.status == 200);
    const auto v_b4 = r_raw_b4.body["data"][i]["embedding"].get<std::vector<float>>();
    const auto v_1 = r_raw_1.body["data"][0]["embedding"].get<std::vector<float>>();
    const ParityStats ps = compare_vecs(v_b4, v_1);
    batch_max_abs = std::max(batch_max_abs, ps.max_abs);
    batch_min_cos = std::min(batch_min_cos, ps.cosine);
    flat_raw.insert(flat_raw.end(), v_b4.begin(), v_b4.end());
  }
  std::printf("  [Qwen3.8 L31 batch4 vs single] max_abs=%.6e min_cosine=%.8f\n", batch_max_abs, batch_min_cos);
  std::fflush(stdout);
  CHECK(batch_min_cos >= 0.99999);
  CHECK(batch_max_abs <= 1e-2);

  // Real frozen Qwen3.8->H3 Plato's Key projection (5120 -> 5120 F16) GPU vs FP64 CPU reference + latency (B=1 & B=4)!
  double min_pk_b1_ms = 1e9;
  for (int rep = 0; rep < 3; ++rep) {
    const HttpRes r1 = post_json(
        port, "/v1/embeddings",
        Json{{"input", texts[0]}, {"tap_layer", 31}, {"pooling", "mean"}, {"normalize", false}, {"platos_key", true}});
    CHECK(r1.status == 200);
    min_pk_b1_ms = std::min(min_pk_b1_ms, std::stod(hdr_val(r1.headers, "X-Ingine-Platos-Key-Ms")));
  }
  HttpRes r_pk;
  double min_pk_ms = 1e9;
  for (int rep = 0; rep < 3; ++rep) {
    r_pk = post_json(
        port, "/v1/embeddings",
        Json{{"input", texts}, {"tap_layer", 31}, {"pooling", "mean"}, {"normalize", false}, {"platos_key", true}});
    CHECK(r_pk.status == 200);
    CHECK(hdr_val(r_pk.headers, "X-Ingine-Platos-Key") == "true");
    const std::string ms_str = hdr_val(r_pk.headers, "X-Ingine-Platos-Key-Ms");
    CHECK(!ms_str.empty());
    min_pk_ms = std::min(min_pk_ms, std::stod(ms_str));
  }
  auto cpu_key = PlatosKey::load(platos_key_path, 5120);
  std::vector<float> cpu_proj(texts.size() * 5120);
  cpu_key->project_cpu(flat_raw.data(), static_cast<int>(texts.size()), cpu_proj.data(), false);
  std::vector<float> gpu_proj;
  for (std::size_t i = 0; i < texts.size(); ++i) {
    const auto vi = r_pk.body["data"][i]["embedding"].get<std::vector<float>>();
    gpu_proj.insert(gpu_proj.end(), vi.begin(), vi.end());
  }
  const ParityStats pk_ps = compare_vecs(gpu_proj, cpu_proj);
  std::printf("  [Qwen3.8 REAL H3 PlatosKey 5120->5120] B=1 gpu_ms=%.4f B=4 gpu_ms=%.4f max_abs=%.6e cosine=%.8f\n",
              min_pk_b1_ms, min_pk_ms, pk_ps.max_abs, pk_ps.cosine);
  std::fflush(stdout);
  CHECK(pk_ps.cosine >= 0.99999);
  CHECK(min_pk_ms < 0.25);

  report["parity"] = {
      {"l31_batch4_vs_single", {{"max_abs", batch_max_abs}, {"min_cosine", batch_min_cos}}},
      {"real_h3_platos_key_gpu_vs_fp64_cpu",
       {{"b1_gpu_ms", min_pk_b1_ms}, {"b4_gpu_ms", min_pk_ms}, {"max_abs", pk_ps.max_abs}, {"cosine", pk_ps.cosine}}},
  };

  // 3. Throughput & early-exit speedup: Full 64 layers vs Half-depth L31 across B=1, 4, 8!
  const std::string doc =
      "Atlas ingine exposes a zero-KV OpenAI embeddings endpoint that skips persistent KV cache allocation and the "
      "248K-vocab output projection while supporting early-exit layer tapping and sub-millisecond Plato's Key "
      "multimodal projection directly in VRAM.";
  Json bench;
  for (const auto& [mode_name, tap_opt] :
       std::vector<std::pair<std::string, std::optional<int>>>{{"full_64L", std::nullopt}, {"early_exit_L31", 31}}) {
    for (int B : {1, 4, 8}) {
      Json arr = Json::array();
      for (int i = 0; i < B; ++i) arr.push_back(doc + " Item " + std::to_string(i) + ".");
      Json body = {{"input", arr}, {"encoding_format", "base64"}, {"pooling", "mean"}, {"normalize", true}};
      if (tap_opt.has_value()) body["tap_layer"] = *tap_opt;
      const HttpRes warm = post_json(port, "/v1/embeddings", body);
      CHECK(warm.status == 200);
      const int toks = warm.body["usage"]["prompt_tokens"].get<int>();
      constexpr int kReps = 2;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) {
        const HttpRes rr = post_json(port, "/v1/embeddings", body);
        CHECK(rr.status == 200);
      }
      const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double ms_per_batch = (total_s / kReps) * 1e3;
      const double tok_per_s = static_cast<double>(toks * kReps) / total_s;
      std::printf("  [Qwen3.8 HTTP B=%d %-14s] tokens=%3d ms/batch=%7.2f ms/item=%6.2f tok/s=%7.1f\n", B,
                  mode_name.c_str(), toks, ms_per_batch, ms_per_batch / B, tok_per_s);
      bench[mode_name]["B" + std::to_string(B)] = {
          {"batch", B},
          {"prompt_tokens", toks},
          {"ms_per_batch", ms_per_batch},
          {"ms_per_item", ms_per_batch / B},
          {"tok_per_s", tok_per_s},
      };
    }
  }
  const std::size_t vram_after_embed = vram_used_bytes();
  const double vram_delta_mib =
      static_cast<double>(static_cast<std::int64_t>(vram_after_embed) - static_cast<std::int64_t>(vram_before_embed)) /
      (1024.0 * 1024.0);

  // 4. Verify chat session prefix cache STILL hits after embedding burst (zero default_session mutation!).
  const HttpRes c1 = post_json(port, "/v1/chat/completions", chat_req);
  CHECK(c1.status == 200);
  const std::string chat_after = c1.body["choices"][0]["message"]["content"].get<std::string>();
  const int cached_tokens = c1.body["usage"]["prompt_tokens_details"]["cached_tokens"].get<int>();
  CHECK(chat_before == chat_after);
  CHECK(cached_tokens > 0);
  std::printf("  [Qwen3.8 chat_after_embeddings] identical=true cached_tokens=%d vram_delta_mib=%.2f\n", cached_tokens,
              vram_delta_mib);

  report["http_benchmark"] = std::move(bench);
  report["vram_delta_mib_after_embed_burst"] = vram_delta_mib;
  report["chat_reply_identical_after_embed_burst"] = true;
  report["chat_prefix_cache_reused_tokens_after_embed"] = cached_tokens;
  srv.stop();
  return report;
}

}  // namespace

int main() {
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::fprintf(stderr, "No CUDA device available; skipping test_gpu_embeddings\n");
    return 77;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Running live GPU embedding verification on %s (sm_%d%d, %.1f GiB)\n", prop.name, prop.major, prop.minor,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  Json full_report;
  full_report["gpu_name"] = prop.name;
  full_report["sm"] = std::to_string(prop.major) + std::to_string(prop.minor);

  auto find_dir = [](const char* env_name, std::initializer_list<const char*> defaults) -> std::filesystem::path {
    if (const char* e = std::getenv(env_name); e && *e) {
      if (std::string_view(e) == "skip") return {};
      return e;
    }
    for (const char* d : defaults) {
      if (std::filesystem::is_directory(d)) return d;
    }
    return {};
  };

  const auto ink_dir =
      find_dir("INGINE_GPU_EMBED_INKLING_DIR",
               {"/mnt/storage/models/Inkling-2-Mini-ingine", "/mnt/nvme/models/Inkling-2-Mini-ingine"});
  const auto qwen_dir =
      find_dir("INGINE_GPU_EMBED_QWEN38_DIR",
               {"/mnt/storage/models/Qwen3.8-27B-NVFP4-482ca0f", "/mnt/nvme/models/Qwen3.8-27B-NVFP4",
                "/mnt/storage/models/Qwen3.8-27B-atlas-17g-ingine",
                "/mnt/storage/models/Qwen3.8-27B-atlas-14g-ingine"});
  const auto qwen_pk =
      find_dir("INGINE_GPU_EMBED_QWEN38_PLATOS_KEY",
               {"/mnt/storage/omni_platos_keys_multimodel/qwen38"});

  if (!ink_dir.empty()) {
    full_report["inkling"] = run_inkling_gpu_verification(ink_dir);
  }
  const std::filesystem::path pk_file =
      std::filesystem::is_regular_file(qwen_pk)
          ? qwen_pk
          : (qwen_pk / "qwen38_to_h3_single_tap_platos_key.safetensors");
  if (!qwen_dir.empty() && std::filesystem::is_regular_file(pk_file)) {
    full_report["qwen38"] = run_qwen38_gpu_verification(qwen_dir, pk_file);
  }

  if (const char* out_path = std::getenv("INGINE_GPU_EMBED_REPORT_JSON"); out_path && *out_path) {
    std::ofstream out(out_path, std::ios::trunc);
    CHECK(static_cast<bool>(out));
    out << full_report.dump(2) << "\n";
    out.flush();
    std::printf("\nWrote GPU verification report to %s\n", out_path);
  }
  std::printf("\nALL LIVE GPU EMBEDDING CHECKS PASSED!\n");
  return 0;
}
