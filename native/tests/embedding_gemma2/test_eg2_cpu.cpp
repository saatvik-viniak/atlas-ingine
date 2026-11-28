// Phase 1 CPU unit & real-checkpoint oracle tests for google/embeddinggemma-2 (#968):
//   1) ModelFamily::EmbeddingGemma2 detection & parsing
//   2) GemmaBpe tokenizer exact token-ID & decode parity vs HuggingFace tokenizers
//   3) Eg2Config + selective safetensors manifest audit (text vs vision vs audio bytes)
//   4) Synthetic 2-layer Eg2CpuEncoder: sliding + full attn, PLE, layer_scalar, Matryoshka, tap_layer, PlatosKey
//   5) End-to-end HTTP ChatServer with EmbeddingGemma2 (prompt_name, dimensions, encoder-only 400, /health, /v1/models)
//   6) Real checkpoint FP32 CPU oracle parity (when /mnt/storage/models/embeddinggemma-2 is present)
#include "backends/embedding_gemma2/eg2_audio.h"
#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_cpu.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "core/sha256.h"
#include "engine/fake_generator.h"
#include "ingine/model_backend.h"
#include "ingine/platos_key.h"
#include "serve/chat_server.h"
#include "serve/openai.h"
#include "text/chat.h"
#include "text/tokenizer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

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

double dot_f32(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size() && !a.empty());
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return s;
}

double l2_norm(const std::vector<float>& v) { return std::sqrt(dot_f32(v, v)); }

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
  return dot_f32(a, b) / (l2_norm(a) * l2_norm(b));
}

std::string base64_encode(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kAlpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((bytes.size() + 2) / 3) * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const std::uint32_t b0 = bytes[i];
    const std::uint32_t b1 = (i + 1 < bytes.size()) ? bytes[i + 1] : 0u;
    const std::uint32_t b2 = (i + 2 < bytes.size()) ? bytes[i + 2] : 0u;
    const std::uint32_t t = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(kAlpha[(t >> 18) & 0x3fu]);
    out.push_back(kAlpha[(t >> 12) & 0x3fu]);
    out.push_back(i + 1 < bytes.size() ? kAlpha[(t >> 6) & 0x3fu] : '=');
    out.push_back(i + 2 < bytes.size() ? kAlpha[t & 0x3fu] : '=');
  }
  return out;
}

// Builds a minimal valid uncompressed 24-bit RGB BMP data URI (W x H) for CPU request-validation tests.
std::string make_bmp_data_uri(int width, int height) {
  const int row_stride = ((width * 3) + 3) & ~3;
  const std::uint32_t pix_bytes = static_cast<std::uint32_t>(row_stride * height);
  const std::uint32_t file_bytes = 54u + pix_bytes;
  std::vector<std::uint8_t> bmp(file_bytes, 0u);
  bmp[0] = 'B';
  bmp[1] = 'M';
  std::memcpy(bmp.data() + 2, &file_bytes, 4);
  const std::uint32_t pix_off = 54u;
  std::memcpy(bmp.data() + 10, &pix_off, 4);
  const std::uint32_t dib_sz = 40u;
  std::memcpy(bmp.data() + 14, &dib_sz, 4);
  const std::int32_t w32 = width, h32 = height;
  std::memcpy(bmp.data() + 18, &w32, 4);
  std::memcpy(bmp.data() + 22, &h32, 4);
  const std::uint16_t planes = 1u, bpp = 24u;
  std::memcpy(bmp.data() + 26, &planes, 2);
  std::memcpy(bmp.data() + 28, &bpp, 2);
  std::memcpy(bmp.data() + 34, &pix_bytes, 4);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t off = 54u + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride) +
                              static_cast<std::size_t>(x) * 3u;
      bmp[off + 0] = static_cast<std::uint8_t>((x * 7 + y) & 0xff);
      bmp[off + 1] = static_cast<std::uint8_t>((x + y * 5) & 0xff);
      bmp[off + 2] = static_cast<std::uint8_t>((x * 3 + y * 3) & 0xff);
    }
  }
  return "data:image/bmp;base64," + base64_encode(bmp);
}

// Writes a valid F32 Plato's Key R [d_in, d_out] (+ optional donor_mean, target_mean, scale) safetensors file
// and returns its exact lowercase 64-hex SHA-256.
std::string write_f32_platos_key(const std::filesystem::path& path, int d_in, int d_out, int seed,
                                 const std::map<std::string, std::string>& meta = {}, bool with_affine = true) {
  std::vector<float> R(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_out));
  for (std::size_t i = 0; i < R.size(); ++i) {
    R[i] = static_cast<float>(std::sin(static_cast<double>(i + 1) * 0.137 + static_cast<double>(seed) * 0.71)) * 0.25f;
  }
  std::vector<float> dm(static_cast<std::size_t>(d_in)), tm(static_cast<std::size_t>(d_out));
  for (int i = 0; i < d_in; ++i) dm[i] = static_cast<float>(std::cos(static_cast<double>(i + seed) * 0.19)) * 0.05f;
  for (int j = 0; j < d_out; ++j) tm[j] = static_cast<float>(std::sin(static_cast<double>(j + seed) * 0.31)) * 0.1f;
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
  if (with_affine) {
    add_tensor("donor_mean", {d_in}, dm);
    add_tensor("target_mean", {d_out}, tm);
    add_tensor("scale", {1}, sc);
  }
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

std::filesystem::path find_eg2_model_dir() {
  if (const char* env = std::getenv("INGINE_EG2_MODEL_DIR"); env && *env) {
    std::filesystem::path p(env);
    if (std::filesystem::is_regular_file(p / "model.safetensors")) return p;
  }
  const std::filesystem::path def("/mnt/storage/models/embeddinggemma-2");
  if (std::filesystem::is_regular_file(def / "model.safetensors")) return def;
  return {};
}

// Build deterministic synthetic 2-layer Eg2TextWeightsF32 (1 sliding + 1 full layer) for fast unit tests.
Eg2TextWeightsF32 make_synthetic_weights() {
  Eg2TextWeightsF32 w;
  Eg2Config& c = w.config;
  c.vocab_size = 64;
  c.hidden_size = 16;
  c.intermediate_size = 32;
  c.hidden_size_per_layer_input = 16;
  c.num_hidden_layers = 2;
  c.num_attention_heads = 2;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.sliding_window = 2;
  c.embedding_dim = 24;
  c.matryoshka_dims = {24, 16, 8};
  c.prompts = {{"SearchQuery", "task: search result | query: "}, {"Document", "title: none | text: "}};
  c.layers = {
      LayerSpec{LayerAttnKind::Sliding, 2, 2, 8, 10000.0},
      LayerSpec{LayerAttnKind::Full, 2, 1, 16, 1000000.0},
  };
  c.validate();

  auto fill = [](std::vector<float>& dst, std::size_t n, int seed, float scale) {
    dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double x = std::sin(static_cast<double>(i + 1) * 0.173 + static_cast<double>(seed) * 1.131);
      dst[i] = static_cast<float>(x * scale);
    }
  };
  auto fill_norm = [](std::vector<float>& dst, std::size_t n, int seed) {
    dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = 1.0f + 0.1f * static_cast<float>(std::cos(static_cast<double>(i + seed)));
    }
  };

  fill(w.embed_tokens, 64 * 16, 1, 0.2f);
  fill(w.per_layer_model_projection, 2 * 16 * 16, 2, 0.1f);
  fill_norm(w.per_layer_projection_norm, 16, 3);
  fill_norm(w.norm, 16, 4);
  fill(w.embedding_projection, 24 * 16, 5, 0.15f);

  w.layers.resize(2);
  for (int li = 0; li < 2; ++li) {
    auto& lw = w.layers[static_cast<std::size_t>(li)];
    const LayerSpec& ls = c.layers[static_cast<std::size_t>(li)];
    const int qd = ls.num_attention_heads * ls.head_dim;
    const int kvd = ls.num_key_value_heads * ls.head_dim;
    const int b = (li + 1) * 100;
    fill_norm(lw.input_layernorm, 16, b + 1);
    fill(lw.q_proj, qd * 16, b + 2, 0.1f);
    fill(lw.k_proj, kvd * 16, b + 3, 0.1f);
    fill(lw.v_proj, kvd * 16, b + 4, 0.1f);
    fill(lw.o_proj, 16 * qd, b + 5, 0.1f);
    fill_norm(lw.q_norm, ls.head_dim, b + 6);
    fill_norm(lw.k_norm, ls.head_dim, b + 7);
    fill_norm(lw.post_attention_layernorm, 16, b + 8);
    fill_norm(lw.pre_feedforward_layernorm, 16, b + 9);
    fill(lw.gate_proj, 32 * 16, b + 10, 0.1f);
    fill(lw.up_proj, 32 * 16, b + 11, 0.1f);
    fill(lw.down_proj, 16 * 32, b + 12, 0.1f);
    fill_norm(lw.post_feedforward_layernorm, 16, b + 13);
    fill(lw.per_layer_input_gate, 16 * 16, b + 14, 0.1f);
    fill(lw.per_layer_projection, 16 * 16, b + 15, 0.1f);
    fill_norm(lw.post_per_layer_input_norm, 16, b + 16);
    lw.layer_scalar = (li == 0) ? 0.39f : 0.21f;
  }
  return w;
}

void test_model_family_enum() {
  using namespace ingine::model;
  CHECK(std::string(to_string(ModelFamily::EmbeddingGemma2)) == "embedding_gemma2");
  CHECK(parse_model_family("embedding_gemma2") == ModelFamily::EmbeddingGemma2);
  CHECK(parse_model_family("EmbeddingGemma-2") == ModelFamily::EmbeddingGemma2);
  CHECK(parse_model_family("embeddinggemma2") == ModelFamily::EmbeddingGemma2);
  std::printf("[PASS] test_model_family_enum\n");
}

void test_synthetic_cpu_encoder_and_matryoshka() {
  Eg2CpuBackend backend(make_synthetic_weights(), 128);
  CHECK(backend.is_encoder_only());
  CHECK(backend.supports_embeddings());
  CHECK(backend.embedding_dim() == 24);
  CHECK(backend.num_tap_layers() == 2);
  CHECK(backend.matryoshka_dims() == (std::vector<int>{24, 16, 8}));

  // generate() must fail closed on encoder-only backend
  bool threw_gen = false;
  try {
    std::vector<std::int32_t> p = {2, 5, 1};
    (void)backend.generate(p, 4, ingine::model::Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });
  } catch (const std::invalid_argument&) {
    threw_gen = true;
  }
  CHECK(threw_gen);

  // Unnormalized 24d -> Matryoshka 16d / 8d must equal prefix slice + FP32 L2 re-normalization!
  ingine::model::EmbeddingRequest req_raw;
  req_raw.inputs = {{2, 5, 11, 17, 1}, {2, 7, 13, 1}};
  req_raw.normalize = false;
  const auto res_raw = backend.embed(req_raw);
  CHECK(res_raw.embeddings.size() == 2);
  CHECK(res_raw.stats.output_dim == 24);
  CHECK(res_raw.stats.kv_slots_allocated == 0);
  CHECK(res_raw.stats.lm_head_skipped);

  for (int mrl_dim : {24, 16, 8}) {
    ingine::model::EmbeddingRequest r = req_raw;
    r.normalize = true;
    r.dimensions = mrl_dim;
    const auto out = backend.embed(r);
    CHECK(out.stats.output_dim == mrl_dim);
    for (int b = 0; b < 2; ++b) {
      CHECK(static_cast<int>(out.embeddings[b].size()) == mrl_dim);
      CHECK(std::abs(l2_norm(out.embeddings[b]) - 1.0) < 1e-5);
      std::vector<float> expected(res_raw.embeddings[b].begin(), res_raw.embeddings[b].begin() + mrl_dim);
      const double nrm = l2_norm(expected);
      for (int d = 0; d < mrl_dim; ++d) {
        CHECK(std::abs(out.embeddings[b][d] - static_cast<float>(expected[d] / nrm)) < 1e-5f);
      }
    }
  }

  // Invalid Matryoshka dim must throw fail-closed
  bool threw_dim = false;
  try {
    ingine::model::EmbeddingRequest r = req_raw;
    r.dimensions = 12;
    (void)backend.embed(r);
  } catch (const std::invalid_argument&) {
    threw_dim = true;
  }
  CHECK(threw_dim);

  // Early-exit tap_layer=0 outputs hidden_size=16 (different from layer 1)
  ingine::model::EmbeddingRequest r0 = req_raw;
  r0.tap_layer = 0;
  r0.normalize = true;
  const auto t0 = backend.embed(r0);
  CHECK(t0.stats.effective_tap_layer == 0);
  CHECK(t0.stats.output_dim == 16);
  CHECK(std::abs(l2_norm(t0.embeddings[0]) - 1.0) < 1e-5);

  // Batch invariance: B=1 vs B=2 must match bit-for-bit on CPU
  ingine::model::EmbeddingRequest r_one;
  r_one.inputs = {{2, 7, 13, 1}};
  r_one.normalize = false;
  const auto res_one = backend.embed(r_one);
  for (int d = 0; d < 24; ++d) {
    CHECK(std::abs(res_one.embeddings[0][d] - res_raw.embeddings[1][d]) < 1e-6f);
  }
  std::printf("[PASS] test_synthetic_cpu_encoder_and_matryoshka\n");
}

void test_real_checkpoint_tokenizer_manifest_http_and_oracle(const std::filesystem::path& model_dir) {
  // 1) detect_model_family
  CHECK(ingine::model::detect_model_family(model_dir) == ingine::model::ModelFamily::EmbeddingGemma2);

  // 2) Manifest selective inspection
  const Eg2ManifestSummary ms = inspect_eg2_manifest(model_dir);
  CHECK(ms.total_tensors == 1376);
  CHECK(ms.text_tensors == 413);
  CHECK(ms.vision_tensors == 211);
  CHECK(ms.audio_tensors == 752);
  CHECK(ms.vision_present && ms.audio_present);
  CHECK(ms.config.embedding_dim == 768);
  CHECK(ms.config.num_hidden_layers == 24);
  CHECK(ms.config.layers[5].kind == LayerAttnKind::Full && ms.config.layers[5].head_dim == 512 &&
        ms.config.layers[5].num_key_value_heads == 1);
  CHECK(ms.config.layers[0].kind == LayerAttnKind::Sliding && ms.config.layers[0].head_dim == 256 &&
        ms.config.layers[0].num_key_value_heads == 2);
  CHECK(ms.config.find_prompt_prefix("SearchQuery") == std::optional<std::string_view>("task: search result | query: "));
  CHECK(ms.config.find_prompt_prefix("Document") == std::optional<std::string_view>("title: none | text: "));
  std::printf("[PASS] real manifest: text=%zu (%.2f MiB), vision=%zu (%.2f MiB), audio=%zu (%.2f MiB)\n",
              ms.text_tensors, double(ms.text_bytes) / (1024.0 * 1024.0), ms.vision_tensors,
              double(ms.vision_bytes) / (1024.0 * 1024.0), ms.audio_tensors,
              double(ms.audio_bytes) / (1024.0 * 1024.0));

  // 3) GemmaBpe exact token-ID parity vs HuggingFace tokenizers
  const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
  CHECK(fmt.kind() == ingine::text::ChatFormatKind::EmbeddingGemma2);
  const ingine::text::Tokenizer& tok = fmt.tokenizer();
  CHECK(tok.split_mode() == ingine::text::Tokenizer::SplitMode::GemmaBpe);
  CHECK(tok.byte_fallback());
  CHECK(tok.bos_token_id() == std::optional<int32_t>(2));
  CHECK(tok.eos_token_id() == std::optional<int32_t>(1));

  struct TokCase {
    std::string text;
    bool special;
    std::vector<int32_t> expected_no_pp;
    std::vector<int32_t> expected_with_pp;
  };
  const std::vector<TokCase> cases = {
      {"", true, {}, {2, 1}},
      {"hello", true, {23391}, {2, 23391, 1}},
      {" hello", true, {29104}, {2, 29104, 1}},
      {"hello world", true, {23391, 1902}, {2, 23391, 1902, 1}},
      {"hello  world", true, {23391, 138, 12392}, {2, 23391, 138, 12392, 1}},
      {"hello   world", true, {23391, 139, 12392}, {2, 23391, 139, 12392, 1}},
      {"a\nb\tc", true, {236746, 107, 236763, 255968, 236755}, {2, 236746, 107, 236763, 255968, 236755, 1}},
      {"caf\xC3\xA9 \xF0\x9F\x9A\x80 \xE4\xBD\xA0\xE5\xA5\xBD",
       true,
       {123125, 236859, 236743, 242015, 43758, 237389},
       {2, 123125, 236859, 236743, 242015, 43758, 237389, 1}},
      {"\xF0\x90\x8C\x80", true, {478, 382, 378, 366}, {2, 478, 382, 378, 366, 1}},  // byte_fallback U+10300
      {"task: search result | query: What causes the northern lights?",
       false,
       {8071, 236787, 3927, 1354, 1109, 7609, 236787, 2900, 9454, 506, 16962, 11544, 236881},
       {2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 2900, 9454, 506, 16962, 11544, 236881, 1}},
      {"title: none | text: The northern lights are caused by charged particles from the sun.",
       false,
       {3250, 236787, 7293, 1109, 1816, 236787, 669, 16962, 11544, 659, 7321, 684, 11055, 10390, 699, 506, 3768, 236761},
       {2, 3250, 236787, 7293, 1109, 1816, 236787, 669, 16962, 11544, 659, 7321, 684, 11055, 10390, 699, 506, 3768, 236761, 1}},
      {"a <|image|> b", true, {236746, 236743, 258880, 518}, {2, 236746, 236743, 258880, 518, 1}},
      {"<|image><|image|><|image|><image|>", true, {255999, 258880, 258880, 258882}, {2, 255999, 258880, 258880, 258882, 1}},
      {"<|audio><|audio|><audio|>", true, {256000, 258881, 258883}, {2, 256000, 258881, 258883, 1}},
  };
  for (const auto& tc : cases) {
    const auto got = tok.encode(tc.text, tc.special);
    CHECK(got == tc.expected_no_pp);
    const auto got_pp = tok.encode_with_post_processor(tc.text, tc.special);
    CHECK(got_pp == tc.expected_with_pp);
    const std::string dec = tok.decode(got);
    CHECK(dec == tc.text);
  }
  std::printf("[PASS] GemmaBpe tokenizer exact HF token-ID & decode parity (%zu/14 cases)\n", cases.size());

  // 4) Real checkpoint CPU FP32 backend + HTTP ChatServer end-to-end test
  ingine::model::RuntimeOptions opts;
  opts.model_dir = model_dir;
  opts.max_ctx = 8192;
  Eg2CpuBackend real_backend(opts);
  ingine::serve::ServerOptions sopts;
  sopts.max_model_len = 8192;
  ingine::serve::ChatServer server(real_backend, fmt, sopts);
  CHECK(server.bind("127.0.0.1", 0));
  const int port = server.port();
  server.start();

  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(120, 0);

  // GET /health and GET /v1/models
  {
    auto r = cli.Get("/health");
    CHECK(r && r->status == 200);
    const Json h = Json::parse(r->body);
    CHECK(h["model_family"] == "embedding_gemma2");
    CHECK(h["supports_embeddings"] == true);
    CHECK(h["encoder_only"] == true);
    CHECK(h["embedding_dim"] == 768);
    CHECK(h["num_tap_layers"] == 24);
    CHECK(h["matryoshka_dims"] == Json::array({768, 512, 256, 128}));

    auto m = cli.Get("/v1/models");
    CHECK(m && m->status == 200);
    const Json mj = Json::parse(m->body);
    CHECK(mj["data"][0]["encoder_only"] == true);
    CHECK(mj["data"][0]["input_modalities"] == Json::array({"text"}));
    CHECK(mj["data"][0]["supported_prompt_names"].is_array());
  }

  // POST /v1/chat/completions must fail closed with 400 (encoder-only)
  {
    auto r = cli.Post("/v1/chat/completions",
                      R"({"model":"embeddinggemma-2","messages":[{"role":"user","content":"hi"}]})",
                      "application/json");
    CHECK(r && r->status == 400);
  }

  // Invalid prompt_name and invalid dimensions must fail closed with 400
  {
    auto r1 = cli.Post("/v1/embeddings", R"({"input":"hi","prompt_name":"NoSuchTask"})", "application/json");
    CHECK(r1 && r1->status == 400);
    auto r2 = cli.Post("/v1/embeddings", R"({"input":"hi","dimensions":300})", "application/json");
    CHECK(r2 && r2->status == 400);
  }

  // Real semantic retrieval + Matryoshka oracle check via HTTP POST /v1/embeddings
  auto embed_http = [&](const std::string& text, const std::string& pname, int dim) -> std::vector<float> {
    Json req{{"input", text}, {"prompt_name", pname}, {"dimensions", dim}, {"normalize", true}};
    auto r = cli.Post("/v1/embeddings", req.dump(), "application/json");
    CHECK(r && r->status == 200);
    CHECK(r->get_header_value("X-Ingine-Embedding-Dim") == std::to_string(dim));
    CHECK(r->get_header_value("X-Ingine-Kv-Slots-Allocated") == "0");
    const Json resp = Json::parse(r->body);
    CHECK(resp["object"] == "list");
    const auto vec = resp["data"][0]["embedding"].get<std::vector<float>>();
    CHECK(static_cast<int>(vec.size()) == dim);
    CHECK(std::abs(l2_norm(vec) - 1.0) < 1e-4);
    return vec;
  };

  // Exact PyTorch FP32 oracle expectations across all 4 Matryoshka dims
  struct GoldenMrl {
    int dim;
    double cos_m;
    double cos_u;
  };
  const GoldenMrl goldens[] = {
      {768, 0.8707120, 0.5477407},
      {512, 0.8738954, 0.5540743},
      {256, 0.8907856, 0.5476265},
      {128, 0.9186663, 0.6453618},
  };
  const std::vector<float> expected_q_first8 = {
      0.0124939f, -0.0027802f, 0.0401911f, -0.0182558f, -0.0405981f, 0.0108970f, -0.0110940f, -0.0101309f};

  for (const auto& g : goldens) {
    const auto vq = embed_http("What causes the northern lights?", "SearchQuery", g.dim);
    const auto vm = embed_http("The northern lights are caused by charged particles from the sun.", "Document", g.dim);
    const auto vu = embed_http("A recipe for chocolate chip cookies uses butter, sugar, and flour.", "Document", g.dim);
    const double cm = cosine_sim(vq, vm);
    const double cu = cosine_sim(vq, vu);
    std::printf("[ORACLE CPU FP32] dim=%d cos(q,match)=%.7f cos(q,unrelated)=%.7f margin=%+.7f\n", g.dim, cm, cu, cm - cu);
    CHECK(std::abs(cm - g.cos_m) < 5e-5);
    CHECK(std::abs(cu - g.cos_u) < 5e-5);
    if (g.dim == 768) {
      for (std::size_t i = 0; i < expected_q_first8.size(); ++i) {
        CHECK(std::abs(vq[i] - expected_q_first8[i]) < 1e-5f);
      }
    }
  }

  server.stop();
  std::printf("[PASS] real checkpoint CPU FP32 oracle & live HTTP ChatServer verified\n");

  // 5) Finding (1): EG2 intermediate (512-d) & final (768-d) Plato's Key HTTP binding + mismatched controls
  {
    const auto tmp = std::filesystem::temp_directory_path() / "ingine_eg2_cpu_platos_key_968";
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    // Baseline unkeyed tap_layer=2 (512d) and final (768d) unnormalized embeddings for exact analytical checks
    const auto ids_q = tok.encode_with_post_processor("What causes the northern lights?", false);
    ingine::model::EmbeddingRequest r_tap2;
    r_tap2.inputs = {ids_q};
    r_tap2.tap_layer = 2;
    r_tap2.normalize = false;
    const auto ref_tap2 = real_backend.embed(r_tap2).embeddings[0];
    CHECK(ref_tap2.size() == 512);

    ingine::model::EmbeddingRequest r_fin;
    r_fin.inputs = {ids_q};
    r_fin.normalize = false;
    const auto ref_fin = real_backend.embed(r_fin).embeddings[0];
    CHECK(ref_fin.size() == 768);

    // (a) Valid F32 [512, 2] R cartridge bound to visit_2 (matches astartes-maxima's exact repro!)
    const auto k512_path = tmp / "eg2_512x2_visit2.safetensors";
    const auto k512_meta = tmp / "eg2_512x2_visit2.json";
    const std::string sha512 = write_f32_platos_key(k512_path, 512, 2, 11, {{"format", "platos_key_v1"}});
    {
      std::ofstream mf(k512_meta);
      mf << Json{{"cartridge_id", "eg2-visit2-512x2"},
                 {"cartridge_sha256", sha512},
                 {"donor_model_id", "google/embeddinggemma-2"},
                 {"tap_spec", "visit_2"},
                 {"d_donor", 512},
                 {"d_target", 2}}
                .dump();
    }
    {
      ingine::model::RuntimeOptions o512 = opts;
      o512.platos_key = k512_path;
      o512.platos_key_metadata = k512_meta;
      Eg2CpuBackend b512(o512);
      CHECK(b512.tap_embedding_dim() == 512);
      CHECK(b512.embedding_dim() == 768);

      // ChatServer with mismatched --model-id override must STILL fail closed!
      bool threw_override = false;
      try {
        ingine::serve::ServerOptions bad_sopts = sopts;
        bad_sopts.model_id = "wrong-donor-model";
        ingine::serve::ChatServer bad_srv(b512, fmt, bad_sopts);
      } catch (const std::invalid_argument&) {
        threw_override = true;
      }
      CHECK(threw_override);

      ingine::serve::ChatServer s512(b512, fmt, sopts);
      CHECK(s512.bind("127.0.0.1", 0));
      s512.start();
      httplib::Client c512("127.0.0.1", s512.port());
      c512.set_read_timeout(120, 0);

      auto h = c512.Get("/health");
      CHECK(h && h->status == 200);
      const Json hj = Json::parse(h->body);
      CHECK(hj["platos_key_loaded"] == true && hj["platos_key_out_dim"] == 2 && hj["platos_key_tap_layer"] == 2);

      // Positive HTTP 200 at tap_layer=2 with cartridge pins
      const Json req_ok{{"input", "What causes the northern lights?"},
                        {"tap_layer", 2},
                        {"platos_key", true},
                        {"cartridge_id", "eg2-visit2-512x2"},
                        {"cartridge_sha256", sha512},
                        {"normalize", false}};
      auto r_ok = c512.Post("/v1/embeddings", req_ok.dump(), "application/json");
      CHECK(r_ok && r_ok->status == 200);
      const auto got_raw = Json::parse(r_ok->body)["data"][0]["embedding"].get<std::vector<float>>();
      CHECK(got_raw.size() == 2);
      std::vector<float> exp_raw(2, 0.0f);
      b512.platos_key()->project_cpu(ref_tap2.data(), 1, exp_raw.data(), false);
      for (int j = 0; j < 2; ++j) CHECK(std::abs(got_raw[j] - exp_raw[j]) < 1e-5f);

      // Mismatched HTTP controls -> 400
      auto r_notap = c512.Post("/v1/embeddings", R"({"input":"hi","platos_key":true})", "application/json");
      CHECK(r_notap && r_notap->status == 400 && Json::parse(r_notap->body)["error"]["param"] == "tap_layer");
      auto r_badtap = c512.Post("/v1/embeddings", R"({"input":"hi","tap_layer":3,"platos_key":true})", "application/json");
      CHECK(r_badtap && r_badtap->status == 400 && Json::parse(r_badtap->body)["error"]["param"] == "tap_layer");
      auto r_badid = c512.Post("/v1/embeddings",
                               R"({"input":"hi","tap_layer":2,"platos_key":true,"cartridge_id":"wrong"})",
                               "application/json");
      CHECK(r_badid && r_badid->status == 400 && Json::parse(r_badid->body)["error"]["param"] == "cartridge_id");
      auto r_badsha = c512.Post("/v1/embeddings",
                                Json{{"input", "hi"},
                                     {"tap_layer", 2},
                                     {"platos_key", true},
                                     {"cartridge_sha256", std::string(64, '0')}}
                                    .dump(),
                                "application/json");
      CHECK(r_badsha && r_badsha->status == 400 && Json::parse(r_badsha->body)["error"]["param"] == "cartridge_sha256");
      s512.stop();
    }

    // (b) Valid F32 [768, 2] R cartridge bound to final_normed
    const auto k768_path = tmp / "eg2_768x2_final.safetensors";
    const auto k768_meta = tmp / "eg2_768x2_final.json";
    const std::string sha768 = write_f32_platos_key(k768_path, 768, 2, 23, {{"format", "platos_key_v1"}});
    {
      std::ofstream mf(k768_meta);
      mf << Json{{"cartridge_id", "eg2-final-768x2"},
                 {"cartridge_sha256", sha768},
                 {"donor_model_id", "google/embeddinggemma-2"},
                 {"tap_spec", "final_normed"},
                 {"d_donor", 768},
                 {"d_target", 2}}
                .dump();
    }
    {
      ingine::model::RuntimeOptions o768 = opts;
      o768.platos_key = k768_path;
      o768.platos_key_metadata = k768_meta;
      Eg2CpuBackend b768(o768);
      ingine::serve::ChatServer s768(b768, fmt, sopts);
      CHECK(s768.bind("127.0.0.1", 0));
      s768.start();
      httplib::Client c768("127.0.0.1", s768.port());
      c768.set_read_timeout(120, 0);

      const Json req_ok{{"input", "What causes the northern lights?"},
                        {"platos_key", true},
                        {"cartridge_id", "eg2-final-768x2"},
                        {"cartridge_sha256", sha768},
                        {"normalize", false}};
      auto r_ok = c768.Post("/v1/embeddings", req_ok.dump(), "application/json");
      CHECK(r_ok && r_ok->status == 200);
      const auto got_raw = Json::parse(r_ok->body)["data"][0]["embedding"].get<std::vector<float>>();
      CHECK(got_raw.size() == 2);
      std::vector<float> exp_raw(2, 0.0f);
      b768.platos_key()->project_cpu(ref_fin.data(), 1, exp_raw.data(), false);
      for (int j = 0; j < 2; ++j) CHECK(std::abs(got_raw[j] - exp_raw[j]) < 1e-5f);

      // Final-bound key must reject intermediate tap_layer with HTTP 400
      auto r_badtap = c768.Post("/v1/embeddings", R"({"input":"hi","tap_layer":2,"platos_key":true})", "application/json");
      CHECK(r_badtap && r_badtap->status == 400 && Json::parse(r_badtap->body)["error"]["param"] == "tap_layer");
      s768.stop();
    }

    // (c) Unbound [512, 2] key: HTTP 200 with tap_layer=2, HTTP 400 (NOT 500) when omitting tap_layer
    {
      ingine::model::RuntimeOptions ou = opts;
      ou.platos_key = k512_path;
      Eg2CpuBackend bu(ou);
      ingine::serve::ChatServer su(bu, fmt, sopts);
      CHECK(su.bind("127.0.0.1", 0));
      su.start();
      httplib::Client cu("127.0.0.1", su.port());
      cu.set_read_timeout(120, 0);
      auto r_ok = cu.Post("/v1/embeddings", R"({"input":"hi","tap_layer":2,"platos_key":true})", "application/json");
      CHECK(r_ok && r_ok->status == 200);
      auto r_400 = cu.Post("/v1/embeddings", R"({"input":"hi","platos_key":true})", "application/json");
      CHECK(r_400 && r_400->status == 400 && Json::parse(r_400->body)["error"]["param"] == "tap_layer");
      su.stop();
    }

    // (d) Mismatched startup controls: ALL must throw fail-closed!
    auto expect_startup_reject = [&](const std::filesystem::path& kp, const std::filesystem::path& mp) {
      bool threw = false;
      try {
        ingine::model::RuntimeOptions ob = opts;
        ob.platos_key = kp;
        ob.platos_key_metadata = mp;
        Eg2CpuBackend bb(ob);
        ingine::serve::ChatServer sb(bb, fmt, sopts);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    };
    // 768-d key bound to intermediate visit_2 -> REJECT!
    const auto m_768_visit2 = tmp / "bad_768_visit2.json";
    {
      std::ofstream mf(m_768_visit2);
      mf << Json{{"cartridge_sha256", sha768}, {"donor_model_id", "google/embeddinggemma-2"}, {"tap_spec", "visit_2"}}.dump();
    }
    expect_startup_reject(k768_path, m_768_visit2);
    // 512-d key bound to final_normed -> REJECT!
    const auto m_512_final = tmp / "bad_512_final.json";
    {
      std::ofstream mf(m_512_final);
      mf << Json{{"cartridge_sha256", sha512}, {"donor_model_id", "google/embeddinggemma-2"}, {"tap_spec", "final_normed"}}.dump();
    }
    expect_startup_reject(k512_path, m_512_final);
    // 256-d unbound key -> REJECT!
    const auto k256_path = tmp / "bad_256.safetensors";
    (void)write_f32_platos_key(k256_path, 256, 2, 31);
    expect_startup_reject(k256_path, {});
    // 512-d visit_2 key bound to wrong donor (Inkling-2-Mini) -> REJECT!
    const auto m_wrong_donor = tmp / "bad_wrong_donor.json";
    {
      std::ofstream mf(m_wrong_donor);
      mf << Json{{"cartridge_sha256", sha512}, {"donor_model_id", "Inkling-2-Mini"}, {"tap_spec", "visit_2"}}.dump();
    }
    expect_startup_reject(k512_path, m_wrong_donor);

    std::filesystem::remove_all(tmp);
    std::printf("[PASS] EG2 intermediate (512d) + final (768d) Plato's Key HTTP & mismatched controls verified\n");
  }

  // 6) Finding (3): Public embeddings parser limits (max_images, max_total_tokens incl. video frames across batch)
  //    + integer narrowing / representability rejection BEFORE conversion to int!
  {
    const std::string bmp_uri = make_bmp_data_uri(48, 48);
    // For 48x48 square image: max_soft_tokens=70 -> 384x384 -> 64 soft tokens; max_soft_tokens=140 -> 528x528 -> 121;
    // max_soft_tokens=280 -> 768x768 -> 256 soft tokens.
    ingine::serve::EmbeddingParseOptions popts;
    popts.max_model_len = 8192;
    popts.embedding_dim = 768;
    popts.tap_embedding_dim = 512;
    popts.num_tap_layers = 24;
    popts.supports_images = true;
    popts.supports_audio = true;
    popts.matryoshka_dims = {768, 512, 256, 128};
    popts.max_images = 2;
    popts.max_total_tokens = 150;

    auto expect_bad_req = [&](const Json& body, const std::string& expected_param,
                              const ingine::serve::EmbeddingParseOptions& o) {
      bool threw = false;
      try {
        (void)ingine::serve::parse_embedding_request(body, fmt, o);
      } catch (const ingine::serve::BadRequest& e) {
        CHECK(e.status == 400);
        if (!expected_param.empty()) CHECK(e.param.has_value() && *e.param == expected_param);
        threw = true;
      }
      CHECK(threw);
    };

    // Single image (64 soft tokens @ max_soft_tokens=70): accepted under max_images=2, max_total_tokens=150
    {
      const Json ok_img{{"input", Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 70}}})}};
      const auto parsed = ingine::serve::parse_embedding_request(ok_img, fmt, popts);
      CHECK(parsed.request.images.size() == 1 && parsed.request.images[0].size() == 1);
    }

    // Cross-batch max_images enforcement: item 0 has 1 image + item 1 has 2 video frames = 3 > max_images=2 -> REJECT!
    {
      const Json item0 = Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 70}}});
      const Json item1 = Json::array({Json{{"type", "video"},
                                           {"max_soft_tokens", 70},
                                           {"frames", Json::array({bmp_uri, bmp_uri})}}});
      const Json batch_3{{"input", Json::array({item0, item1})}};
      // Exceeds max_images=2 (3 > 2)
      expect_bad_req(batch_3, "input", popts);

      // Raise max_images=3, keep max_total_tokens=150: 3 * 64 = 192 > 150 -> REJECT on max_total_tokens!
      auto o_tok = popts;
      o_tok.max_images = 3;
      o_tok.max_total_tokens = 150;
      expect_bad_req(batch_3, "input", o_tok);

      // Raise max_total_tokens=192: exact boundary 3 images/frames & 192 tokens -> ACCEPT!
      o_tok.max_total_tokens = 192;
      const auto parsed = ingine::serve::parse_embedding_request(batch_3, fmt, o_tok);
      CHECK(parsed.request.inputs.size() == 2);
      CHECK(parsed.request.images.size() == 2);
      CHECK(parsed.request.images[0].size() == 1 && parsed.request.images[1].size() == 2);

      // max_images=0 rejects even 1 image or 1 video frame
      auto o_zero = popts;
      o_zero.max_images = 0;
      expect_bad_req(Json{{"input", item0}}, "input", o_zero);
      expect_bad_req(Json{{"input", Json::array({Json{{"type", "video"}, {"frames", Json::array({bmp_uri})}}})}},
                     "input", o_zero);
    }

    // Single-item video max_images (3 frames > max_images=2) & max_total_tokens (2x121=242 > 150)
    {
      const Json vid3{{"input", Json::array({Json{{"type", "video"},
                                                  {"max_soft_tokens", 70},
                                                  {"frames", Json::array({bmp_uri, bmp_uri, bmp_uri})}}})}};
      expect_bad_req(vid3, "input", popts);

      const Json vid2_140{{"input", Json::array({Json{{"type", "video"},
                                                      {"max_soft_tokens", 140},
                                                      {"frames", Json::array({bmp_uri, bmp_uri})}}})}};
      expect_bad_req(vid2_140, "input", popts);  // 2 frames <= max_images=2, but 242 tokens > max_total_tokens=150
    }

    // Integer narrowing / representability checks BEFORE conversion to int!
    // 1) dimensions: (1ULL<<32)+768, (1ULL<<32)+256, -(1LL<<32)+256, UINT64_MAX, INT64_MAX, float overflow, tap_layer > 512
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 4294968064ULL}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 4294967552ULL}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", -4294967040LL}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 18446744073709551615ULL}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 9223372036854775807LL}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 4294967552.0}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 256.5}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", 0}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"dimensions", -128}}, "dimensions", popts);
    expect_bad_req(Json{{"input", "hi"}, {"tap_layer", 2}, {"dimensions", 600}}, "dimensions", popts);

    // 2) image max_soft_tokens (top-level & nested image_url & outer-invalid + inner-valid)
    auto o_wide = popts;
    o_wide.max_images = 16;
    o_wide.max_total_tokens = 32768;
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 4294967576ULL}}})}},
        "input", o_wide);  // (1<<32)+280
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", -4294967016LL}}})}},
        "input", o_wide);  // -(1<<32)+280
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "image_url"},
                                         {"image_url", Json{{"url", bmp_uri}, {"max_soft_tokens", 4294967576ULL}}}}})}},
        "input", o_wide);
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "image_url"},
                                         {"max_soft_tokens", 4294967576ULL},
                                         {"image_url", Json{{"url", bmp_uri}, {"max_soft_tokens", 280}}}}})}},
        "input", o_wide);
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 280.0}}})}},
        "input", o_wide);

    // 3) video max_soft_tokens (video-level & per-frame object)
    expect_bad_req(Json{{"input", Json::array({Json{{"type", "video"},
                                                    {"max_soft_tokens", 4294967436ULL},
                                                    {"frames", Json::array({bmp_uri})}}})}},
                   "input", o_wide);  // (1<<32)+140
    expect_bad_req(Json{{"input", Json::array({Json{{"type", "video"},
                                                    {"max_soft_tokens", -4294967156LL},
                                                    {"frames", Json::array({bmp_uri})}}})}},
                   "input", o_wide);  // -(1<<32)+140
    expect_bad_req(
        Json{{"input", Json::array({Json{{"type", "video"},
                                         {"frames", Json::array({Json{{"url", bmp_uri}, {"max_soft_tokens", 4294967436ULL}}})}}})}},
        "input", o_wide);

    // 4) tap_layer narrowing traps
    expect_bad_req(Json{{"input", "hi"}, {"tap_layer", 4294967298ULL}}, "tap_layer", o_wide);
    expect_bad_req(Json{{"input", "hi"}, {"tap_layer", -4294967294LL}}, "tap_layer", o_wide);

    // 5) Live HTTP ChatServer carries ServerOptions::images.{max_images, max_total_tokens} into /v1/embeddings!
    {
      ingine::engine::FakeGenerator vfake(tok, "embeddinggemma-2-fake", 8192);
      vfake.set_model_family("embedding_gemma2");
      vfake.set_is_encoder_only(true);
      vfake.set_supports_images(true);
      vfake.set_embedding_dim(768);
      vfake.set_tap_embedding_dim(512);
      vfake.set_num_tap_layers(24);
      vfake.set_matryoshka_dims({768, 512, 256, 128});
      ingine::serve::ServerOptions vsopts;
      vsopts.max_model_len = 8192;
      vsopts.images.enabled = true;
      vsopts.images.max_images = 2;
      vsopts.images.max_total_tokens = 150;
      ingine::serve::ChatServer vsrv(vfake, fmt, vsopts);
      CHECK(vsrv.bind("127.0.0.1", 0));
      vsrv.start();
      httplib::Client vcli("127.0.0.1", vsrv.port());
      vcli.set_read_timeout(30, 0);

      // Valid 2-image batch totaling 128 tokens <= 150 -> HTTP 200
      const Json item_img = Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 70}}});
      auto r_ok = vcli.Post("/v1/embeddings", Json{{"input", Json::array({item_img, item_img})}}.dump(), "application/json");
      CHECK(r_ok && r_ok->status == 200);

      // 1 image + 2 video frames across batch = 3 > max_images=2 -> HTTP 400
      const Json item_vid2 = Json::array({Json{{"type", "video"}, {"max_soft_tokens", 70}, {"frames", Json::array({bmp_uri, bmp_uri})}}});
      auto r_lim_img = vcli.Post("/v1/embeddings", Json{{"input", Json::array({item_img, item_vid2})}}.dump(), "application/json");
      CHECK(r_lim_img && r_lim_img->status == 400);

      // 2 images @ max_soft_tokens=140 = 242 tokens > max_total_tokens=150 -> HTTP 400
      const Json item_img140 = Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 140}}});
      auto r_lim_tok = vcli.Post("/v1/embeddings", Json{{"input", Json::array({item_img140, item_img140})}}.dump(), "application/json");
      CHECK(r_lim_tok && r_lim_tok->status == 400);

      // Narrowing traps over live HTTP -> HTTP 400
      auto r_dim_ovf = vcli.Post("/v1/embeddings", Json{{"input", "hi"}, {"dimensions", 4294967552ULL}}.dump(), "application/json");
      CHECK(r_dim_ovf && r_dim_ovf->status == 400 && Json::parse(r_dim_ovf->body)["error"]["param"] == "dimensions");
      auto r_mst_ovf = vcli.Post(
          "/v1/embeddings",
          Json{{"input", Json::array({Json{{"type", "image_url"}, {"image_url", bmp_uri}, {"max_soft_tokens", 4294967576ULL}}})}}.dump(),
          "application/json");
      CHECK(r_mst_ovf && r_mst_ovf->status == 400 && Json::parse(r_mst_ovf->body)["error"]["param"] == "input");
      vsrv.stop();
    }
    std::printf("[PASS] Public embeddings parser limits (max_images/max_total_tokens incl. video) & int representability verified\n");
  }
}

void test_phase2_vision_preprocess_and_synthetic_scatter() {
  // Exact Gemma4ImageProcessor aspect-ratio sizing checks
  {
    const auto s1 = eg2_aspect_ratio_preserving_size(96, 96, 280);
    CHECK(s1.target_height == 768 && s1.target_width == 768 && s1.num_patches == 2304 && s1.num_soft_tokens == 256);
    const auto s2 = eg2_aspect_ratio_preserving_size(127, 193, 280);
    CHECK(s2.target_height == 624 && s2.target_width == 960 && s2.num_patches == 2340 && s2.num_soft_tokens == 260);
    const auto s3 = eg2_aspect_ratio_preserving_size(127, 193, 70);
    CHECK(s3.target_height == 288 && s3.target_width == 480 && s3.num_patches == 540 && s3.num_soft_tokens == 60);
    const auto s4 = eg2_aspect_ratio_preserving_size(96, 96, 140);
    CHECK(s4.target_height == 528 && s4.target_width == 528 && s4.num_soft_tokens == 121);
    bool threw = false;
    try {
      (void)eg2_aspect_ratio_preserving_size(96, 96, 200);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }

  // Bicubic antialias resize sanity on constant + linear gradient image
  {
    std::vector<std::uint8_t> solid(64 * 48 * 3);
    for (std::size_t i = 0; i < 64 * 48; ++i) {
      solid[3 * i + 0] = 20;
      solid[3 * i + 1] = 100;
      solid[3 * i + 2] = 220;
    }
    const auto resized = ingine::image::resize_bicubic_antialias(solid, 48, 64, 96, 144);
    CHECK(resized.size() == 96u * 144u * 3u);
    for (std::size_t i = 0; i < 96u * 144u; ++i) {
      CHECK(resized[3 * i + 0] == 20 && resized[3 * i + 1] == 100 && resized[3 * i + 2] == 220);
    }
  }

  // Synthetic tiny Vision + Text multimodal scatter end-to-end check
  {
    Eg2VisionWeightsF32 vw;
    vw.config.hidden_size = 16;
    vw.config.intermediate_size = 32;
    vw.config.num_hidden_layers = 2;
    vw.config.num_attention_heads = 2;
    vw.config.num_key_value_heads = 2;
    vw.config.head_dim = 8;
    vw.config.patch_size = 2;
    vw.config.pooling_kernel_size = 2;
    vw.config.position_embedding_size = 64;
    vw.config.default_output_length = 70;
    vw.text_hidden_size = 8;
    const int patch_in = 3 * 2 * 2;
    vw.patch_input_proj.assign(16 * patch_in, 0.05f);
    vw.position_embedding_table.assign(2 * 64 * 16, 0.01f);
    vw.embed_vision_projection.assign(8 * 16, 0.04f);
    vw.layers.resize(2);
    for (auto& lw : vw.layers) {
      lw.input_layernorm.assign(16, 1.0f);
      lw.q_proj.assign(16 * 16, 0.02f);
      lw.k_proj.assign(16 * 16, 0.02f);
      lw.v_proj.assign(16 * 16, 0.02f);
      lw.o_proj.assign(16 * 16, 0.02f);
      lw.q_norm.assign(8, 1.0f);
      lw.k_norm.assign(8, 1.0f);
      lw.post_attention_layernorm.assign(16, 1.0f);
      lw.pre_feedforward_layernorm.assign(16, 1.0f);
      lw.gate_proj.assign(32 * 16, 0.02f);
      lw.up_proj.assign(32 * 16, 0.02f);
      lw.down_proj.assign(16 * 32, 0.02f);
      lw.post_feedforward_layernorm.assign(16, 1.0f);
    }
    Eg2CpuVisionEncoder venc(std::move(vw));
    Eg2PreprocessedImage prep;
    prep.size = {4, 4, 2, 2, 4, 1};
    prep.patches_f32.assign(4 * patch_in, 0.75f);
    prep.pos_x = {0, 1, 0, 1};
    prep.pos_y = {0, 0, 1, 1};
    const auto soft = venc.encode_preprocessed(prep);
    CHECK(soft.size() == 8);
    for (float v : soft) CHECK(std::isfinite(v));
  }
  std::printf("[PASS] Phase 2 vision preprocessing, bicubic antialias & synthetic ViT encoder verified\n");
}

void test_phase3_audio_wav_mel_and_synthetic_conformer() {
  // Build deterministic PCM16 16kHz WAV in memory (1.0s = 16000 samples, 440Hz sine)
  auto make_wav_pcm16 = [](int n_samples, int sr, int channels, double freq) {
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(n_samples * channels * 2);
    std::vector<std::uint8_t> b(44 + data_bytes, 0);
    std::memcpy(b.data() + 0, "RIFF", 4);
    const std::uint32_t riff_sz = 36 + data_bytes;
    std::memcpy(b.data() + 4, &riff_sz, 4);
    std::memcpy(b.data() + 8, "WAVE", 4);
    std::memcpy(b.data() + 12, "fmt ", 4);
    const std::uint32_t fmt_sz = 16;
    const std::uint16_t audio_fmt = 1, n_ch = static_cast<std::uint16_t>(channels), bps = 16;
    const std::uint32_t sr32 = static_cast<std::uint32_t>(sr);
    const std::uint32_t byte_rate = sr32 * n_ch * 2;
    const std::uint16_t block_align = static_cast<std::uint16_t>(n_ch * 2);
    std::memcpy(b.data() + 16, &fmt_sz, 4);
    std::memcpy(b.data() + 20, &audio_fmt, 2);
    std::memcpy(b.data() + 22, &n_ch, 2);
    std::memcpy(b.data() + 24, &sr32, 4);
    std::memcpy(b.data() + 28, &byte_rate, 4);
    std::memcpy(b.data() + 32, &block_align, 2);
    std::memcpy(b.data() + 34, &bps, 2);
    std::memcpy(b.data() + 36, "data", 4);
    std::memcpy(b.data() + 40, &data_bytes, 4);
    for (int i = 0; i < n_samples; ++i) {
      const double v = 0.5 * std::sin(2.0 * 3.14159265358979323846 * freq * static_cast<double>(i) / sr);
      const std::int16_t s = static_cast<std::int16_t>(std::llround(v * 32767.0));
      for (int c = 0; c < channels; ++c) {
        std::memcpy(b.data() + 44 + (i * channels + c) * 2, &s, 2);
      }
    }
    return b;
  };

  // 1.0s mono 16kHz -> 99 mel frames -> 25 SSCP valid audio soft tokens
  {
    const auto wav = make_wav_pcm16(16000, 16000, 1, 440.0);
    const auto dec = eg2_decode_wav_bytes(wav);
    CHECK(dec.sample_rate == 16000 && dec.samples.size() == 16000);
    const auto prep = eg2_preprocess_audio(dec);
    CHECK(prep.mel_frames == 99 && prep.num_mels == 128 && prep.sscp_frames == 25 && prep.num_valid_tokens == 25);
    for (float v : prep.mel_f32) CHECK(std::isfinite(v));
  }
  // Odd-length 21950 samples (~1.372s, tests pad_to_multiple_of=128) -> 137 mel frames -> 35 valid audio soft tokens
  {
    const auto wav = make_wav_pcm16(21950, 16000, 2, 880.0);  // stereo downmix check too
    const auto dec = eg2_decode_wav_bytes(wav);
    CHECK(dec.sample_rate == 16000 && dec.samples.size() == 21950);
    const auto prep = eg2_preprocess_audio(dec);
    CHECK(prep.mel_frames == 137 && prep.num_valid_tokens == 35);
  }
  // Resampling check (8kHz 1.0s -> 16kHz 16000 samples)
  {
    const auto wav8 = make_wav_pcm16(8000, 8000, 1, 440.0);
    const auto dec = eg2_decode_wav_bytes(wav8);
    CHECK(dec.sample_rate == 16000 && dec.samples.size() == 16000);
  }
  // Corrupt WAV rejected fail-closed
  {
    const std::vector<std::uint8_t> junk = {'N', 'O', 'P', 'E', 0, 1, 2, 3};
    bool threw = false;
    try {
      (void)eg2_decode_wav_bytes(junk);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }

  // Synthetic mini Audio Conformer forward sanity check (1 layer, AH=16, 2 heads, hd=8, C0=4, C1=2, OP=12, TH=8)
  {
    Eg2AudioWeightsF32 aw;
    aw.config.hidden_size = 16;
    aw.config.num_hidden_layers = 1;
    aw.config.num_attention_heads = 2;
    aw.config.output_proj_dims = 12;
    aw.config.conv_kernel_size = 5;
    aw.config.sscp_conv_channels_0 = 4;
    aw.config.sscp_conv_channels_1 = 2;
    aw.config.use_clipped_linears = true;
    aw.text_hidden_size = 8;
    aw.sscp_layer0_conv.assign(4 * 1 * 9, 0.02f);
    aw.sscp_layer0_norm.assign(4, 1.0f);
    aw.sscp_layer1_conv.assign(2 * 4 * 9, 0.02f);
    aw.sscp_layer1_norm.assign(2, 1.0f);
    aw.sscp_input_proj_linear.assign(16 * (32 * 2), 0.01f);
    aw.output_proj_weight.assign(12 * 16, 0.02f);
    aw.output_proj_bias.assign(12, 0.01f);
    aw.embed_audio_projection.assign(8 * 12, 0.03f);
    aw.layers.resize(1);
    auto fill_clip = [](int rows, int cols) {
      Eg2ClippedLinearF32 c;
      c.weight.assign(static_cast<std::size_t>(rows) * cols, 0.01f);
      c.input_min = -10.0f;
      c.input_max = 10.0f;
      c.output_min = -10.0f;
      c.output_max = 10.0f;
      return c;
    };
    auto& lw = aw.layers[0];
    lw.feed_forward1.pre_layer_norm.assign(16, 1.0f);
    lw.feed_forward1.ffw_layer_1 = fill_clip(64, 16);
    lw.feed_forward1.ffw_layer_2 = fill_clip(16, 64);
    lw.feed_forward1.post_layer_norm.assign(16, 1.0f);
    lw.norm_pre_attn.assign(16, 1.0f);
    lw.q_proj = fill_clip(16, 16);
    lw.k_proj = fill_clip(16, 16);
    lw.v_proj = fill_clip(16, 16);
    lw.relative_k_proj.assign(16 * 16, 0.01f);
    lw.per_dim_scale.assign(8, 0.0f);
    lw.post = fill_clip(16, 16);
    lw.norm_post_attn.assign(16, 1.0f);
    lw.lconv_pre_layer_norm.assign(16, 1.0f);
    lw.lconv_linear_start = fill_clip(32, 16);
    lw.lconv_depthwise_conv1d.assign(16 * 1 * 5, 0.05f);
    lw.lconv_conv_norm.assign(16, 1.0f);
    lw.lconv_linear_end = fill_clip(16, 16);
    lw.feed_forward2.pre_layer_norm.assign(16, 1.0f);
    lw.feed_forward2.ffw_layer_1 = fill_clip(64, 16);
    lw.feed_forward2.ffw_layer_2 = fill_clip(16, 64);
    lw.feed_forward2.post_layer_norm.assign(16, 1.0f);
    lw.norm_out.assign(16, 1.0f);

    Eg2CpuAudioEncoder aenc(std::move(aw));
    ingine::model::EmbeddingAudio clip;
    clip.sample_rate = 16000;
    clip.samples.assign(3200, 0.25f);  // 0.2s -> 19 mel -> 5 valid audio tokens
    const auto soft = aenc.encode_audio(clip);
    CHECK(soft.size() == 5u * 8u);
    for (float v : soft) CHECK(std::isfinite(v));
  }
  std::printf("[PASS] Phase 3 audio WAV decode, 128-bin HTK log-mel STFT & synthetic Conformer verified\n");
}

}  // namespace

int main() {
  try {
    test_model_family_enum();
    test_synthetic_cpu_encoder_and_matryoshka();
    test_phase2_vision_preprocess_and_synthetic_scatter();
    test_phase3_audio_wav_mel_and_synthetic_conformer();
    const auto eg2_dir = find_eg2_model_dir();
    if (!eg2_dir.empty()) {
      test_real_checkpoint_tokenizer_manifest_http_and_oracle(eg2_dir);
    } else {
      std::printf("[SKIP] real checkpoint tests skipped (set INGINE_EG2_MODEL_DIR or place in /mnt/storage/models/embeddinggemma-2)\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eg2_cpu failed: %s\n", e.what());
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
