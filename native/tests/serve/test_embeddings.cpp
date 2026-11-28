// End-to-end CPU unit tests for POST /v1/embeddings, PlatosKey .safetensors + JSON metadata loader,
// OpenAI schema/base64 parity, HTTP 400 validation, and zero-KV-slot-leak mixed chat+embedding traffic.
#include "engine/fake_generator.h"
#include "engine/scheduler.h"
#include "ingine/platos_key.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::PlatosKey;
using ingine::engine::FakeGenerator;
using ingine::serve::ChatServer;
using ingine::text::ChatFormat;

int g_failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                                   \
      throw std::runtime_error("check failed");                                       \
    }                                                                                 \
  } while (0)

const ChatFormat* g_fmt = nullptr;

std::uint16_t f32_to_fp16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((u >> 16) & 0x8000u);
  const std::uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u == 0) return sign;
  if (abs_u < 0x38800000u) return sign;
  const std::uint32_t exp = (abs_u >> 23) - 112u;
  const std::uint32_t mant = abs_u & 0x007fffffu;
  const std::uint32_t rounded = ((exp << 23) | mant) + 0x00000fffu + ((mant >> 13) & 1u);
  return static_cast<std::uint16_t>(sign | (rounded >> 13));
}

struct TensorWriteSpec {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

void write_safetensors(const std::filesystem::path& path, const std::vector<TensorWriteSpec>& tensors,
                       const std::map<std::string, std::string>& meta = {}) {
  nlohmann::ordered_json hdr = nlohmann::ordered_json::object();
  if (!meta.empty()) {
    nlohmann::ordered_json m = nlohmann::ordered_json::object();
    for (const auto& [k, v] : meta) m[k] = v;
    hdr["__metadata__"] = std::move(m);
  }
  std::uint64_t off = 0;
  for (const auto& t : tensors) {
    const std::uint64_t end = off + static_cast<std::uint64_t>(t.bytes.size());
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, end}}};
    off = end;
  }
  const std::string hdr_str = hdr.dump();
  const std::uint64_t hdr_len = static_cast<std::uint64_t>(hdr_str.size());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  CHECK(static_cast<bool>(out));
  for (int i = 0; i < 8; ++i) {
    const char c = static_cast<char>((hdr_len >> (8 * i)) & 0xffu);
    out.write(&c, 1);
  }
  out.write(hdr_str.data(), static_cast<std::streamsize>(hdr_str.size()));
  for (const auto& t : tensors) {
    out.write(reinterpret_cast<const char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
  }
  out.flush();
  CHECK(static_cast<bool>(out));
}

std::vector<std::uint8_t> pack_f16(const std::vector<float>& vals) {
  std::vector<std::uint8_t> out(vals.size() * 2);
  for (std::size_t i = 0; i < vals.size(); ++i) {
    const std::uint16_t h = f32_to_fp16(vals[i]);
    out[2 * i + 0] = static_cast<std::uint8_t>(h & 0xffu);
    out[2 * i + 1] = static_cast<std::uint8_t>((h >> 8) & 0xffu);
  }
  return out;
}

std::vector<std::uint8_t> pack_f32(const std::vector<float>& vals) {
  std::vector<std::uint8_t> out(vals.size() * 4);
  std::memcpy(out.data(), vals.data(), out.size());
  return out;
}

std::vector<float> decode_base64_f32(std::string_view b64) {
  auto val = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  CHECK(b64.size() % 4 == 0);
  std::vector<std::uint8_t> raw;
  raw.reserve((b64.size() / 4) * 3);
  for (std::size_t i = 0; i < b64.size(); i += 4) {
    const int a = val(static_cast<unsigned char>(b64[i]));
    const int b = val(static_cast<unsigned char>(b64[i + 1]));
    const bool p2 = (b64[i + 2] == '=');
    const bool p3 = (b64[i + 3] == '=');
    const int c = p2 ? 0 : val(static_cast<unsigned char>(b64[i + 2]));
    const int d = p3 ? 0 : val(static_cast<unsigned char>(b64[i + 3]));
    CHECK(a >= 0 && b >= 0 && c >= 0 && d >= 0);
    const std::uint32_t t = (static_cast<std::uint32_t>(a) << 18) | (static_cast<std::uint32_t>(b) << 12) |
                            (static_cast<std::uint32_t>(c) << 6) | static_cast<std::uint32_t>(d);
    raw.push_back(static_cast<std::uint8_t>((t >> 16) & 0xffu));
    if (!p2) raw.push_back(static_cast<std::uint8_t>((t >> 8) & 0xffu));
    if (!p3) raw.push_back(static_cast<std::uint8_t>(t & 0xffu));
  }
  CHECK(raw.size() % 4 == 0);
  std::vector<float> out(raw.size() / 4);
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(raw[i * 4 + 0]) |
                               (static_cast<std::uint32_t>(raw[i * 4 + 1]) << 8) |
                               (static_cast<std::uint32_t>(raw[i * 4 + 2]) << 16) |
                               (static_cast<std::uint32_t>(raw[i * 4 + 3]) << 24);
    std::memcpy(&out[i], &bits, sizeof(float));
  }
  return out;
}

double l2_norm(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s);
}

struct Response {
  int status = 0;
  httplib::Headers headers;
  std::string body;
};

Response http_req(int port, const std::string& method, const std::string& path, const std::string& body = "") {
  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(20, 0);
  httplib::Result r = method == "GET" ? cli.Get(path) : cli.Post(path, body, "application/json");
  if (!r) throw std::runtime_error("HTTP request failed: " + httplib::to_string(r.error()));
  return {r->status, r->headers, r->body};
}

std::string header_val(const httplib::Headers& h, const std::string& key) {
  auto it = h.find(key);
  return it == h.end() ? "" : it->second;
}

struct Served {
  std::unique_ptr<FakeGenerator> gen;
  std::unique_ptr<ChatServer> srv;
  int port() const { return srv->port(); }
  ~Served() {
    if (srv) srv->stop();
  }
};

std::unique_ptr<Served> make_served(int max_model_len = 1024, int max_queue = 8,
                                    const std::filesystem::path& pk_path = {},
                                    const std::filesystem::path& pk_meta = {}, bool enable_embeddings = true) {
  auto s = std::make_unique<Served>();
  s->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "Inkling-2-Mini-fake", max_model_len);
  s->gen->set_supports_embeddings(enable_embeddings);
  s->gen->set_embedding_dim(64);
  s->gen->set_num_tap_layers(48);
  if (!pk_path.empty()) s->gen->load_platos_key(pk_path, pk_meta);
  ingine::serve::ServerOptions opts;
  opts.max_queue = max_queue;
  opts.max_model_len = max_model_len;
  s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, opts);
  CHECK(s->srv->bind("127.0.0.1", 0));
  s->srv->start();
  return s;
}

void test_platos_key_loader_unit() {
  const auto tmp = std::filesystem::temp_directory_path() / "ingine_test_embeddings_963";
  std::filesystem::create_directories(tmp);
  const auto pk_v1_path = tmp / "key_v1.safetensors";
  const auto pk_meta_path = tmp / "key_v1.meta.json";
  const auto pk_bad_meta_path = tmp / "key_v1_bad.meta.json";
  const auto pk_wb_path = tmp / "key_wb.safetensors";

  constexpr int D_IN = 64, D_OUT = 32;
  std::vector<float> R(D_IN * D_OUT), dm(D_IN), tm(D_OUT);
  for (int i = 0; i < D_IN * D_OUT; ++i) R[i] = static_cast<float>((i % 19) - 9) * 0.0625f;
  for (int i = 0; i < D_IN; ++i) dm[i] = static_cast<float>((i % 7) - 3) * 0.125f;
  for (int j = 0; j < D_OUT; ++j) tm[j] = static_cast<float>((j % 5) - 2) * 0.25f;
  const std::vector<float> sc = {1.25f};

  write_safetensors(
      pk_v1_path,
      {{"R", "F16", {D_IN, D_OUT}, pack_f16(R)},
       {"donor_mean", "F16", {D_IN}, pack_f16(dm)},
       {"target_mean", "F16", {D_OUT}, pack_f16(tm)},
       {"scale", "F16", {1}, pack_f16(sc)}},
      {{"format", "platos_key_v1"}, {"donor_model", "Inkling-2-Mini"}, {"tap_spec", "visit_23"}, {"method", "ridge"}});

  {
    std::ofstream mf(pk_meta_path);
    mf << R"({"d_donor": 64, "d_target": 32, "target_model": "h3_test", "cartridge_id": "cart-001"})";
  }
  {
    std::ofstream mf(pk_bad_meta_path);
    mf << R"({"d_donor": 999, "d_target": 32})";
  }

  auto key = PlatosKey::load(pk_v1_path, pk_meta_path, D_IN);
  CHECK(key->in_dim() == D_IN && key->out_dim() == D_OUT);
  CHECK(std::fabs(key->scale() - 1.25f) < 1e-6f);
  CHECK(key->metadata().format == "platos_key_v1");
  CHECK(key->metadata().tap_spec == "visit_23");
  CHECK(key->metadata().target_model == "h3_test");
  CHECK(key->metadata().raw_metadata.at("cartridge_id") == "cart-001");

  // Fail-closed checks: wrong expected_in_dim and contradicting JSON metadata.
  bool threw_dim = false;
  try {
    (void)PlatosKey::load(pk_v1_path, 128);
  } catch (const std::invalid_argument&) {
    threw_dim = true;
  }
  CHECK(threw_dim);

  bool threw_meta = false;
  try {
    (void)PlatosKey::load(pk_v1_path, pk_bad_meta_path, D_IN);
  } catch (const std::invalid_argument&) {
    threw_meta = true;
  }
  CHECK(threw_meta);

  // Numerical parity check against analytical FP64 reference.
  std::vector<float> x(2 * D_IN);
  for (int i = 0; i < 2 * D_IN; ++i) x[i] = std::sin(static_cast<float>(i) * 0.13f);
  std::vector<float> y(2 * D_OUT, 0.0f);
  key->project_cpu(x.data(), 2, y.data(), /*l2_normalize=*/false);
  for (int b = 0; b < 2; ++b) {
    for (int j = 0; j < D_OUT; ++j) {
      double acc = 0.0;
      for (int i = 0; i < D_IN; ++i) {
        acc += (static_cast<double>(x[b * D_IN + i]) - static_cast<double>(dm[i])) * static_cast<double>(R[i * D_OUT + j]);
      }
      const double ref = static_cast<double>(tm[j]) + 1.25 * acc;
      CHECK(std::fabs(static_cast<double>(y[b * D_OUT + j]) - ref) < 1e-4);
    }
  }

  // Affine W/b fallback format.
  write_safetensors(pk_wb_path, {{"W", "F32", {D_IN, D_OUT}, pack_f32(R)}, {"b", "F32", {D_OUT}, pack_f32(tm)}});
  auto wb = PlatosKey::load(pk_wb_path, D_IN);
  CHECK(wb->in_dim() == D_IN && wb->out_dim() == D_OUT && wb->metadata().format == "affine_wb");
  std::vector<float> y_wb(D_OUT, 0.0f);
  wb->project_cpu(x.data(), 1, y_wb.data(), /*l2_normalize=*/true);
  CHECK(std::fabs(l2_norm(y_wb) - 1.0) < 1e-5);

  std::filesystem::remove_all(tmp);
}

void test_openai_schema_batch_and_base64() {
  auto s = make_served();

  // Check /health & /v1/models advertise embedding capabilities.
  {
    const Response hr = http_req(s->port(), "GET", "/health");
    CHECK(hr.status == 200);
    const Json h = Json::parse(hr.body);
    CHECK(h["supports_embeddings"] == true);
    CHECK(h["embedding_dim"] == 64);
    CHECK(h["num_tap_layers"] == 48);
    CHECK(h["platos_key_loaded"] == false);
    CHECK(h["platos_key_out_dim"] == 0);

    const Response mr = http_req(s->port(), "GET", "/v1/models");
    CHECK(mr.status == 200);
    const Json m = Json::parse(mr.body);
    CHECK(m["data"][0]["supports_embeddings"] == true);
    CHECK(m["data"][0]["embedding_dim"] == 64);
    CHECK(m["data"][0]["num_tap_layers"] == 48);
  }

  // Single string input.
  const std::string s0 = "Atlas embeddings test string";
  const std::string s1 = "Second prompt with more tokens for batched prefill";
  const std::string s2 = "Third";
  const int tok0 = static_cast<int>(g_fmt->tokenizer().encode(s0, false).size());
  const int tok1 = static_cast<int>(g_fmt->tokenizer().encode(s1, false).size());
  const int tok2 = static_cast<int>(g_fmt->tokenizer().encode(s2, false).size());

  const Response r_single = http_req(s->port(), "POST", "/v1/embeddings",
                                     Json{{"model", "ignored-client-alias"}, {"input", s0}}.dump());
  CHECK(r_single.status == 200);
  CHECK(header_val(r_single.headers, "X-Ingine-Kv-Slots-Allocated") == "0");
  CHECK(header_val(r_single.headers, "X-Ingine-Lm-Head-Skipped") == "true");
  CHECK(header_val(r_single.headers, "X-Ingine-Tap-Layer") == "full");
  CHECK(header_val(r_single.headers, "X-Ingine-Platos-Key") == "false");

  const Json j_single = Json::parse(r_single.body);
  CHECK(j_single["object"] == "list");
  CHECK(j_single["model"] == "Inkling-2-Mini-fake");
  CHECK(j_single["data"].is_array() && j_single["data"].size() == 1);
  CHECK(j_single["data"][0]["object"] == "embedding");
  CHECK(j_single["data"][0]["index"] == 0);
  CHECK(j_single["data"][0]["embedding"].is_array() && j_single["data"][0]["embedding"].size() == 64);
  CHECK(j_single["usage"]["prompt_tokens"] == tok0);
  CHECK(j_single["usage"]["total_tokens"] == tok0);

  std::vector<float> v0 = j_single["data"][0]["embedding"].get<std::vector<float>>();
  CHECK(std::fabs(l2_norm(v0) - 1.0) < 1e-5);

  // Batch array input -> matches single string bit-for-bit on index 0 and exact total usage.
  const Response r_batch = http_req(s->port(), "POST", "/v1/embeddings",
                                    Json{{"input", Json::array({s0, s1, s2})}, {"encoding_format", "float"}}.dump());
  CHECK(r_batch.status == 200);
  const Json j_batch = Json::parse(r_batch.body);
  CHECK(j_batch["object"] == "list");
  CHECK(j_batch["data"].size() == 3);
  CHECK(j_batch["usage"]["prompt_tokens"] == tok0 + tok1 + tok2);
  CHECK(j_batch["usage"]["total_tokens"] == tok0 + tok1 + tok2);
  for (int i = 0; i < 3; ++i) {
    CHECK(j_batch["data"][i]["object"] == "embedding");
    CHECK(j_batch["data"][i]["index"] == i);
    const auto vi = j_batch["data"][i]["embedding"].get<std::vector<float>>();
    CHECK(vi.size() == 64);
    CHECK(std::fabs(l2_norm(vi) - 1.0) < 1e-5);
  }
  const auto batch_v0 = j_batch["data"][0]["embedding"].get<std::vector<float>>();
  for (int d = 0; d < 64; ++d) CHECK(std::fabs(batch_v0[d] - v0[d]) < 1e-6f);

  // encoding_format="base64" exact roundtrip parity with "float".
  const Response r_b64 = http_req(s->port(), "POST", "/v1/embeddings",
                                  Json{{"input", Json::array({s0, s1, s2})}, {"encoding_format", "base64"}}.dump());
  CHECK(r_b64.status == 200);
  const Json j_b64 = Json::parse(r_b64.body);
  CHECK(j_b64["data"].size() == 3);
  for (int i = 0; i < 3; ++i) {
    CHECK(j_b64["data"][i]["embedding"].is_string());
    const auto decoded = decode_base64_f32(j_b64["data"][i]["embedding"].get<std::string>());
    const auto expected = j_batch["data"][i]["embedding"].get<std::vector<float>>();
    CHECK(decoded.size() == expected.size());
    for (std::size_t d = 0; d < expected.size(); ++d) {
      CHECK(std::fabs(decoded[d] - expected[d]) < 1e-6f);
    }
  }
}

void test_atlas_extensions_and_platos_key_http() {
  const auto tmp = std::filesystem::temp_directory_path() / "ingine_test_embeddings_http_pk";
  std::filesystem::create_directories(tmp);
  const auto pk_path = tmp / "http_key.safetensors";
  const auto meta_path = tmp / "http_key.json";

  constexpr int D_IN = 64, D_OUT = 24;
  std::vector<float> R(D_IN * D_OUT), dm(D_IN), tm(D_OUT);
  for (int i = 0; i < D_IN * D_OUT; ++i) R[i] = static_cast<float>((i % 13) - 6) * 0.0625f;
  for (int i = 0; i < D_IN; ++i) dm[i] = static_cast<float>((i % 5) - 2) * 0.125f;
  for (int j = 0; j < D_OUT; ++j) tm[j] = static_cast<float>((j % 7) - 3) * 0.25f;
  write_safetensors(
      pk_path,
      {{"R", "F16", {D_IN, D_OUT}, pack_f16(R)},
       {"donor_mean", "F16", {D_IN}, pack_f16(dm)},
       {"target_mean", "F16", {D_OUT}, pack_f16(tm)},
       {"scale", "F16", {1}, pack_f16({0.75f})}},
      {{"format", "platos_key_v1"}, {"tap_spec", "visit_23"}});
  {
    std::ofstream mf(meta_path);
    mf << R"({"d_donor": 64, "d_target": 24, "tap_spec": "visit_23"})";
  }

  auto s = make_served(1024, 8, pk_path, meta_path);

  const Response hr = http_req(s->port(), "GET", "/health");
  CHECK(hr.status == 200);
  const Json hj = Json::parse(hr.body);
  CHECK(hj["platos_key_loaded"] == true);
  CHECK(hj["platos_key_out_dim"] == D_OUT);
  CHECK(hj["platos_key_tap_spec"] == "visit_23");

  // Pooling modes: mean vs last_token vs cls.
  const std::string p_a = "Hello world alpha";
  const std::string p_b = "Hello different tail";
  auto embed_vec = [&](const Json& body) {
    const Response r = http_req(s->port(), "POST", "/v1/embeddings", body.dump());
    CHECK(r.status == 200);
    return std::pair{r, Json::parse(r.body)["data"][0]["embedding"].get<std::vector<float>>()};
  };

  const auto [r_mean, v_mean] = embed_vec({{"input", p_a}, {"pooling", "mean"}, {"normalize", false}});
  const auto [r_last, v_last] = embed_vec({{"input", p_a}, {"pooling", "last_token"}, {"normalize", false}});
  const auto [r_cls_a, v_cls_a] = embed_vec({{"input", p_a}, {"pooling", "cls"}, {"normalize", false}});
  const auto [r_cls_b, v_cls_b] = embed_vec({{"input", p_b}, {"pooling", "cls"}, {"normalize", false}});
  CHECK(std::fabs(l2_norm(v_mean) - 1.0) > 1e-2);  // unnormalized
  CHECK(v_mean != v_last && v_mean != v_cls_a);
  // Causal CLS depends only on token 0 ("Hello"), so p_a and p_b must match identically!
  for (int d = 0; d < D_IN; ++d) CHECK(std::fabs(v_cls_a[d] - v_cls_b[d]) < 1e-6f);

  // Early-exit tap_layer=23 vs 41 vs full.
  const auto [r_t23, v_t23] = embed_vec({{"input", p_a}, {"tap_layer", 23}, {"normalize", false}});
  const auto [r_t41, v_t41] = embed_vec({{"input", p_a}, {"tap_layer", 41}, {"normalize", false}});
  CHECK(header_val(r_t23.headers, "X-Ingine-Tap-Layer") == "23");
  CHECK(header_val(r_t41.headers, "X-Ingine-Tap-Layer") == "41");
  CHECK(v_t23 != v_t41 && v_t23 != v_mean);

  // Opt-in platos_key projection at tap_layer=23 (raw + normalized).
  const auto [r_pk_raw, v_pk_raw] =
      embed_vec({{"input", p_a}, {"tap_layer", 23}, {"platos_key", true}, {"normalize", false}});
  CHECK(header_val(r_pk_raw.headers, "X-Ingine-Platos-Key") == "true");
  CHECK(!header_val(r_pk_raw.headers, "X-Ingine-Platos-Key-Ms").empty());
  CHECK(v_pk_raw.size() == D_OUT);

  auto loaded_key = PlatosKey::load(pk_path, meta_path, D_IN);
  std::vector<float> ref_pk(D_OUT);
  loaded_key->project_cpu(v_t23.data(), 1, ref_pk.data(), false);
  for (int j = 0; j < D_OUT; ++j) CHECK(std::fabs(v_pk_raw[j] - ref_pk[j]) < 1e-5f);

  const auto [r_pk_norm, v_pk_norm] =
      embed_vec({{"input", p_a}, {"tap_layer", 23}, {"platos_key", true}, {"normalize", true}});
  CHECK(v_pk_norm.size() == D_OUT);
  CHECK(std::fabs(l2_norm(v_pk_norm) - 1.0) < 1e-5);

  // Fail-closed checks: bound tap_layer=23 rejects omitted tap_layer and conflicting tap_layer=41 with HTTP 400!
  {
    const Response r_omit = http_req(
        s->port(), "POST", "/v1/embeddings",
        Json{{"input", p_a}, {"platos_key", true}, {"normalize", false}}.dump());
    CHECK(r_omit.status == 400);
    CHECK(Json::parse(r_omit.body)["error"]["param"] == "tap_layer");

    const Response r_conflict = http_req(
        s->port(), "POST", "/v1/embeddings",
        Json{{"input", p_a}, {"tap_layer", 41}, {"platos_key", true}, {"normalize", false}}.dump());
    CHECK(r_conflict.status == 400);
    CHECK(Json::parse(r_conflict.body)["error"]["param"] == "tap_layer");

    // Matching cartridge_sha256 succeeds; mismatched cartridge_sha256 fails closed with HTTP 400.
    const std::string real_sha = loaded_key->metadata().file_sha256;
    CHECK(real_sha.size() == 64);
    CHECK(hj["platos_key_sha256"] == real_sha);
    CHECK(hj["platos_key_tap_layer"] == 23);
    CHECK(hj["platos_key_layout"] == "ridge_R");
    const Response r_sha_ok = http_req(
        s->port(), "POST", "/v1/embeddings",
        Json{{"input", p_a}, {"tap_layer", 23}, {"platos_key", true}, {"cartridge_sha256", real_sha}}.dump());
    CHECK(r_sha_ok.status == 200);

    std::string bad_sha = real_sha;
    bad_sha[0] = (bad_sha[0] == 'a') ? 'b' : 'a';
    const Response r_sha_bad = http_req(
        s->port(), "POST", "/v1/embeddings",
        Json{{"input", p_a}, {"tap_layer", 23}, {"platos_key", true}, {"cartridge_sha256", bad_sha}}.dump());
    CHECK(r_sha_bad.status == 400);
    CHECK(Json::parse(r_sha_bad.body)["error"]["param"] == "cartridge_sha256");
  }

  std::filesystem::remove_all(tmp);
}

void test_cartridge_964_binding_and_layouts_fail_closed() {
  const auto tmp = std::filesystem::temp_directory_path() / "ingine_test_embeddings_964_binding";
  std::filesystem::remove_all(tmp);
  std::filesystem::create_directories(tmp);

  constexpr int D_IN = 64, D_OUT = 24, RANK = 8;
  std::vector<float> R(D_IN * D_OUT), R2(D_IN * D_OUT), dm(D_IN), tm(D_OUT);
  for (int i = 0; i < D_IN * D_OUT; ++i) {
    R[i] = static_cast<float>((i % 11) - 5) * 0.0625f;
    R2[i] = static_cast<float>((i % 13) - 6) * 0.0625f;
  }
  for (int i = 0; i < D_IN; ++i) dm[i] = static_cast<float>((i % 5) - 2) * 0.125f;
  for (int j = 0; j < D_OUT; ++j) tm[j] = static_cast<float>((j % 7) - 3) * 0.25f;

  // 1) Wrong same-width artifact: sidecar pins cartridge_sha256 of key A, loaded with key B -> fail-closed!
  const auto p_a = tmp / "key_a.safetensors";
  const auto p_b = tmp / "key_b.safetensors";
  const auto m_a = tmp / "key_a.json";
  write_safetensors(p_a, {{"R", "F32", {D_IN, D_OUT}, pack_f32(R)}}, {{"format", "platos_key_v1"}});
  write_safetensors(p_b, {{"R", "F32", {D_IN, D_OUT}, pack_f32(R2)}}, {{"format", "platos_key_v1"}});
  const auto ka = PlatosKey::load(p_a, D_IN);
  const std::string sha_a = ka->metadata().file_sha256;
  CHECK(sha_a.size() == 64);
  {
    std::ofstream mf(m_a);
    mf << Json{{"cartridge_id", "inkling_test_cart"},
               {"cartridge_sha256", sha_a},
               {"donor_model_id", "thinkingmachines/Inkling-2-Mini"},
               {"d_donor", D_IN},
               {"d_target", D_OUT},
               {"tap_spec", "visit_41"},
               {"tap_layer", 41}}
              .dump();
  }
  auto ka_pinned = PlatosKey::load(p_a, m_a, D_IN);
  CHECK(ka_pinned->metadata().cartridge_id == "inkling_test_cart");
  CHECK(ka_pinned->metadata().donor_model == "thinkingmachines/Inkling-2-Mini");
  CHECK(ka_pinned->metadata().expected_tap_layer == std::optional<int>(41));

  bool threw_wrong_artifact = false;
  try {
    (void)PlatosKey::load(p_b, m_a, D_IN);
  } catch (const std::invalid_argument&) {
    threw_wrong_artifact = true;
  }
  CHECK(threw_wrong_artifact);

  // 2) Mismatched donor_model_id (both cross-family AND distinct same-family checkpoints) -> fail-closed!
  for (const std::string& bad_donor :
       {"Qwen/Qwen3.8-27B", "inclusionAI/Ling-3.2-air", "OtherModel-7B",
        "example/Inkling-2-Different-Checkpoint", "thinkingmachines/Inkling-2-Mini-abliterated"}) {
    const auto m_bad_donor = tmp / "bad_donor.json";
    {
      std::ofstream mf(m_bad_donor);
      mf << Json{{"cartridge_sha256", sha_a},
                 {"donor_model_id", bad_donor},
                 {"d_donor", D_IN},
                 {"d_target", D_OUT},
                 {"tap_spec", "visit_41"},
                 {"tap_layer", 41}}
                .dump();
    }
    bool threw_donor = false;
    try {
      (void)make_served(1024, 8, p_a, m_bad_donor);
    } catch (const std::invalid_argument&) {
      threw_donor = true;
    }
    CHECK(threw_donor);
  }

  // 3) Conflicting tap_spec vs tap_layer AND unrecognized nonempty tap_spec ("visit41") -> fail-closed at load!
  {
    const auto m_bad_tap = tmp / "bad_tap.json";
    std::ofstream mf(m_bad_tap);
    mf << Json{{"cartridge_sha256", sha_a}, {"tap_spec", "visit_10"}, {"tap_layer", 41}}.dump();
    mf.close();
    bool threw_conflict = false;
    try {
      (void)PlatosKey::load(p_a, m_bad_tap, D_IN);
    } catch (const std::invalid_argument&) {
      threw_conflict = true;
    }
    CHECK(threw_conflict);

    for (const std::string& bad_spec : {"visit41", "layer-31", "mid_block", "slot0"}) {
      const auto m_unk_tap = tmp / "unk_tap.json";
      std::ofstream uf(m_unk_tap);
      uf << Json{{"cartridge_sha256", sha_a},
                 {"donor_model_id", "thinkingmachines/Inkling-2-Mini"},
                 {"d_donor", D_IN},
                 {"d_target", D_OUT},
                 {"tap_spec", bad_spec}}
                .dump();
      uf.close();
      bool threw_unk_spec = false;
      try {
        (void)PlatosKey::load(p_a, m_unk_tap, D_IN);
      } catch (const std::invalid_argument&) {
        threw_unk_spec = true;
      }
      CHECK(threw_unk_spec);
    }

    // Legacy unbound key (no tap_spec, no tap_layer, no donor_model) remains supported.
    auto s_legacy = make_served(1024, 8, p_a, {});
    const Response r_leg = http_req(s_legacy->port(), "POST", "/v1/embeddings",
                                    Json{{"input", "hello"}, {"platos_key", true}}.dump());
    CHECK(r_leg.status == 200);

    // Rev4 regression: contradictory embedded tap_spec=visit_23 vs sidecar tap_spec=visit_41 MUST fail closed!
    const auto p_emb23 = tmp / "key_emb23.safetensors";
    write_safetensors(p_emb23, {{"R", "F32", {D_IN, D_OUT}, pack_f32(R)}},
                      {{"format", "platos_key_v1"}, {"tap_spec", "visit_23"}, {"donor_model", "Inkling-2-Mini-NVFP4"}});
    const std::string sha_emb23 = PlatosKey::load(p_emb23, D_IN)->metadata().file_sha256;
    const auto m_sidecar41 = tmp / "sidecar_41.json";
    {
      std::ofstream mf(m_sidecar41);
      mf << Json{{"cartridge_sha256", sha_emb23},
                 {"donor_model_id", "thinkingmachines/Inkling-2-Mini"},
                 {"d_donor", D_IN},
                 {"d_target", D_OUT},
                 {"tap_spec", "visit_41"},
                 {"tap_layer", 41}}
                .dump();
    }
    bool threw_emb_vs_sc = false;
    try {
      (void)make_served(1024, 8, p_emb23, m_sidecar41);
    } catch (const std::invalid_argument&) {
      threw_emb_vs_sc = true;
    }
    CHECK(threw_emb_vs_sc);

    // Compatible alias (embedded visit_23 + sidecar layer_23 / tap_layer=23) succeeds and binds tap_layer=23!
    const auto m_compat23 = tmp / "sidecar_compat23.json";
    {
      std::ofstream mf(m_compat23);
      mf << Json{{"cartridge_sha256", sha_emb23},
                 {"donor_model_id", "thinkingmachines/Inkling-2-Mini"},
                 {"d_donor", D_IN},
                 {"d_target", D_OUT},
                 {"tap_spec", "layer_23"},
                 {"tap_layer", 23}}
                .dump();
    }
    auto s_compat = make_served(1024, 8, p_emb23, m_compat23);
    const Response r_c23 = http_req(s_compat->port(), "POST", "/v1/embeddings",
                                    Json{{"input", "hello"}, {"tap_layer", 23}, {"platos_key", true}}.dump());
    CHECK(r_c23.status == 200);

    // Embedded multi-tap concat_all4 cannot be masked by sidecar single-tap visit_41!
    const auto p_emb_concat = tmp / "key_emb_concat.safetensors";
    write_safetensors(p_emb_concat, {{"R", "F32", {D_IN, D_OUT}, pack_f32(R)}},
                      {{"format", "platos_key_v1"}, {"tap_spec", "concat_all4"}});
    const std::string sha_emb_concat = PlatosKey::load(p_emb_concat, D_IN)->metadata().file_sha256;
    const auto m_mask_concat = tmp / "mask_concat.json";
    {
      std::ofstream mf(m_mask_concat);
      mf << Json{{"cartridge_sha256", sha_emb_concat}, {"tap_spec", "visit_41"}, {"tap_layer", 41}}.dump();
    }
    bool threw_mask_concat = false;
    try {
      (void)PlatosKey::load(p_emb_concat, m_mask_concat, D_IN);
    } catch (const std::invalid_argument&) {
      threw_mask_concat = true;
    }
    CHECK(threw_mask_concat);
  }

  // 4) Final-layer bound key (`tap_spec: "final_normed"`): accepts omitted tap_layer, rejects explicit tap_layer!
  {
    const auto m_final = tmp / "final.json";
    std::ofstream mf(m_final);
    mf << Json{{"cartridge_sha256", sha_a},
               {"donor_model_id", "thinkingmachines/Inkling-2-Mini"},
               {"d_donor", D_IN},
               {"d_target", D_OUT},
               {"tap_spec", "final_normed"}}
              .dump();
    mf.close();
    auto sf = make_served(1024, 8, p_a, m_final);
    const Response r_ok = http_req(sf->port(), "POST", "/v1/embeddings",
                                   Json{{"input", "hello"}, {"platos_key", true}}.dump());
    CHECK(r_ok.status == 200);
    const Response r_bad = http_req(sf->port(), "POST", "/v1/embeddings",
                                    Json{{"input", "hello"}, {"tap_layer", 23}, {"platos_key", true}}.dump());
    CHECK(r_bad.status == 400);
    CHECK(Json::parse(r_bad.body)["error"]["param"] == "tap_layer");
  }

  // 5) Factorized SVD layout (`U_r`, `V_r`) + affine Yue2 layout (`W_yue2`, `b_yue2`) numerical parity!
  {
    std::vector<float> U(D_IN * RANK), V(RANK * D_OUT);
    for (int i = 0; i < D_IN * RANK; ++i) U[i] = std::sin(static_cast<float>(i + 1) * 0.17f) * 0.25f;
    for (int i = 0; i < RANK * D_OUT; ++i) V[i] = std::cos(static_cast<float>(i + 1) * 0.23f) * 0.25f;
    const auto p_svd = tmp / "key_svd.safetensors";
    write_safetensors(p_svd,
                      {{"U_r", "F32", {D_IN, RANK}, pack_f32(U)},
                       {"V_r", "F32", {RANK, D_OUT}, pack_f32(V)},
                       {"donor_mean", "F32", {D_IN}, pack_f32(dm)},
                       {"target_mean", "F32", {D_OUT}, pack_f32(tm)},
                       {"scale", "F32", {1}, pack_f32({1.5f})}},
                      {{"format", "platos_key_svd_v1"}});
    auto ksvd = PlatosKey::load(p_svd, D_IN);
    CHECK(ksvd->metadata().tensor_layout == "svd_Ur_Vr");
    CHECK(ksvd->metadata().svd_rank == RANK);
    std::vector<float> x(D_IN), y(D_OUT, 0.0f);
    for (int i = 0; i < D_IN; ++i) x[i] = std::cos(static_cast<float>(i) * 0.11f);
    ksvd->project_cpu(x.data(), 1, y.data(), false);
    for (int j = 0; j < D_OUT; ++j) {
      double acc = 0.0;
      for (int r = 0; r < RANK; ++r) {
        double xr = 0.0;
        for (int i = 0; i < D_IN; ++i) xr += (static_cast<double>(x[i]) - dm[i]) * U[i * RANK + r];
        acc += xr * V[r * D_OUT + j];
      }
      const double ref = tm[j] + 1.5 * acc;
      CHECK(std::fabs(static_cast<double>(y[j]) - ref) < 1e-4);
    }

    const auto p_yue = tmp / "key_yue2.safetensors";
    write_safetensors(p_yue,
                      {{"W_yue2", "F32", {D_IN, D_OUT}, pack_f32(R)}, {"b_yue2", "F32", {D_OUT}, pack_f32(tm)}},
                      {{"format", "omni946-inkling-yue2-key/v1"}});
    auto kyue = PlatosKey::load(p_yue, D_IN);
    CHECK(kyue->metadata().tensor_layout == "affine_W_yue2");
    std::vector<float> yy(D_OUT, 0.0f);
    kyue->project_cpu(x.data(), 1, yy.data(), false);
    for (int j = 0; j < D_OUT; ++j) {
      double acc = tm[j];
      for (int i = 0; i < D_IN; ++i) acc += static_cast<double>(x[i]) * R[i * D_OUT + j];
      CHECK(std::fabs(static_cast<double>(yy[j]) - acc) < 1e-4);
    }
  }

  // 6) Inkling forbidden_coord_d2027_zero_masked: zeros x[2027] BEFORE centering so +Inf/outliers never poison y!
  {
    constexpr int D_INK = 2304, D_OUT_S = 8;
    std::vector<float> R_ink(D_INK * D_OUT_S, 0.0f);
    for (int j = 0; j < D_OUT_S; ++j) {
      R_ink[0 * D_OUT_S + j] = 1.0f;
      R_ink[2027 * D_OUT_S + j] = 2.0f;
    }
    const auto p_ink = tmp / "key_d2027.safetensors";
    const auto m_ink = tmp / "key_d2027.json";
    write_safetensors(p_ink, {{"R", "F32", {D_INK, D_OUT_S}, pack_f32(R_ink)}}, {{"format", "platos_key_v1"}});
    {
      std::ofstream mf(m_ink);
      mf << Json{{"forbidden_coord_d2027_zero_masked", true}}.dump();
    }
    auto kink = PlatosKey::load(p_ink, m_ink, D_INK);
    CHECK(kink->metadata().forbidden_coord_d2027_zero_masked == true);
    std::vector<float> x_ink(D_INK, 0.0f), y_ink(D_OUT_S, 0.0f);
    x_ink[0] = 3.5f;
    x_ink[2027] = std::numeric_limits<float>::infinity();
    kink->project_cpu(x_ink.data(), 1, y_ink.data(), false);
    for (int j = 0; j < D_OUT_S; ++j) {
      CHECK(std::isfinite(y_ink[j]));
      CHECK(std::fabs(y_ink[j] - 3.5f) < 1e-5f);
    }
  }

  // 7) Unsupported layouts fail closed: anti_cart (`U_bad`), multi-tap `concat_all4`, half-SVD, unknown key.
  {
    const auto p_anti = tmp / "anti.safetensors";
    write_safetensors(p_anti, {{"U_bad", "F32", {D_IN, RANK}, pack_f32(std::vector<float>(D_IN * RANK, 0.1f))}},
                      {{"format", "atlas_anti_cart_v1"}});
    bool threw_anti = false;
    try {
      (void)PlatosKey::load(p_anti, D_IN);
    } catch (const std::invalid_argument&) {
      threw_anti = true;
    }
    CHECK(threw_anti);

    const auto m_concat = tmp / "concat.json";
    std::ofstream mf(m_concat);
    mf << Json{{"cartridge_sha256", sha_a}, {"tap_spec", "concat_all4"}}.dump();
    mf.close();
    bool threw_concat = false;
    try {
      (void)make_served(1024, 8, p_a, m_concat);
    } catch (const std::invalid_argument&) {
      threw_concat = true;
    }
    CHECK(threw_concat);

    const auto p_half_svd = tmp / "half_svd.safetensors";
    write_safetensors(p_half_svd, {{"U_r", "F32", {D_IN, RANK}, pack_f32(std::vector<float>(D_IN * RANK, 0.1f))}});
    bool threw_half = false;
    try {
      (void)PlatosKey::load(p_half_svd, D_IN);
    } catch (const std::invalid_argument&) {
      threw_half = true;
    }
    CHECK(threw_half);

    const auto p_unk = tmp / "unknown.safetensors";
    write_safetensors(p_unk, {{"mystery_matrix", "F32", {D_IN, D_OUT}, pack_f32(R)}});
    bool threw_unk = false;
    try {
      (void)PlatosKey::load(p_unk, D_IN);
    } catch (const std::invalid_argument&) {
      threw_unk = true;
    }
    CHECK(threw_unk);
  }

  std::filesystem::remove_all(tmp);
}

void test_http_400_validation() {
  auto s = make_served(/*max_model_len=*/8);
  auto check_400 = [&](const std::string& raw_body, const std::string& expected_param = "") {
    const Response r = http_req(s->port(), "POST", "/v1/embeddings", raw_body);
    CHECK(r.status == 400);
    const Json j = Json::parse(r.body);
    CHECK(j["error"]["type"] == "invalid_request_error");
    CHECK(j["error"]["code"] == 400);
    if (!expected_param.empty()) CHECK(j["error"]["param"] == expected_param);
  };

  check_400("{not-json");
  check_400("[]");
  check_400(Json::object().dump(), "input");
  check_400(Json{{"input", nullptr}}.dump(), "input");
  check_400(Json{{"input", ""}}.dump(), "input");
  check_400(Json{{"input", Json::array()}}.dump(), "input");
  check_400(Json{{"input", Json::array({""})}}.dump(), "input");
  check_400(Json{{"input", 12345}}.dump(), "input");
  check_400(Json{{"input", Json::array({"ok", 42})}}.dump(), "input");
  check_400(Json{{"input", "one two three four five six seven eight nine ten eleven twelve"}}.dump(), "input");
  check_400(Json{{"input", "ok"}, {"model", 123}}.dump(), "model");
  check_400(Json{{"input", "ok"}, {"encoding_format", "int8"}}.dump(), "encoding_format");
  check_400(Json{{"input", "ok"}, {"encoding_format", 7}}.dump(), "encoding_format");
  check_400(Json{{"input", "ok"}, {"pooling", "max"}}.dump(), "pooling");
  check_400(Json{{"input", "ok"}, {"pooling", false}}.dump(), "pooling");
  check_400(Json{{"input", "ok"}, {"normalize", "true"}}.dump(), "normalize");
  check_400(Json{{"input", "ok"}, {"tap_layer", -1}}.dump(), "tap_layer");
  check_400(Json{{"input", "ok"}, {"tap_layer", 48}}.dump(), "tap_layer");
  check_400(Json{{"input", "ok"}, {"tap_layer", 1.5}}.dump(), "tap_layer");
  check_400(Json{{"input", "ok"}, {"tap_layer", "31"}}.dump(), "tap_layer");
  check_400(Json{{"input", "ok"}, {"platos_key", "yes"}}.dump(), "platos_key");
  check_400(Json{{"input", "ok"}, {"platos_key", true}}.dump(), "platos_key");

  // Generator without embedding support returns HTTP 400.
  auto s_off = make_served(1024, 4, {}, {}, /*enable_embeddings=*/false);
  const Response r_off = http_req(s_off->port(), "POST", "/v1/embeddings", Json{{"input", "hello"}}.dump());
  CHECK(r_off.status == 400);
}

// Batched continuous-slot test generator verifying zero KV-slot allocation/leak and fair interleaving
// between concurrent /v1/chat/completions streams and /v1/embeddings requests.
class SlotTrackingGenerator final : public ingine::model::Generator {
 public:
  explicit SlotTrackingGenerator(const ingine::text::Tokenizer& tok)
      : tok_(tok), reply_(tok.encode("<|message_model|><|content_text|>alpha beta gamma delta epsilon<|end_message|><|content_model_end_sampling|>", true)) {}

  std::string model_id() const override { return "slot-tracking-batched"; }
  int max_model_len() const override { return 4096; }
  int max_batch_size() const override { return 4; }
  bool supports_embeddings() const override { return true; }
  int embedding_dim() const override { return 16; }
  int num_tap_layers() const override { return 48; }

  bool can_admit_slot(int, int) const override {
    std::lock_guard lock(mu_);
    return active_count_locked() < 4;
  }

  int start_slot(const ingine::model::Prompt&, int, const ingine::model::Sampling&, int32_t& first_token,
                 ingine::model::GenerateStats& stats) override {
    std::lock_guard lock(mu_);
    for (int s = 0; s < 4; ++s) {
      if (!in_use_[s]) {
        in_use_[s] = true;
        step_idx_[s] = 1;
        kv_cookie_[s] = ++next_cookie_;
        ++starts_;
        peak_active_ = std::max(peak_active_, active_count_locked());
        first_token = reply_[0];
        stats.prompt_tokens = 4;
        return s;
      }
    }
    return -1;
  }

  void step_slots(std::span<const ingine::model::BatchSlotStep> inputs, std::span<int32_t> out_tokens) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::lock_guard lock(mu_);
    ++decode_steps_;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const int s = inputs[i].slot;
      CHECK(s >= 0 && s < 4 && in_use_[s] && kv_cookie_[s] > 0);
      const std::size_t idx = std::min<std::size_t>(step_idx_[s]++, reply_.size() - 1);
      out_tokens[i] = reply_[idx];
    }
  }

  void release_slot(int slot) override {
    std::lock_guard lock(mu_);
    CHECK(slot >= 0 && slot < 4 && in_use_[slot]);
    in_use_[slot] = false;
    kv_cookie_[slot] = 0;
    ++releases_;
  }

  ingine::model::GenerateStats generate(std::span<const int32_t>, int, const ingine::model::Sampling&,
                                        std::span<const int32_t>,
                                        const std::function<bool(std::span<const int32_t>)>&) override {
    throw std::logic_error("batched scheduler should use slot API");
  }

  ingine::model::EmbeddingResult embed(const ingine::model::EmbeddingRequest& req) override {
    std::array<std::uint64_t, 4> cookies_before{};
    int active_now = 0;
    {
      std::lock_guard lock(mu_);
      cookies_before = kv_cookie_;
      active_now = active_count_locked();
      if (active_now > 0) ++embeds_while_chat_live_;
      ++embed_calls_;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ingine::model::EmbeddingResult res;
    int total = 0;
    for (const auto& seq : req.inputs) {
      total += static_cast<int>(seq.size());
      std::vector<float> v(16, 0.0f);
      for (int d = 0; d < 16; ++d) v[d] = static_cast<float>((seq[0] + d) % 17 + 1);
      if (req.normalize) ingine::l2_normalize_rows_cpu(v.data(), 1, 16);
      res.embeddings.push_back(std::move(v));
    }
    res.stats.total_prompt_tokens = total;
    res.stats.output_dim = 16;
    res.stats.effective_tap_layer = req.tap_layer.value_or(-1);
    res.stats.kv_slots_allocated = 0;
    res.stats.lm_head_skipped = true;
    {
      std::lock_guard lock(mu_);
      // Zero KV slots allocated and existing live chat slots' KV cookies completely untouched.
      CHECK(kv_cookie_ == cookies_before);
      CHECK(active_count_locked() == active_now);
    }
    return res;
  }

  int active_slots() const {
    std::lock_guard lock(mu_);
    return active_count_locked();
  }
  int starts() const {
    std::lock_guard lock(mu_);
    return starts_;
  }
  int releases() const {
    std::lock_guard lock(mu_);
    return releases_;
  }
  int embed_calls() const {
    std::lock_guard lock(mu_);
    return embed_calls_;
  }
  int embeds_while_chat_live() const {
    std::lock_guard lock(mu_);
    return embeds_while_chat_live_;
  }

 private:
  int active_count_locked() const {
    int n = 0;
    for (bool u : in_use_)
      if (u) ++n;
    return n;
  }

  const ingine::text::Tokenizer& tok_;
  std::vector<int32_t> reply_;
  mutable std::mutex mu_;
  std::array<bool, 4> in_use_{};
  std::array<std::size_t, 4> step_idx_{};
  std::array<std::uint64_t, 4> kv_cookie_{};
  std::uint64_t next_cookie_ = 0;
  int starts_ = 0;
  int releases_ = 0;
  int peak_active_ = 0;
  int decode_steps_ = 0;
  int embed_calls_ = 0;
  int embeds_while_chat_live_ = 0;
};

void test_mixed_concurrent_chat_and_embeddings_zero_kv_leak() {
  SlotTrackingGenerator gen(g_fmt->tokenizer());
  ingine::serve::ServerOptions opts;
  opts.max_queue = 16;
  ChatServer srv(gen, *g_fmt, opts);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();
  const int port = srv.port();

  std::atomic<int> chat_ok{0}, embed_ok{0}, errors{0};
  std::vector<std::thread> threads;

  // Launch 4 concurrent chat completions alongside 8 concurrent batched embedding requests.
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i] {
      try {
        const Json req = {{"messages", Json::array({{{"role", "user"}, {"content", "Chat #" + std::to_string(i)}}})},
                          {"max_tokens", 32}};
        const Response r = http_req(port, "POST", "/v1/chat/completions", req.dump());
        if (r.status == 200) {
          const Json j = Json::parse(r.body);
          if (j["choices"][0]["message"]["content"] == "alpha beta gamma delta epsilon") ++chat_ok;
          else ++errors;
        } else {
          ++errors;
        }
      } catch (...) {
        ++errors;
      }
    });
  }
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] {
      try {
        const Json req = {{"input", Json::array({"query " + std::to_string(i), "doc " + std::to_string(i)})},
                          {"tap_layer", 23}};
        const Response r = http_req(port, "POST", "/v1/embeddings", req.dump());
        if (r.status == 200 && header_val(r.headers, "X-Ingine-Kv-Slots-Allocated") == "0" &&
            header_val(r.headers, "X-Ingine-Lm-Head-Skipped") == "true") {
          const Json j = Json::parse(r.body);
          if (j["data"].size() == 2 && j["data"][0]["embedding"].size() == 16) ++embed_ok;
          else ++errors;
        } else {
          ++errors;
        }
      } catch (...) {
        ++errors;
      }
    });
  }
  for (auto& t : threads) t.join();
  srv.stop();

  CHECK(errors.load() == 0);
  CHECK(chat_ok.load() == 4);
  CHECK(embed_ok.load() == 8);
  CHECK(gen.starts() == 4);
  CHECK(gen.releases() == 4);
  CHECK(gen.active_slots() == 0);  // ZERO KV-slot leak!
  CHECK(gen.embed_calls() == 8);
  std::printf("mixed_chat_embed: chats=%d embeds=%d slots_started=%d slots_released=%d embeds_during_live_chat=%d\n",
              chat_ok.load(), embed_ok.load(), gen.starts(), gen.releases(), gen.embeds_while_chat_live());
}

void test_platos_key_fp16_subnormals_0_ulp() {
  const auto tmp = std::filesystem::temp_directory_path() / "ingine_test_embeddings_subnormals_963";
  std::filesystem::create_directories(tmp);
  const auto p_f16 = tmp / "subnormal_f16.safetensors";
  const auto p_f32 = tmp / "subnormal_f32.safetensors";

  // Include FP16 subnormals (0x0001 = 2^-24, 0x0080 = 2^-17, 0x03ff = max subnormal, 0x83ff = -max subnormal),
  // smallest normal (0x0400 = 2^-14), +/-0, and ridge-regression-scale values (87.5, -9.59375).
  const std::vector<std::uint16_t> raw_bits = {
      0x0001u, 0x0080u, 0x03ffu, 0x83ffu,  // row 0
      0x0400u, 0x8400u, 0x0000u, 0x5578u,  // row 1 (0x5578 = 87.5)
  };
  std::vector<std::uint8_t> f16_bytes(raw_bits.size() * 2);
  std::vector<float> f32_vals(raw_bits.size());
  for (std::size_t k = 0; k < raw_bits.size(); ++k) {
    f16_bytes[2 * k] = static_cast<std::uint8_t>(raw_bits[k] & 0xffu);
    f16_bytes[2 * k + 1] = static_cast<std::uint8_t>((raw_bits[k] >> 8) & 0xffu);
    const std::uint16_t b = raw_bits[k];
    const int s = (b & 0x8000u) ? -1 : 1;
    const int e = (b >> 10) & 0x1f;
    const int m = b & 0x3ff;
    f32_vals[k] = (e == 0) ? (s * std::ldexp(static_cast<float>(m), -24))
                           : (s * std::ldexp(static_cast<float>(1024 + m), e - 25));
  }
  std::vector<std::uint8_t> f32_bytes(f32_vals.size() * 4);
  std::memcpy(f32_bytes.data(), f32_vals.data(), f32_bytes.size());

  write_safetensors(p_f16, {{"R", "F16", {2, 4}, f16_bytes}}, {{"format", "platos_key_v1"}});
  write_safetensors(p_f32, {{"R", "F32", {2, 4}, f32_bytes}}, {{"format", "platos_key_v1"}});

  auto k_f16 = PlatosKey::load(p_f16, 2);
  auto k_f32 = PlatosKey::load(p_f32, 2);
  const auto& rt16 = k_f16->weights_f16_transposed();
  const auto& rt32 = k_f32->weights_f16_transposed();
  CHECK(rt16.size() == 8 && rt32.size() == 8);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      const std::uint16_t expect = raw_bits[static_cast<std::size_t>(i * 4 + j)];
      CHECK(rt16[static_cast<std::size_t>(j * 2 + i)] == expect);
      CHECK(rt32[static_cast<std::size_t>(j * 2 + i)] == expect);
    }
  }
  std::filesystem::remove_all(tmp);
}

}  // namespace

int main() {
  const char* dir = std::getenv("INGINE_MODEL_DIR");
  if (!dir || !*dir) {
    for (const char* fallback :
         {"/mnt/storage/models/Inkling-2-Mini-ingine", "/mnt/nvme/models/Inkling-2-Mini-ingine"}) {
      if (std::filesystem::is_regular_file(std::filesystem::path(fallback) / "tokenizer.json")) {
        dir = fallback;
        break;
      }
    }
  }
  if (!dir || !*dir) {
    std::fprintf(stderr, "INGINE_MODEL_DIR not set and no default tokenizer found; skipping\n");
    return 77;
  }

  const ChatFormat fmt = ChatFormat::load(dir);
  g_fmt = &fmt;

  try {
    test_platos_key_loader_unit();
    test_platos_key_fp16_subnormals_0_ulp();
    test_openai_schema_batch_and_base64();
    test_atlas_extensions_and_platos_key_http();
    test_cartridge_964_binding_and_layouts_fail_closed();
    test_http_400_validation();
    test_mixed_concurrent_chat_and_embeddings_zero_kv_leak();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_embeddings failed: %s\n", e.what());
    return 1;
  }
  std::printf(g_failures ? "FAILED (%d checks)\n" : "ok\n", g_failures);
  return g_failures ? 1 : 0;
}
