#include "ingine/model_backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ingine::model {
namespace {

std::uint64_t fnv1a_mix(std::uint64_t h, const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= std::uint64_t(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}

void fnv1a_mix_string(std::uint64_t& h, std::string_view s) noexcept {
  const std::size_t len = s.size();
  h = fnv1a_mix(h, &len, sizeof(len));
  if (len > 0) h = fnv1a_mix(h, s.data(), len);
}

std::string to_lower_ascii(std::string_view s) {
  std::string out(s);
  for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

std::string read_file_bytes(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool is_zero_sha256(const std::array<std::uint8_t, 32>& d) noexcept {
  for (std::uint8_t b : d) {
    if (b != 0) return false;
  }
  return true;
}

class Sha256Local {
 public:
  Sha256Local() { reset(); }

  void reset() {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    total_bits_ = 0;
    buf_len_ = 0;
  }

  void update(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bits_ += static_cast<std::uint64_t>(len) * 8ull;
    while (len > 0) {
      const std::size_t take = std::min<std::size_t>(len, 64 - buf_len_);
      std::memcpy(buf_.data() + buf_len_, p, take);
      buf_len_ += take;
      p += take;
      len -= take;
      if (buf_len_ == 64) {
        compress(buf_.data());
        buf_len_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finalize() {
    buf_[buf_len_++] = 0x80u;
    if (buf_len_ > 56) {
      while (buf_len_ < 64) buf_[buf_len_++] = 0;
      compress(buf_.data());
      buf_len_ = 0;
    }
    while (buf_len_ < 56) buf_[buf_len_++] = 0;
    for (int i = 0; i < 8; ++i) {
      buf_[56 + i] = static_cast<std::uint8_t>((total_bits_ >> (56 - 8 * i)) & 0xffu);
    }
    compress(buf_.data());
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[4 * i + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xffu);
      out[4 * i + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xffu);
      out[4 * i + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xffu);
      out[4 * i + 3] = static_cast<std::uint8_t>(state_[i] & 0xffu);
    }
    return out;
  }

 private:
  static inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept { return (x >> n) | (x << (32 - n)); }

  void compress(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> kK = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i + 0]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + kK[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::uint64_t total_bits_ = 0;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
};

bool is_auxiliary_gguf_arch(std::string_view tag) {
  const std::string lower = to_lower_ascii(tag);
  return lower == "dflash" || lower == "clip" || lower == "mmproj" || lower == "projector" || lower == "mtp" ||
         lower == "draft";
}

// Exact supported HF config / Safetensors family tags ONLY (never fuzzy prefix matches that could alias V3/GLM-4).
ModelFamily classify_safetensors_tag(std::string_view tag) {
  const std::string lower = to_lower_ascii(tag);
  if (lower == "inkling2" || lower == "inkling_2" || lower == "inkling2_loop" || lower == "inkling2_loop_text" ||
      lower == "inkling2forconditionalgeneration" || lower == "inkling2forcausallm" ||
      lower == "inkling2loopforconditionalgeneration" || lower == "inkling2loopforcausallm") {
    return ModelFamily::Inkling2;
  }
  if (lower == "qwen3_5" || lower == "qwen3_5_text" || lower == "qwen3_5forconditionalgeneration" ||
      lower == "qwen3_5forcausallm") {
    return ModelFamily::Qwen3_5;
  }
  if (lower == "qwen4exp" || lower == "qwen4_exp" || lower == "qwen4expforconditionalgeneration" ||
      lower == "qwen4expforcausallm" || lower == "qwen3_8_flash_next") {
    return ModelFamily::Qwen4Exp;
  }
  if (lower == "deepseek_v41" || lower == "deepseek_v41_text" || lower == "deepseekv41forcausallm" ||
      lower == "deepseekv41forconditionalgeneration") {
    return ModelFamily::DeepSeekV4;
  }
  if (lower == "glm5_next" || lower == "glm5_next_text" || lower == "glm5-next" ||
      lower == "glm5nextforconditionalgeneration" || lower == "glm5nextforcausallm") {
    return ModelFamily::GlmFlash;
  }
  if (lower == "embedding_gemma2" || lower == "embedding_gemma2_text" || lower == "embeddinggemma2model" ||
      lower == "embeddinggemma2") {
    return ModelFamily::EmbeddingGemma2;
  }
  return ModelFamily::Unknown;
}

// Exact supported GGUF general.architecture tags ONLY.
ModelFamily classify_gguf_arch(std::string_view arch) {
  const std::string lower = to_lower_ascii(arch);
  if (lower == "inkling2_loop") return ModelFamily::Inkling2;
  if (lower == "qwen35") return ModelFamily::Qwen3_5;
  if (lower == "qwen4exp") return ModelFamily::Qwen4Exp;
  if (lower == "deepseek41") return ModelFamily::DeepSeekV4;
  if (lower == "glm5-next" || lower == "glm5_next") return ModelFamily::GlmFlash;
  return ModelFamily::Unknown;
}

std::uint32_t gguf_read_u32(std::ifstream& in, std::uint64_t& pos, std::uint64_t file_size) {
  if (file_size - pos < 4) throw std::invalid_argument("truncated GGUF header (u32)");
  std::array<std::uint8_t, 4> b{};
  in.read(reinterpret_cast<char*>(b.data()), 4);
  if (!in) throw std::invalid_argument("read error in GGUF header (u32)");
  pos += 4;
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t gguf_read_u64(std::ifstream& in, std::uint64_t& pos, std::uint64_t file_size) {
  if (file_size - pos < 8) throw std::invalid_argument("truncated GGUF header (u64)");
  std::array<std::uint8_t, 8> b{};
  in.read(reinterpret_cast<char*>(b.data()), 8);
  if (!in) throw std::invalid_argument("read error in GGUF header (u64)");
  pos += 8;
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
  return v;
}

void gguf_skip_bytes(std::ifstream& in, std::uint64_t& pos, std::uint64_t file_size, std::uint64_t n) {
  if (n > file_size - pos) throw std::invalid_argument("truncated GGUF metadata payload");
  if (n > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::invalid_argument("GGUF metadata skip overflow");
  }
  in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
  if (!in) throw std::invalid_argument("seek error in GGUF header");
  pos += n;
}

std::string gguf_read_str(std::ifstream& in, std::uint64_t& pos, std::uint64_t file_size) {
  const std::uint64_t len = gguf_read_u64(in, pos, file_size);
  if (len > file_size - pos || len > 16ull * 1024ull * 1024ull) {
    throw std::invalid_argument("invalid GGUF string length");
  }
  std::string s(len, '\0');
  if (len > 0) {
    in.read(s.data(), static_cast<std::streamsize>(len));
    if (!in) throw std::invalid_argument("read error in GGUF string");
    pos += len;
  }
  return s;
}

std::size_t gguf_scalar_size(std::uint32_t vtype) noexcept {
  switch (vtype) {
    case 0:   // UINT8
    case 1:   // INT8
    case 7:   // BOOL
      return 1;
    case 2:   // UINT16
    case 3:   // INT16
      return 2;
    case 4:   // UINT32
    case 5:   // INT32
    case 6:   // FLOAT32
      return 4;
    case 10:  // UINT64
    case 11:  // INT64
    case 12:  // FLOAT64
      return 8;
    default: return 0;
  }
}

void gguf_skip_value(std::ifstream& in, std::uint64_t& pos, std::uint64_t file_size, std::uint32_t vtype, int depth) {
  if (depth > 8) throw std::invalid_argument("excessive GGUF metadata array nesting");
  if (const std::size_t scalar_sz = gguf_scalar_size(vtype); scalar_sz > 0) {
    gguf_skip_bytes(in, pos, file_size, scalar_sz);
    return;
  }
  if (vtype == 8) {  // STRING
    const std::uint64_t len = gguf_read_u64(in, pos, file_size);
    gguf_skip_bytes(in, pos, file_size, len);
    return;
  }
  if (vtype == 9) {  // ARRAY
    const std::uint32_t elem_type = gguf_read_u32(in, pos, file_size);
    const std::uint64_t count = gguf_read_u64(in, pos, file_size);
    if (const std::size_t elem_sz = gguf_scalar_size(elem_type); elem_sz > 0) {
      if (count > (file_size - pos) / elem_sz) {
        throw std::invalid_argument("GGUF array bounds exceed file size");
      }
      gguf_skip_bytes(in, pos, file_size, count * elem_sz);
      return;
    }
    if (count > 10000000ull) throw std::invalid_argument("GGUF array count exceeds sanity cap");
    for (std::uint64_t i = 0; i < count; ++i) {
      gguf_skip_value(in, pos, file_size, elem_type, depth + 1);
    }
    return;
  }
  throw std::invalid_argument("unknown GGUF metadata type " + std::to_string(vtype));
}

std::optional<std::string> peek_gguf_architecture(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::invalid_argument("cannot open GGUF file: " + path.string());
  const std::streamoff end_pos = in.tellg();
  if (end_pos < 24) throw std::invalid_argument("GGUF file truncated (< 24 bytes): " + path.string());
  const std::uint64_t file_size = static_cast<std::uint64_t>(end_pos);
  in.seekg(0, std::ios::beg);

  std::array<char, 4> magic{};
  in.read(magic.data(), 4);
  if (!in || std::string_view(magic.data(), 4) != "GGUF") {
    throw std::invalid_argument("invalid GGUF magic in " + path.string());
  }
  std::uint64_t pos = 4;
  const std::uint32_t version = gguf_read_u32(in, pos, file_size);
  if (version != 2 && version != 3) {
    throw std::invalid_argument("unsupported GGUF version " + std::to_string(version) + " in " + path.string());
  }
  (void)gguf_read_u64(in, pos, file_size);  // tensor_count
  const std::uint64_t kv_count = gguf_read_u64(in, pos, file_size);
  if (kv_count > 1000000ull) {
    throw std::invalid_argument("GGUF metadata kv_count exceeds sanity cap in " + path.string());
  }

  for (std::uint64_t i = 0; i < kv_count; ++i) {
    const std::string key = gguf_read_str(in, pos, file_size);
    const std::uint32_t vtype = gguf_read_u32(in, pos, file_size);
    if (key == "general.architecture") {
      if (vtype != 8) {
        throw std::invalid_argument("GGUF general.architecture must be string in " + path.string());
      }
      return gguf_read_str(in, pos, file_size);
    }
    gguf_skip_value(in, pos, file_size, vtype, 0);
  }
  return std::nullopt;
}

struct RegistryState {
  std::mutex mu;
  std::unordered_map<std::uint8_t, BackendRegistration> by_family;
};

RegistryState& registry_state() {
  static RegistryState state;
  return state;
}

}  // namespace

const char* to_string(ModelFamily family) noexcept {
  switch (family) {
    case ModelFamily::Inkling2: return "inkling2";
    case ModelFamily::Qwen3_5: return "qwen3_5";
    case ModelFamily::DeepSeekV4: return "deepseek_v4";
    case ModelFamily::GlmFlash: return "glm_flash";
    case ModelFamily::Qwen4Exp: return "qwen4exp";
    case ModelFamily::EmbeddingGemma2: return "embedding_gemma2";
    default: return "unknown";
  }
}

const char* to_string(CheckpointFormat fmt) noexcept {
  switch (fmt) {
    case CheckpointFormat::Safetensors: return "safetensors";
    case CheckpointFormat::Gguf: return "gguf";
    default: return "unknown";
  }
}

const char* to_string(ActivationDtype dt) noexcept {
  switch (dt) {
    case ActivationDtype::Bf16: return "bf16";
    case ActivationDtype::Fp16: return "fp16";
    case ActivationDtype::Fp32: return "fp32";
    default: return "unknown";
  }
}

const char* to_string(VerificationLevel level) noexcept {
  switch (level) {
    case VerificationLevel::HeadersOnly: return "headers_only";
    case VerificationLevel::SelfHashed: return "self_hashed";
    case VerificationLevel::PinnedVerified: return "pinned_verified";
    default: return "unverified";
  }
}

std::array<std::uint8_t, 32> parse_sha256_hex(std::string_view hex) {
  if (hex.size() != 64) {
    throw std::invalid_argument("sha256 hex string must be exactly 64 hex chars");
  }
  auto nibble = [](char ch) -> std::uint8_t {
    if (ch >= '0' && ch <= '9') return static_cast<std::uint8_t>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return static_cast<std::uint8_t>(10 + (ch - 'a'));
    if (ch >= 'A' && ch <= 'F') return static_cast<std::uint8_t>(10 + (ch - 'A'));
    throw std::invalid_argument("invalid hex character in sha256 digest");
  };
  std::array<std::uint8_t, 32> out{};
  for (std::size_t i = 0; i < 32; ++i) {
    out[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
  }
  return out;
}

ModelFamily parse_model_family(std::string_view s) {
  const std::string lower = to_lower_ascii(s);
  if (lower == "inkling2" || lower == "inkling-2") return ModelFamily::Inkling2;
  if (lower == "qwen3_5" || lower == "qwen35" || lower == "qwen3.5" || lower == "qwen3_5_text" ||
      lower == "qwen38" || lower == "qwen3.8") {
    return ModelFamily::Qwen3_5;
  }
  if (lower == "qwen4exp" || lower == "qwen4_exp" || lower == "qwen3.8-flash-next" ||
      lower == "qwen38-flash-next" || lower == "qwen3_8_flash_next") {
    return ModelFamily::Qwen4Exp;
  }
  if (lower == "deepseek_v4" || lower == "deepseek_v41" || lower == "deepseek41") {
    return ModelFamily::DeepSeekV4;
  }
  if (lower == "glm_flash" || lower == "glm53" || lower == "glm_53_flash" || lower == "glm5-next" ||
      lower == "glm5_next") {
    return ModelFamily::GlmFlash;
  }
  if (lower == "embedding_gemma2" || lower == "embeddinggemma2" || lower == "embeddinggemma-2" ||
      lower == "embedding-gemma-2") {
    return ModelFamily::EmbeddingGemma2;
  }
  throw std::invalid_argument("unknown model family '" + std::string(s) + "'");
}

ModelFamily detect_model_family(const std::filesystem::path& model_path) {
  if (std::filesystem::is_regular_file(model_path)) {
    if (model_path.extension() != ".gguf") {
      throw std::invalid_argument("expected checkpoint directory or .gguf file, got: " + model_path.string());
    }
    const auto arch = peek_gguf_architecture(model_path);
    if (!arch.has_value()) {
      throw std::invalid_argument("GGUF file lacks general.architecture (continuation shard?): " + model_path.string());
    }
    if (is_auxiliary_gguf_arch(*arch)) {
      throw std::invalid_argument("auxiliary non-backbone GGUF architecture '" + *arch + "' in " + model_path.string());
    }
    const ModelFamily fam = classify_gguf_arch(*arch);
    if (fam == ModelFamily::Unknown) {
      throw std::invalid_argument("unrecognized GGUF architecture '" + *arch + "' in " + model_path.string());
    }
    return fam;
  }
  if (!std::filesystem::is_directory(model_path)) {
    throw std::invalid_argument("model path does not exist: " + model_path.string());
  }

  // Gather any GGUF shards in the directory upfront so mixed/ambiguous directories are always caught.
  std::vector<std::filesystem::path> gguf_files;
  bool has_safetensors = false;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(model_path, ec)) {
    if (ec) {
      throw std::invalid_argument("cannot scan model directory " + model_path.string() + ": " + ec.message());
    }
    if (!entry.is_regular_file(ec)) continue;
    const std::string name = entry.path().filename().string();
    if (name == "model.safetensors" || name == "model.safetensors.index.json" ||
        entry.path().extension() == ".safetensors") {
      has_safetensors = true;
    } else if (entry.path().extension() == ".gguf") {
      gguf_files.push_back(entry.path());
    }
  }
  if (has_safetensors && !gguf_files.empty()) {
    throw std::invalid_argument("ambiguous checkpoint directory contains both safetensors and gguf: " +
                                model_path.string());
  }

  std::optional<ModelFamily> cfg_family;
  const std::filesystem::path cfg_path = model_path / "config.json";
  if (std::filesystem::is_regular_file(cfg_path)) {
    nlohmann::json cfg;
    try {
      cfg = nlohmann::json::parse(read_file_bytes(cfg_path));
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid config.json in " + model_path.string() + ": " + e.what());
    }
    if (!cfg.is_object()) {
      throw std::invalid_argument("config.json in " + model_path.string() + " must be a JSON object");
    }
    auto record_tag = [&](const std::string& tag, const char* src) {
      const ModelFamily fam = classify_safetensors_tag(tag);
      if (fam == ModelFamily::Unknown) {
        throw std::invalid_argument(std::string("unsupported model family tag '") + tag + "' in " + src + " (" +
                                    cfg_path.string() + ")");
      }
      if (cfg_family.has_value() && *cfg_family != fam) {
        throw std::invalid_argument("conflicting model family tags in " + cfg_path.string());
      }
      cfg_family = fam;
    };
    if (auto it = cfg.find("architectures"); it != cfg.end()) {
      if (!it->is_array()) throw std::invalid_argument("config.json architectures must be an array");
      for (const auto& arch : *it) {
        if (!arch.is_string()) throw std::invalid_argument("config.json architecture entry must be string");
        record_tag(arch.get<std::string>(), "architectures");
      }
    }
    if (auto it = cfg.find("model_type"); it != cfg.end()) {
      if (!it->is_string()) throw std::invalid_argument("config.json model_type must be string");
      record_tag(it->get<std::string>(), "model_type");
    }
    if (auto tc = cfg.find("text_config"); tc != cfg.end()) {
      if (!tc->is_object()) throw std::invalid_argument("config.json text_config must be object");
      if (auto mt = tc->find("model_type"); mt != tc->end()) {
        if (!mt->is_string()) throw std::invalid_argument("config.json text_config.model_type must be string");
        record_tag(mt->get<std::string>(), "text_config.model_type");
      }
    }
    if (!cfg_family.has_value()) {
      throw std::invalid_argument("no recognized model family tag in " + cfg_path.string());
    }
  }

  std::optional<ModelFamily> gguf_family;
  if (!gguf_files.empty()) {
    std::sort(gguf_files.begin(), gguf_files.end());
    for (const auto& gp : gguf_files) {
      const auto arch = peek_gguf_architecture(gp);
      if (!arch.has_value()) continue;  // continuation shard without general.architecture
      if (is_auxiliary_gguf_arch(*arch)) {
        throw std::invalid_argument("auxiliary non-backbone GGUF architecture '" + *arch + "' in " + gp.string());
      }
      const ModelFamily fam = classify_gguf_arch(*arch);
      if (fam == ModelFamily::Unknown) {
        throw std::invalid_argument("unrecognized GGUF architecture '" + *arch + "' in " + gp.string());
      }
      if (gguf_family.has_value() && *gguf_family != fam) {
        throw std::invalid_argument("conflicting GGUF model families in directory " + model_path.string());
      }
      gguf_family = fam;
    }
    if (!gguf_family.has_value()) {
      throw std::invalid_argument("GGUF directory contains no shard with general.architecture: " + model_path.string());
    }
  }

  if (cfg_family.has_value() && gguf_family.has_value()) {
    if (*cfg_family != *gguf_family) {
      throw std::invalid_argument("config.json and GGUF disagree on model family in " + model_path.string());
    }
    return *cfg_family;
  }
  if (cfg_family.has_value()) return *cfg_family;
  if (gguf_family.has_value()) return *gguf_family;

  // Inkling checkpoints may omit config.json and ship tokenizer.json + chat_template.jinja.
  const std::filesystem::path tok_path = model_path / "tokenizer.json";
  if (std::filesystem::is_regular_file(tok_path)) {
    nlohmann::json tok;
    try {
      tok = nlohmann::json::parse(read_file_bytes(tok_path));
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid tokenizer.json in " + model_path.string() + ": " + e.what());
    }
    if (auto added = tok.find("added_tokens"); added != tok.end() && added->is_array()) {
      for (const auto& item : *added) {
        if (item.is_object() && item.value("content", std::string{}) == "<|message_model|>") {
          return ModelFamily::Inkling2;
        }
      }
    }
  }
  throw std::invalid_argument("cannot detect model family in " + model_path.string() +
                              " (missing config.json / backbone GGUF)");
}

CheckpointFormat detect_checkpoint_format(const std::filesystem::path& model_path) {
  if (std::filesystem::is_regular_file(model_path)) {
    const std::string name = model_path.filename().string();
    if (model_path.extension() == ".gguf") return CheckpointFormat::Gguf;
    if (name == "model.safetensors.index.json" || model_path.extension() == ".safetensors") {
      return CheckpointFormat::Safetensors;
    }
    throw std::invalid_argument("unsupported checkpoint file extension: " + model_path.string());
  }
  if (!std::filesystem::is_directory(model_path)) {
    throw std::invalid_argument("checkpoint path does not exist: " + model_path.string());
  }
  bool has_safetensors = false;
  bool has_gguf = false;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(model_path, ec)) {
    if (ec) {
      throw std::invalid_argument("cannot scan checkpoint directory " + model_path.string() + ": " + ec.message());
    }
    if (!entry.is_regular_file(ec)) continue;
    const std::string name = entry.path().filename().string();
    if (name == "model.safetensors" || name == "model.safetensors.index.json" ||
        entry.path().extension() == ".safetensors") {
      has_safetensors = true;
    } else if (entry.path().extension() == ".gguf") {
      has_gguf = true;
    }
  }
  if (has_safetensors && has_gguf) {
    throw std::invalid_argument("ambiguous checkpoint directory contains both safetensors and gguf: " +
                                model_path.string());
  }
  if (has_safetensors) return CheckpointFormat::Safetensors;
  if (has_gguf) return CheckpointFormat::Gguf;
  throw std::invalid_argument("no safetensors or gguf weights found in " + model_path.string());
}

bool BackendStateIdentity::is_reusable_with(const BackendStateIdentity& other) const noexcept {
  if (family == ModelFamily::Unknown || other.family == ModelFamily::Unknown) return false;
  if (backend_name.empty() || other.backend_name.empty()) return false;
  if (checkpoint.format == CheckpointFormat::Unknown || other.checkpoint.format == CheckpointFormat::Unknown) {
    return false;
  }
  if (checkpoint.verification_level != VerificationLevel::PinnedVerified ||
      other.checkpoint.verification_level != VerificationLevel::PinnedVerified) {
    return false;
  }
  if (checkpoint.model_id.empty() || other.checkpoint.model_id.empty()) return false;
  if (checkpoint.schema_variant.empty() || other.checkpoint.schema_variant.empty()) return false;
  if (checkpoint.shard_count == 0 || other.checkpoint.shard_count == 0) return false;
  if (!checkpoint.weight_manifest_verified || !other.checkpoint.weight_manifest_verified) return false;
  if (!checkpoint.shard_digests_verified || !other.checkpoint.shard_digests_verified) return false;
  if (is_zero_sha256(checkpoint.weight_manifest_sha256) ||
      is_zero_sha256(other.checkpoint.weight_manifest_sha256)) {
    return false;
  }
  if (is_zero_sha256(checkpoint.shard_set_sha256) || is_zero_sha256(other.checkpoint.shard_set_sha256)) {
    return false;
  }
  if (state_layout_id.empty() || other.state_layout_id.empty()) return false;
  if (act_dtype == ActivationDtype::Unknown || other.act_dtype == ActivationDtype::Unknown) return false;
  if (numeric_abi.empty() || other.numeric_abi.empty()) return false;
  if (weight_precision.empty() || other.weight_precision.empty()) return false;
  if (kv_format.empty() || other.kv_format.empty()) return false;
  if (kv_cache_mode.empty() || other.kv_cache_mode.empty()) return false;
  if (gear.empty() || other.gear.empty()) return false;
  if (max_ctx <= 0 || other.max_ctx <= 0 || prefill_chunk <= 0 || other.prefill_chunk <= 0) return false;
  if ((persona_key != 0) != !persona_fingerprint.empty() ||
      (other.persona_key != 0) != !other.persona_fingerprint.empty()) {
    return false;
  }
  return *this == other;
}

std::uint64_t hash_backend_state_identity(const BackendStateIdentity& id) noexcept {
  std::uint64_t h = 14695981039346656037ull;
  const auto fam = static_cast<std::uint8_t>(id.family);
  const auto fmt = static_cast<std::uint8_t>(id.checkpoint.format);
  const auto ver = static_cast<std::uint8_t>(id.checkpoint.verification_level);
  const auto act = static_cast<std::uint8_t>(id.act_dtype);
  h = fnv1a_mix(h, &fam, sizeof(fam));
  fnv1a_mix_string(h, id.backend_name);
  h = fnv1a_mix(h, &fmt, sizeof(fmt));
  h = fnv1a_mix(h, &ver, sizeof(ver));
  fnv1a_mix_string(h, id.checkpoint.schema_variant);
  fnv1a_mix_string(h, id.checkpoint.model_id);
  fnv1a_mix_string(h, id.checkpoint.revision);
  h = fnv1a_mix(h, &id.checkpoint.shard_count, sizeof(id.checkpoint.shard_count));
  h = fnv1a_mix(h, id.checkpoint.shard_set_sha256.data(), id.checkpoint.shard_set_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.config_sha256.data(), id.checkpoint.config_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.tokenizer_sha256.data(), id.checkpoint.tokenizer_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.chat_template_sha256.data(), id.checkpoint.chat_template_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.weight_manifest_sha256.data(), id.checkpoint.weight_manifest_sha256.size());
  h = fnv1a_mix(h, &id.checkpoint.weight_manifest_verified, sizeof(id.checkpoint.weight_manifest_verified));
  h = fnv1a_mix(h, &id.checkpoint.shard_digests_verified, sizeof(id.checkpoint.shard_digests_verified));
  fnv1a_mix_string(h, id.state_layout_id);
  h = fnv1a_mix(h, &act, sizeof(act));
  fnv1a_mix_string(h, id.numeric_abi);
  fnv1a_mix_string(h, id.weight_precision);
  fnv1a_mix_string(h, id.kv_format);
  fnv1a_mix_string(h, id.kv_cache_mode);
  fnv1a_mix_string(h, id.gear);
  h = fnv1a_mix(h, &id.max_ctx, sizeof(id.max_ctx));
  h = fnv1a_mix(h, &id.prefill_chunk, sizeof(id.prefill_chunk));
  h = fnv1a_mix(h, &id.spec_enabled, sizeof(id.spec_enabled));
  h = fnv1a_mix(h, &id.mtp_enabled, sizeof(id.mtp_enabled));
  h = fnv1a_mix(h, &id.media_enabled, sizeof(id.media_enabled));
  h = fnv1a_mix(h, id.media_sha256.data(), id.media_sha256.size());
  h = fnv1a_mix(h, &id.persona_key, sizeof(id.persona_key));
  fnv1a_mix_string(h, id.persona_fingerprint);
  fnv1a_mix_string(h, id.cache_namespace);
  fnv1a_mix_string(h, id.rendering_fingerprint);
  fnv1a_mix_string(h, id.family_state_fingerprint);
  return h;
}

std::array<std::uint8_t, 32> hash_prompt_media(std::span<const PromptImage> images) noexcept {
  if (images.empty()) return {};
  Sha256Local h;
  const std::uint64_t count = static_cast<std::uint64_t>(images.size());
  h.update(&count, sizeof(count));
  for (const PromptImage& img : images) {
    h.update(&img.pos, sizeof(img.pos));
    h.update(&img.height, sizeof(img.height));
    h.update(&img.width, sizeof(img.width));
    const std::uint8_t has_rgb = img.rgb ? 1u : 0u;
    h.update(&has_rgb, sizeof(has_rgb));
    if (img.rgb) {
      const std::uint64_t n = static_cast<std::uint64_t>(img.rgb->size());
      h.update(&n, sizeof(n));
      if (n > 0) h.update(img.rgb->data(), img.rgb->size());
    }
  }
  return h.finalize();
}

SessionLookupResult lookup_session_prefix(const BackendSession& session, const SessionLookupQuery& query) noexcept {
  if (!query.use_prefix_cache) {
    return SessionLookupResult{false, 0, Reuse::None, "prefix_cache_bypassed"};
  }
  if (!query.identity.is_reusable_with(session.identity())) {
    return SessionLookupResult{false, 0, Reuse::None, "identity_not_reusable"};
  }
  if (!query.images.empty() && !query.identity.media_enabled) {
    return SessionLookupResult{false, 0, Reuse::None, "media_not_enabled"};
  }
  const auto req_media = hash_prompt_media(query.images);
  if (req_media != query.identity.media_sha256 || req_media != session.identity().media_sha256) {
    return SessionLookupResult{false, 0, Reuse::None, "media_digest_mismatch"};
  }

  const auto cached_ids = session.committed_tokens();
  const auto cached_mask = session.committed_token_mask();
  if (cached_mask.empty() != query.prompt_mask.empty()) {
    return SessionLookupResult{false, 0, Reuse::None, "token_mask_presence_mismatch"};
  }
  if (!cached_mask.empty() && cached_mask.size() != cached_ids.size()) {
    return SessionLookupResult{false, 0, Reuse::None, "cached_mask_length_mismatch"};
  }
  if (!query.prompt_mask.empty() && query.prompt_mask.size() != query.prompt_ids.size()) {
    return SessionLookupResult{false, 0, Reuse::None, "prompt_mask_length_mismatch"};
  }

  const std::size_t max_prefix = std::min(cached_ids.size(), query.prompt_ids.size());
  std::size_t reused = 0;
  while (reused < max_prefix) {
    if (cached_ids[reused] != query.prompt_ids[reused]) break;
    if (!cached_mask.empty() && cached_mask[reused] != query.prompt_mask[reused]) break;
    ++reused;
  }
  const std::size_t tip = cached_ids.size();
  while (reused > 0) {
    for (const PromptImage& img : query.images) {
      const std::size_t img_begin = img.pos > 0 ? static_cast<std::size_t>(img.pos) : 0;
      const std::size_t img_end = img_begin + static_cast<std::size_t>(std::max(0, img.tokens()));
      if (img_begin < reused && img_end > reused) {
        reused = img_begin;
      }
    }
    if (reused == 0) {
      return SessionLookupResult{false, 0, Reuse::None, "empty_common_prefix"};
    }
    const std::size_t restorable = session.restorable_prefix_len(reused);
    if (restorable == 0 || restorable > reused || restorable > tip) {
      return SessionLookupResult{false, 0, Reuse::None, "no_restorable_checkpoint"};
    }
    if (restorable == reused) break;
    reused = restorable;
  }
  if (reused == 0) {
    return SessionLookupResult{false, 0, Reuse::None, "empty_common_prefix"};
  }
  const Reuse kind = (reused == tip) ? Reuse::Extend : Reuse::Checkpoint;
  return SessionLookupResult{true, reused, kind,
                             kind == Reuse::Extend ? "exact_prefix_match" : "exact_checkpoint_match"};
}

SessionLookupResult restore_session_prefix(BackendSession& session, const SessionLookupQuery& query) {
  SessionLookupResult res = lookup_session_prefix(session, query);
  if (!res.hit) return res;
  if (!session.try_restore_prefix(res.reused_tokens)) {
    return SessionLookupResult{false, 0, Reuse::None, "restore_failed"};
  }
  return res;
}

bool prefix_matches_exact(std::span<const std::int32_t> cached_ids, std::span<const std::int32_t> prompt_ids,
                          std::size_t prefix_len) noexcept {
  if (prefix_len == 0 || prefix_len > cached_ids.size() || prefix_len > prompt_ids.size()) return false;
  return std::equal(cached_ids.begin(), cached_ids.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_ids.begin());
}

bool prefix_matches_exact_with_mask(std::span<const std::int32_t> cached_ids,
                                    std::span<const std::uint8_t> cached_mask,
                                    std::span<const std::int32_t> prompt_ids,
                                    std::span<const std::uint8_t> prompt_mask, std::size_t prefix_len) noexcept {
  if (prefix_len == 0 || prefix_len > cached_ids.size() || prefix_len > cached_mask.size() ||
      prefix_len > prompt_ids.size() || prefix_len > prompt_mask.size()) {
    return false;
  }
  return std::equal(cached_ids.begin(), cached_ids.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_ids.begin()) &&
         std::equal(cached_mask.begin(), cached_mask.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_mask.begin());
}

void validate_runtime_options_against_capabilities(const BackendCapabilities& caps, const RuntimeOptions& options) {
  if (caps.family == ModelFamily::Unknown) {
    throw std::invalid_argument("backend capabilities must declare a known ModelFamily");
  }
  if (options.max_ctx < 1) {
    throw std::invalid_argument("max_ctx must be positive");
  }
  if (caps.max_position_embeddings > 0 && options.max_ctx > caps.max_position_embeddings) {
    throw std::invalid_argument("max_ctx " + std::to_string(options.max_ctx) + " exceeds model limit " +
                                std::to_string(caps.max_position_embeddings));
  }
  if (options.prefill_chunk < 1) {
    throw std::invalid_argument("prefill_chunk must be positive");
  }
  if (options.max_batch < 1) {
    throw std::invalid_argument("max_batch must be at least 1");
  }
  if (options.max_batch > 1 && !caps.supports_continuous_batching) {
    throw std::invalid_argument(caps.backend_name + ": continuous batching (max_batch > 1) is not supported");
  }
  if (options.max_batch > std::max(1, caps.max_supported_batch)) {
    throw std::invalid_argument(caps.backend_name + ": max_batch " + std::to_string(options.max_batch) +
                                " exceeds max_supported_batch " + std::to_string(caps.max_supported_batch));
  }
  if (options.spec && !caps.supports_speculation) {
    throw std::invalid_argument(caps.backend_name + ": speculative decoding is not supported");
  }
  if (options.vision && !caps.supports_images) {
    throw std::invalid_argument(caps.backend_name + ": vision/image input is not supported");
  }
  if (!options.ablation.empty() && !caps.supports_ablation) {
    throw std::invalid_argument(caps.backend_name + ": ablation files are not supported");
  }
  if (options.enable_persona_hooks && !caps.supports_persona_hooks) {
    throw std::invalid_argument(caps.backend_name + ": persona hooks are not supported");
  }
  if ((!options.kv_ckpt_path.empty() || !options.kv_ckpt_resume.empty() || options.kv_ckpt_every > 0) &&
      !caps.supports_disk_checkpoint) {
    throw std::invalid_argument(caps.backend_name + ": disk KV checkpoints are not supported");
  }
}

void validate_request_against_capabilities(const BackendCapabilities& caps, int max_model_len, const Prompt& prompt,
                                           int max_new_tokens, const Sampling& sampling) {
  if (prompt.ids.empty()) {
    throw std::invalid_argument("prompt must not be empty");
  }
  if (max_new_tokens < 1) {
    throw std::invalid_argument("max_new_tokens must be positive");
  }
  if (!prompt.images.empty() && !caps.supports_images) {
    throw std::invalid_argument(caps.backend_name + ": image prompts are not supported");
  }
  if (sampling.persona != nullptr && !caps.supports_persona_hooks) {
    throw std::invalid_argument(caps.backend_name + ": per-request persona profiles are not supported");
  }
  if (max_model_len <= 0 || static_cast<std::int64_t>(prompt.ids.size()) >= static_cast<std::int64_t>(max_model_len)) {
    throw std::invalid_argument("prompt of " + std::to_string(prompt.ids.size()) +
                                " tokens does not fit max_model_len " + std::to_string(max_model_len));
  }
  if (!caps.supports_images && !caps.media_token_ids.empty()) {
    for (const std::int32_t id : prompt.ids) {
      if (std::find(caps.media_token_ids.begin(), caps.media_token_ids.end(), id) != caps.media_token_ids.end()) {
        throw std::invalid_argument(caps.backend_name + ": media placeholder token id " + std::to_string(id) +
                                    " rejected because image input is unsupported");
      }
    }
  }
  if (caps.vocab_size > 0) {
    for (std::size_t i = 0; i < prompt.ids.size(); ++i) {
      const std::int32_t id = prompt.ids[i];
      if (id < 0 && caps.supports_images) continue;
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("prompt token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
    for (std::size_t i = 0; i < sampling.forced.size(); ++i) {
      const std::int32_t id = sampling.forced[i];
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("forced token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
    for (std::size_t i = 0; i < sampling.banned_tokens.size(); ++i) {
      const std::int32_t id = sampling.banned_tokens[i];
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("banned token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
  }
}

GenerateStats ModelBackend::generate_request(const BackendRequest& req, int max_new_tokens,
                                               const Sampling& sampling, std::span<const std::int32_t> stop_ids,
                                               const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                               const PrefillCallback& on_prefill) {
  validate_request_against_capabilities(capabilities(), max_model_len(), req.prompt, max_new_tokens, sampling);
  if (!req.token_mask.empty() && req.token_mask.size() != req.prompt.ids.size()) {
    throw std::invalid_argument("BackendRequest token_mask length must equal prompt.ids length");
  }
  if (!req.token_mask.empty()) {
    throw std::logic_error(
        "ModelBackend::generate_request base fallback does not support non-empty token_mask; override generate_request()");
  }
  if (!req.use_prefix_cache) {
    throw std::logic_error(
        "ModelBackend::generate_request base fallback cannot honor use_prefix_cache=false; override generate_request()");
  }
  if (!req.cache_namespace.empty()) {
    throw std::logic_error(
        "ModelBackend::generate_request base fallback does not support cache_namespace isolation; override generate_request()");
  }
  return generate(req.prompt, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
}

BackendRegistry& BackendRegistry::instance() {
  static BackendRegistry registry;
  return registry;
}

void BackendRegistry::register_backend(BackendRegistration reg) {
  if (reg.family == ModelFamily::Unknown || reg.name.empty() || !reg.create) {
    throw std::invalid_argument("BackendRegistration requires known family, non-empty name, and factory");
  }
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  const auto key = static_cast<std::uint8_t>(reg.family);
  if (st.by_family.find(key) != st.by_family.end()) {
    throw std::invalid_argument(std::string("duplicate BackendRegistration rejected for model family ") +
                                to_string(reg.family));
  }
  st.by_family.emplace(key, std::move(reg));
}

std::optional<BackendRegistration> BackendRegistry::find(ModelFamily family) const {
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  const auto it = st.by_family.find(static_cast<std::uint8_t>(family));
  if (it == st.by_family.end()) return std::nullopt;
  return it->second;
}

std::unique_ptr<ModelBackend> BackendRegistry::create(ModelFamily family, const RuntimeOptions& options) const {
  std::function<std::unique_ptr<ModelBackend>(const RuntimeOptions&)> factory;
  {
    RegistryState& st = registry_state();
    std::lock_guard lock(st.mu);
    const auto it = st.by_family.find(static_cast<std::uint8_t>(family));
    if (it == st.by_family.end() || !it->second.create) {
      throw std::invalid_argument(std::string("no backend registered for model family ") + to_string(family));
    }
    factory = it->second.create;
  }
  return factory(options);
}

std::vector<ModelFamily> BackendRegistry::registered_families() const {
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  std::vector<ModelFamily> out;
  out.reserve(st.by_family.size());
  for (const auto& [k, _] : st.by_family) out.push_back(static_cast<ModelFamily>(k));
  std::sort(out.begin(), out.end(),
            [](ModelFamily a, ModelFamily b) { return static_cast<std::uint8_t>(a) < static_cast<std::uint8_t>(b); });
  return out;
}

BackendCapabilities inkling_default_capabilities() {
  BackendCapabilities c;
  c.family = ModelFamily::Inkling2;
  c.backend_name = "inkling2-cuda";
  c.architecture = "Inkling2ForConditionalGeneration";
  c.supports_text = true;
  c.supports_images = true;
  c.supports_speculation = true;
  c.supports_continuous_batching = true;
  c.max_supported_batch = 64;
  c.supports_prefix_cache = true;
  c.supports_host_park = true;
  c.supports_disk_checkpoint = true;
  c.supports_persona_hooks = true;
  c.supports_ablation = true;
  c.supports_logit_capture = true;
  c.supported_gears = {"loop_1", "loop_2", "loop_3", "loop_4", "loop_5", "loop_6", "loop_7", "loop_8"};
  c.supported_kv_cache_modes = {"per_pass", "last_full"};
  c.supported_quant_layouts = {"bf16", "dense_fp8", "nvfp4", "mxfp8", "atlas2", "atlas3", "exl3"};
  c.vocab_size = 201024;
  c.max_position_embeddings = 1048576;
  return c;
}

}  // namespace ingine::model
