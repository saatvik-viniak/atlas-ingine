#include "ingine/platos_key.h"

#include "core/act.h"
#include "core/sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ingine {
namespace {

struct RawTensorDesc {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t data_begin = 0;  // absolute byte offset in raw_file
  std::uint64_t data_end = 0;

  std::int64_t numel() const noexcept {
    std::int64_t n = 1;
    for (std::int64_t d : shape) n *= d;
    return n;
  }
};

struct ParsedSafetensorsBuffer {
  std::vector<std::uint8_t> bytes;
  std::string sha256;
  std::map<std::string, std::string> metadata;
  std::unordered_map<std::string, RawTensorDesc> tensors;

  const RawTensorDesc* find(std::string_view name) const noexcept {
    const auto it = tensors.find(std::string(name));
    return it == tensors.end() ? nullptr : &it->second;
  }
};

std::string to_lower_ascii(std::string_view s) {
  std::string out(s);
  for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

std::string normalize_sha256_hex_strict(std::string_view hex, const char* field_name) {
  if (hex.size() != 64) {
    throw std::invalid_argument(std::string("platos_key: '") + field_name + "' must be a 64-char hex SHA-256");
  }
  std::string lower(64, '0');
  for (std::size_t i = 0; i < 64; ++i) {
    const unsigned char c = static_cast<unsigned char>(hex[i]);
    if (c >= '0' && c <= '9') {
      lower[i] = static_cast<char>(c);
    } else if (c >= 'a' && c <= 'f') {
      lower[i] = static_cast<char>(c);
    } else if (c >= 'A' && c <= 'F') {
      lower[i] = static_cast<char>(c - 'A' + 'a');
    } else {
      throw std::invalid_argument(std::string("platos_key: invalid hex char in '") + field_name + "'");
    }
  }
  return lower;
}

// Single-read zero-TOCTOU load of a safetensors file + exact SHA-256 over the consumed bytes.
ParsedSafetensorsBuffer read_and_parse_safetensors_once(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::invalid_argument("platos_key: cannot open '" + path.string() + "'");
  const std::streamoff sz = in.tellg();
  if (sz < 8) throw std::invalid_argument("platos_key: file truncated (<8 bytes): " + path.string());
  in.seekg(0, std::ios::beg);

  ParsedSafetensorsBuffer out;
  out.bytes.resize(static_cast<std::size_t>(sz));
  in.read(reinterpret_cast<char*>(out.bytes.data()), sz);
  if (!in) throw std::invalid_argument("platos_key: read error on '" + path.string() + "'");

  out.sha256 = sha256_hex(sha256_bytes(out.bytes.data(), out.bytes.size()));

  std::uint64_t hdr_len = 0;
  for (int i = 0; i < 8; ++i) hdr_len |= static_cast<std::uint64_t>(out.bytes[static_cast<std::size_t>(i)]) << (8 * i);
  if (hdr_len == 0 || hdr_len > out.bytes.size() - 8) {
    throw std::invalid_argument("platos_key: invalid safetensors header length in '" + path.string() + "'");
  }
  const std::uint64_t data_base = 8 + hdr_len;
  const std::uint64_t data_len = static_cast<std::uint64_t>(out.bytes.size()) - data_base;

  nlohmann::json hdr;
  try {
    hdr = nlohmann::json::parse(out.bytes.data() + 8, out.bytes.data() + data_base);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument("platos_key: malformed safetensors JSON header in '" + path.string() + "': " + e.what());
  }
  if (!hdr.is_object()) {
    throw std::invalid_argument("platos_key: safetensors header root must be an object in '" + path.string() + "'");
  }
  for (const auto& [k, v] : hdr.items()) {
    if (k == "__metadata__") {
      if (v.is_object()) {
        for (const auto& [mk, mv] : v.items()) {
          if (mv.is_string()) out.metadata[mk] = mv.get<std::string>();
        }
      }
      continue;
    }
    if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets")) {
      throw std::invalid_argument("platos_key: malformed tensor descriptor '" + k + "'");
    }
    RawTensorDesc td;
    td.name = k;
    td.dtype = v["dtype"].get<std::string>();
    const auto& sh = v["shape"];
    if (!sh.is_array()) throw std::invalid_argument("platos_key: shape must be array for '" + k + "'");
    for (const auto& d : sh) {
      const std::int64_t dv = d.get<std::int64_t>();
      if (dv < 0) throw std::invalid_argument("platos_key: negative dim in shape for '" + k + "'");
      td.shape.push_back(dv);
    }
    const auto& offs = v["data_offsets"];
    if (!offs.is_array() || offs.size() != 2) {
      throw std::invalid_argument("platos_key: invalid data_offsets for '" + k + "'");
    }
    const std::uint64_t b = offs[0].get<std::uint64_t>();
    const std::uint64_t e = offs[1].get<std::uint64_t>();
    if (e < b || e > data_len) {
      throw std::invalid_argument("platos_key: out-of-bounds data_offsets for '" + k + "'");
    }
    td.data_begin = data_base + b;
    td.data_end = data_base + e;
    out.tensors.emplace(k, std::move(td));
  }
  return out;
}

std::vector<float> decode_tensor_f32(const ParsedSafetensorsBuffer& buf, const RawTensorDesc& t) {
  const std::uint8_t* raw = buf.bytes.data() + t.data_begin;
  const std::size_t raw_size = static_cast<std::size_t>(t.data_end - t.data_begin);
  const std::size_t n = static_cast<std::size_t>(t.numel());
  std::vector<float> out(n);
  if (t.dtype == "F32") {
    if (raw_size != n * 4) throw std::invalid_argument("platos_key: F32 byte size mismatch for " + t.name);
    std::memcpy(out.data(), raw, raw_size);
  } else if (t.dtype == "F64") {
    if (raw_size != n * 8) throw std::invalid_argument("platos_key: F64 byte size mismatch for " + t.name);
    for (std::size_t i = 0; i < n; ++i) {
      double dv = 0.0;
      std::memcpy(&dv, raw + i * 8, 8);
      out[i] = static_cast<float>(dv);
    }
  } else if (t.dtype == "F16") {
    if (raw_size != n * 2) throw std::invalid_argument("platos_key: F16 byte size mismatch for " + t.name);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, raw + i * 2, 2);
      out[i] = act::host_fp16_bits_to_float(bits);
    }
  } else if (t.dtype == "BF16") {
    if (raw_size != n * 2) throw std::invalid_argument("platos_key: BF16 byte size mismatch for " + t.name);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, raw + i * 2, 2);
      out[i] = act::host_bf16_bits_to_float(bits);
    }
  } else {
    throw std::invalid_argument("platos_key: unsupported tensor dtype '" + t.dtype + "' for '" + t.name +
                                "' (expected F16, BF16, F32, or F64)");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(out[i])) {
      throw std::invalid_argument("platos_key: non-finite value in tensor '" + t.name + "'");
    }
  }
  return out;
}

std::string read_text_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::invalid_argument("platos_key: cannot open metadata JSON '" + p.string() + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string meta_get_first(const std::map<std::string, std::string>& m, std::initializer_list<const char*> keys) {
  for (const char* k : keys) {
    auto it = m.find(k);
    if (it != m.end() && !it->second.empty()) return it->second;
  }
  return {};
}

std::optional<int> parse_strict_nonneg_int(std::string_view s) noexcept {
  if (s.empty()) return std::nullopt;
  int v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return std::nullopt;
    if (v > 1000000) return std::nullopt;
    v = v * 10 + (c - '0');
  }
  return v;
}

struct ParsedTapBinding {
  bool has_spec = false;
  std::optional<int> tap_layer;
  bool requires_final_layer = false;
  bool is_multi_tap = false;
  bool is_generic_single_tap = false;
  std::string raw_spec;
};

// Parses and validates a single metadata source's (tap_spec, tap_layer) pair independently before any merge.
// Preserves empty/omitted legacy fields; rejects malformed, conflicting, or unrecognized nonempty tap_spec fail-closed.
ParsedTapBinding parse_single_tap_source(std::string_view tap_spec, std::optional<int> explicit_tap_layer,
                                         const char* source_label) {
  ParsedTapBinding out;
  out.raw_spec = std::string(tap_spec);
  const std::string spec_l = to_lower_ascii(tap_spec);
  std::optional<int> spec_tap;
  if (spec_l.empty()) {
    // Omitted tap_spec in this source.
  } else {
    out.has_spec = true;
    if (spec_l == "final_normed" || spec_l == "final") {
      out.requires_final_layer = true;
    } else if (spec_l == "single_tap") {
      out.is_generic_single_tap = true;
    } else if (spec_l.find("concat") != std::string::npos) {
      out.is_multi_tap = true;
    } else if (spec_l.rfind("layer_", 0) == 0) {
      spec_tap = parse_strict_nonneg_int(std::string_view(spec_l).substr(6));
      if (!spec_tap.has_value()) {
        throw std::invalid_argument(std::string("platos_key: malformed ") + source_label + " tap_spec '" +
                                    std::string(tap_spec) + "'");
      }
    } else if (spec_l.rfind("visit_", 0) == 0) {
      spec_tap = parse_strict_nonneg_int(std::string_view(spec_l).substr(6));
      if (!spec_tap.has_value()) {
        throw std::invalid_argument(std::string("platos_key: malformed ") + source_label + " tap_spec '" +
                                    std::string(tap_spec) + "'");
      }
    } else {
      throw std::invalid_argument(std::string("platos_key: unsupported ") + source_label + " tap_spec '" +
                                  std::string(tap_spec) +
                                  "' (expected 'layer_<N>', 'visit_<N>', 'final_normed', 'final', or 'single_tap')");
    }
  }

  if (explicit_tap_layer.has_value()) {
    if (*explicit_tap_layer < 0) {
      throw std::invalid_argument(std::string("platos_key: negative ") + source_label + " tap_layer " +
                                  std::to_string(*explicit_tap_layer));
    }
    if (out.requires_final_layer) {
      throw std::invalid_argument(std::string("platos_key: ") + source_label + " tap_layer=" +
                                  std::to_string(*explicit_tap_layer) + " conflicts with tap_spec='" +
                                  std::string(tap_spec) + "'");
    }
    if (out.is_multi_tap) {
      throw std::invalid_argument(std::string("platos_key: ") + source_label + " tap_layer=" +
                                  std::to_string(*explicit_tap_layer) + " conflicts with multi-tap tap_spec='" +
                                  std::string(tap_spec) + "'");
    }
    if (spec_tap.has_value() && *spec_tap != *explicit_tap_layer) {
      throw std::invalid_argument(std::string("platos_key: ") + source_label + " tap_layer=" +
                                  std::to_string(*explicit_tap_layer) + " conflicts with tap_spec='" +
                                  std::string(tap_spec) + "'");
    }
    out.tap_layer = *explicit_tap_layer;
  } else if (spec_tap.has_value()) {
    out.tap_layer = *spec_tap;
  }
  return out;
}

// Reconciles two independently validated tap bindings (within a single source or across sources);
// rejects every contradiction fail-closed while preserving compatible aliases and omitted legacy fields.
ParsedTapBinding merge_tap_bindings(const ParsedTapBinding& a, const ParsedTapBinding& b, const char* label_a,
                                    const char* label_b) {
  const bool a_decl = a.has_spec || a.tap_layer.has_value();
  const bool b_decl = b.has_spec || b.tap_layer.has_value();
  if (!a_decl) return b;
  if (!b_decl) return a;

  if (a.is_multi_tap != b.is_multi_tap) {
    throw std::invalid_argument(std::string("platos_key: ") + label_b + " tap_spec '" + b.raw_spec +
                                "' contradicts " + label_a + " tap_spec '" + a.raw_spec +
                                "' (multi-tap vs single-tap)");
  }
  if ((a.requires_final_layer && b.tap_layer.has_value()) ||
      (b.requires_final_layer && a.tap_layer.has_value())) {
    throw std::invalid_argument(std::string("platos_key: ") + label_b + " tap declaration ('" + b.raw_spec +
                                "') contradicts " + label_a + " tap declaration ('" + a.raw_spec +
                                "') (final-layer vs indexed tap)");
  }
  if (a.tap_layer.has_value() && b.tap_layer.has_value() && *a.tap_layer != *b.tap_layer) {
    throw std::invalid_argument(std::string("platos_key: ") + label_b + " tap_layer=" + std::to_string(*b.tap_layer) +
                                " (tap_spec='" + b.raw_spec + "') contradicts " + label_a + " tap_layer=" +
                                std::to_string(*a.tap_layer) + " (tap_spec='" + a.raw_spec + "')");
  }

  ParsedTapBinding merged;
  merged.has_spec = a.has_spec || b.has_spec;
  merged.is_multi_tap = a.is_multi_tap || b.is_multi_tap;
  merged.requires_final_layer = a.requires_final_layer || b.requires_final_layer;
  merged.tap_layer = b.tap_layer.has_value() ? b.tap_layer : a.tap_layer;
  merged.is_generic_single_tap = a.is_generic_single_tap && b.is_generic_single_tap;
  if (!b.raw_spec.empty() && !(b.is_generic_single_tap && !a.raw_spec.empty())) {
    merged.raw_spec = b.raw_spec;
  } else {
    merged.raw_spec = a.raw_spec;
  }
  return merged;
}

// Reconciles all populated tap_spec/target_tap entries + optional explicit tap_layer within one source.
ParsedTapBinding parse_and_reconcile_source_taps(const std::vector<std::pair<std::string, std::string>>& spec_entries,
                                                 std::optional<int> explicit_tap_layer, const char* source_label) {
  if (spec_entries.empty()) {
    return parse_single_tap_source("", explicit_tap_layer, source_label);
  }
  ParsedTapBinding acc;
  std::string acc_label;
  for (const auto& [key_name, spec_val] : spec_entries) {
    const std::string entry_label = std::string(source_label) + " '" + key_name + "'";
    const ParsedTapBinding cur = parse_single_tap_source(spec_val, explicit_tap_layer, entry_label.c_str());
    if (acc_label.empty()) {
      acc = cur;
      acc_label = entry_label;
    } else {
      acc = merge_tap_bindings(acc, cur, acc_label.c_str(), entry_label.c_str());
    }
  }
  return acc;
}

void reconcile_tap_bindings(const ParsedTapBinding& emb, const ParsedTapBinding& sidecar, PlatosKeyMetadata& meta) {
  const ParsedTapBinding final_tap = merge_tap_bindings(emb, sidecar, "embedded", "sidecar");
  meta.is_multi_tap = final_tap.is_multi_tap;
  meta.requires_final_layer = final_tap.requires_final_layer;
  meta.expected_tap_layer = final_tap.tap_layer;
  meta.tap_spec = final_tap.raw_spec;
}

// Canonicalizes an explicit checkpoint identifier via documented packaging/runtime aliases only.
// Strips optional org prefix ("org/repo" -> "repo"), lowercases, strips only exact known quant/packaging/harness
// suffixes ("-nvfp4", "-fp8", "-bf16", "-fp16", "-int8", "-int4", "-awq", "-gptq", "-gguf", "-ingine", "-fake"),
// and collapses punctuation ("-", "_", "."). Distinct checkpoints within the same family
// (e.g. "example/Inkling-2-Different-Checkpoint", "Inkling-2-Mini-abliterated", "Qwen/Qwen3.8-8B") remain distinct.
std::string canonical_checkpoint_id(std::string_view name) {
  const std::size_t slash = name.rfind('/');
  std::string tail = to_lower_ascii((slash == std::string_view::npos) ? name : name.substr(slash + 1));
  static constexpr std::string_view kPackagingSuffixes[] = {
      "-nvfp4", "_nvfp4", "-fp8",    "_fp8",    "-bf16", "_bf16", "-fp16",   "_fp16",
      "-int8",  "_int8",  "-int4",   "_int4",   "-awq",  "_awq",  "-gptq",   "_gptq",
      "-gguf",  "_gguf",  "-ingine", "_ingine", "-fake", "_fake",
  };
  bool stripped = true;
  while (stripped) {
    stripped = false;
    for (std::string_view suf : kPackagingSuffixes) {
      if (tail.size() > suf.size() && std::string_view(tail).substr(tail.size() - suf.size()) == suf) {
        tail.resize(tail.size() - suf.size());
        stripped = true;
        break;
      }
    }
  }
  std::string out;
  out.reserve(tail.size());
  for (char c : tail) {
    if (std::isalnum(static_cast<unsigned char>(c))) out.push_back(c);
  }
  return out;
}

std::string reconcile_donor_aliases(const std::vector<std::pair<std::string, std::string>>& entries,
                                    const char* source_label) {
  std::string chosen;
  std::string chosen_ckpt;
  std::string chosen_key;
  for (const auto& [k, val] : entries) {
    if (val.empty()) continue;
    const std::string ckpt = canonical_checkpoint_id(val);
    if (ckpt.empty()) {
      throw std::invalid_argument(std::string("platos_key: malformed ") + source_label + " '" + k + "'='" + val + "'");
    }
    if (chosen.empty()) {
      chosen = val;
      chosen_ckpt = ckpt;
      chosen_key = k;
    } else if (ckpt != chosen_ckpt) {
      throw std::invalid_argument(std::string("platos_key: ") + source_label + " '" + k + "'='" + val +
                                  "' contradicts '" + chosen_key + "'='" + chosen + "'");
    }
  }
  return chosen;
}

struct SidecarDeclaredFields {
  std::vector<std::pair<std::string, std::string>> tap_specs;
  std::optional<int> tap_layer;
  std::vector<std::pair<std::string, std::string>> donor_models;
  std::string cartridge_id;
};

SidecarDeclaredFields merge_and_validate_json_metadata(std::string_view json_text, PlatosKeyMetadata& meta) {
  SidecarDeclaredFields sc_fields;
  if (json_text.empty()) return sc_fields;

  nlohmann::ordered_json obj;
  try {
    obj = nlohmann::ordered_json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("platos_key: invalid metadata JSON: ") + e.what());
  }
  if (!obj.is_object()) {
    throw std::invalid_argument("platos_key: metadata JSON root must be an object");
  }
  meta.json_metadata = std::string(json_text);

  auto parse_json_int = [&](const nlohmann::ordered_json& val, const char* key) -> int {
    if (val.is_number_integer() && !val.is_boolean()) return val.get<int>();
    if (val.is_string()) {
      const auto parsed = parse_strict_nonneg_int(val.get_ref<const std::string&>());
      if (parsed.has_value()) return *parsed;
      throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "' is not a valid non-negative integer");
    }
    throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "' must be an integer");
  };

  auto check_dim = [&](const char* key, int actual) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return;
    const int declared = parse_json_int(*it, key);
    if (declared != actual) {
      throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "'=" + std::to_string(declared) +
                                  " contradicts safetensors shape (" + std::to_string(actual) + ")");
    }
  };
  check_dim("d_donor", meta.d_in);
  check_dim("d_in", meta.d_in);
  check_dim("d_target", meta.d_out);
  check_dim("d_out", meta.d_out);
  if (meta.svd_rank > 0) check_dim("svd_rank", meta.svd_rank);

  // Enforce pinned artifact SHA-256 against the exact loaded safetensors bytes (fail-closed!).
  for (const char* sha_key : {"cartridge_sha256", "sha256"}) {
    if (auto it = obj.find(sha_key); it != obj.end() && !it->is_null()) {
      if (!it->is_string()) {
        throw std::invalid_argument(std::string("platos_key: metadata JSON '") + sha_key + "' must be a string");
      }
      const std::string expected_sha = normalize_sha256_hex_strict(it->get_ref<const std::string&>(), sha_key);
      if (expected_sha != meta.file_sha256) {
        throw std::invalid_argument(std::string("platos_key: artifact SHA-256 mismatch for '") + sha_key +
                                    "': observed=" + meta.file_sha256 + ", expected=" + expected_sha);
      }
    }
  }

  if (auto it = obj.find("tap_layer"); it != obj.end() && !it->is_null()) {
    sc_fields.tap_layer = parse_json_int(*it, "tap_layer");
  }
  for (const char* tkey : {"tap_spec", "target_tap"}) {
    if (auto it = obj.find(tkey); it != obj.end() && !it->is_null()) {
      if (!it->is_string()) {
        throw std::invalid_argument(std::string("platos_key: metadata JSON '") + tkey + "' must be a string");
      }
      const std::string val = it->get<std::string>();
      if (!val.empty()) sc_fields.tap_specs.emplace_back(tkey, val);
    }
  }
  for (const char* dkey : {"donor_model_id", "donor_model", "source_model", "model_id"}) {
    if (auto it = obj.find(dkey); it != obj.end() && !it->is_null()) {
      if (!it->is_string()) {
        throw std::invalid_argument(std::string("platos_key: metadata JSON '") + dkey + "' must be a string");
      }
      const std::string val = it->get<std::string>();
      if (!val.empty()) sc_fields.donor_models.emplace_back(dkey, val);
    }
  }
  if (auto it = obj.find("cartridge_id"); it != obj.end() && !it->is_null()) {
    if (!it->is_string()) {
      throw std::invalid_argument("platos_key: metadata JSON 'cartridge_id' must be a string");
    }
    sc_fields.cartridge_id = it->get<std::string>();
  }
  if (auto it = obj.find("forbidden_coord_d2027_zero_masked"); it != obj.end() && !it->is_null()) {
    if (!it->is_boolean()) {
      throw std::invalid_argument("platos_key: metadata JSON 'forbidden_coord_d2027_zero_masked' must be a boolean");
    }
    meta.forbidden_coord_d2027_zero_masked = it->get<bool>();
  }

  for (const auto& [k, v] : obj.items()) {
    if (v.is_string()) {
      meta.raw_metadata[k] = v.get<std::string>();
    } else if (v.is_number() || v.is_boolean()) {
      meta.raw_metadata[k] = v.dump();
    }
  }
  return sc_fields;
}

}  // namespace

void l2_normalize_rows_cpu(float* data, int batch, int dim) noexcept {
  if (!data || batch <= 0 || dim <= 0) return;
  for (int b = 0; b < batch; ++b) {
    float* row = data + static_cast<std::size_t>(b) * static_cast<std::size_t>(dim);
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = static_cast<double>(row[d]);
      sum_sq += v * v;
    }
    const double norm = std::sqrt(sum_sq);
    if (norm > 1e-12) {
      const double inv = 1.0 / norm;
      for (int d = 0; d < dim; ++d) row[d] = static_cast<float>(static_cast<double>(row[d]) * inv);
    }
  }
}

std::shared_ptr<const PlatosKey> PlatosKey::load(const std::filesystem::path& safetensors_path, int expected_in_dim) {
  return load_with_json(safetensors_path, {}, expected_in_dim);
}

std::shared_ptr<const PlatosKey> PlatosKey::load(const std::filesystem::path& safetensors_path,
                                                 const std::filesystem::path& metadata_json_path,
                                                 int expected_in_dim) {
  const std::string json_text = metadata_json_path.empty() ? std::string{} : read_text_file(metadata_json_path);
  return load_with_json(safetensors_path, json_text, expected_in_dim);
}

std::shared_ptr<const PlatosKey> PlatosKey::load_with_json(const std::filesystem::path& safetensors_path,
                                                           std::string_view metadata_json, int expected_in_dim) {
  if (safetensors_path.empty()) {
    throw std::invalid_argument("platos_key: empty safetensors path");
  }
  if (!std::filesystem::is_regular_file(safetensors_path)) {
    throw std::invalid_argument("platos_key: file does not exist: " + safetensors_path.string());
  }

  // Single-read load + SHA-256 verification directly on consumed bytes (zero TOCTOU reopen).
  const ParsedSafetensorsBuffer buf = read_and_parse_safetensors_once(safetensors_path);

  // Explicit fail-closed rejection of negative-space anti-cartridge layouts (#965).
  if (buf.find("U_bad") != nullptr) {
    throw std::invalid_argument("platos_key: '" + safetensors_path.string() +
                                "' is a negative-space anti_cart layout ('U_bad'), unsupported by /v1/embeddings");
  }

  // Supported positive-projection layouts (1:1 contract with #964 AtlasCartridge v1):
  //   1) ridge_R:       R [d_in, d_out] (+ optional donor_mean, target_mean, scale)
  //   2) svd_Ur_Vr:     U_r [d_in, r], V_r [r, d_out] (+ optional donor_mean, target_mean, scale)
  //   3) affine_W_yue2: W_yue2 [d_in, d_out] (+ optional b_yue2)
  //   4) affine_W:      W [d_in, d_out] (+ optional b)
  const RawTensorDesc* r_t = buf.find("R");
  const RawTensorDesc* u_t = buf.find("U_r");
  const RawTensorDesc* v_t = buf.find("V_r");
  const RawTensorDesc* wy_t = buf.find("W_yue2");
  const RawTensorDesc* w_t = buf.find("W");

  std::string detected_format;
  std::string tensor_layout;
  std::string primary_dtype;
  int d_in = 0, d_out = 0, svd_rank = 0;
  std::vector<float> r_f32;
  std::vector<std::uint16_t> rt_f16;

  auto populate_rt_f16_from_f32 = [&](const std::vector<float>& mat_f32, int rows, int cols) {
    rt_f16.resize(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) {
      const float* src_row = mat_f32.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(cols);
      for (int j = 0; j < cols; ++j) {
        rt_f16[static_cast<std::size_t>(j) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(i)] =
            act::host_float_to_fp16_bits(src_row[j]);
      }
    }
  };

  auto load_2d_direct = [&](const RawTensorDesc& t, std::string_view fmt, std::string_view layout) {
    if (t.shape.size() != 2 || t.shape[0] <= 0 || t.shape[1] <= 0) {
      throw std::invalid_argument("platos_key: projection tensor '" + t.name + "' must be 2D with positive dims");
    }
    d_in = static_cast<int>(t.shape[0]);
    d_out = static_cast<int>(t.shape[1]);
    detected_format = std::string(fmt);
    tensor_layout = std::string(layout);
    primary_dtype = t.dtype;
    r_f32 = decode_tensor_f32(buf, t);
    if (t.dtype == "F16") {
      // Preserve exact IEEE-754 binary16 bit patterns (including subnormals) without float roundtrip.
      const std::uint8_t* raw_r = buf.bytes.data() + t.data_begin;
      rt_f16.resize(static_cast<std::size_t>(d_out) * static_cast<std::size_t>(d_in));
      for (int i = 0; i < d_in; ++i) {
        const std::size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
        for (int j = 0; j < d_out; ++j) {
          std::uint16_t bits = 0;
          std::memcpy(&bits, raw_r + (row_off + static_cast<std::size_t>(j)) * 2, 2);
          rt_f16[static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in) + static_cast<std::size_t>(i)] = bits;
        }
      }
    } else {
      populate_rt_f16_from_f32(r_f32, d_in, d_out);
    }
  };

  if (r_t) {
    load_2d_direct(*r_t, "platos_key_v1", "ridge_R");
  } else if (u_t || v_t) {
    if (!u_t || !v_t) {
      throw std::invalid_argument("platos_key: factorized SVD layout requires BOTH 'U_r' and 'V_r' tensors");
    }
    if (u_t->shape.size() != 2 || v_t->shape.size() != 2 || u_t->shape[0] <= 0 || u_t->shape[1] <= 0 ||
        v_t->shape[1] <= 0 || u_t->shape[1] != v_t->shape[0]) {
      throw std::invalid_argument("platos_key: invalid SVD factor shapes for 'U_r' and 'V_r'");
    }
    d_in = static_cast<int>(u_t->shape[0]);
    svd_rank = static_cast<int>(u_t->shape[1]);
    d_out = static_cast<int>(v_t->shape[1]);
    detected_format = "platos_key_svd_v1";
    tensor_layout = "svd_Ur_Vr";
    primary_dtype = u_t->dtype;
    const std::vector<float> u_f32 = decode_tensor_f32(buf, *u_t);
    const std::vector<float> v_f32 = decode_tensor_f32(buf, *v_t);
    // Reconstruct dense R = U_r @ V_r in FP64 so CPU & GPU projection paths share identical exact semantics.
    r_f32.assign(static_cast<std::size_t>(d_in) * static_cast<std::size_t>(d_out), 0.0f);
    for (int i = 0; i < d_in; ++i) {
      const float* ui = u_f32.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(svd_rank);
      float* ri = r_f32.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
      for (int j = 0; j < d_out; ++j) {
        double acc = 0.0;
        for (int r = 0; r < svd_rank; ++r) {
          acc += static_cast<double>(ui[r]) *
                 static_cast<double>(v_f32[static_cast<std::size_t>(r) * static_cast<std::size_t>(d_out) + j]);
        }
        ri[j] = static_cast<float>(acc);
      }
    }
    populate_rt_f16_from_f32(r_f32, d_in, d_out);
  } else if (wy_t) {
    load_2d_direct(*wy_t, "omni946-inkling-yue2-key/v1", "affine_W_yue2");
  } else if (w_t) {
    load_2d_direct(*w_t, "affine_wb", "affine_W");
  } else {
    throw std::invalid_argument(
        "platos_key: '" + safetensors_path.string() +
        "' has unsupported tensor layout (expected 'R', 'U_r'+'V_r', 'W_yue2', or 'W')");
  }

  if (expected_in_dim > 0 && d_in != expected_in_dim) {
    throw std::invalid_argument("platos_key: donor dim mismatch in '" + safetensors_path.string() + "' (key d_in=" +
                                std::to_string(d_in) + ", model hidden_dim=" + std::to_string(expected_in_dim) + ")");
  }

  std::shared_ptr<PlatosKey> key(new PlatosKey());
  key->source_path_ = safetensors_path;
  key->meta_.d_in = d_in;
  key->meta_.d_out = d_out;
  key->meta_.svd_rank = svd_rank;
  key->meta_.tensor_layout = tensor_layout;
  key->meta_.weight_dtype = primary_dtype;
  key->meta_.file_sha256 = buf.sha256;
  key->meta_.raw_metadata = buf.metadata;
  key->r_f32_ = std::move(r_f32);
  key->rt_f16_ = std::move(rt_f16);

  key->donor_mean_.assign(static_cast<std::size_t>(d_in), 0.0f);
  if (const RawTensorDesc* dm = buf.find("donor_mean")) {
    if (dm->shape.size() != 1 || dm->shape[0] != d_in) {
      throw std::invalid_argument("platos_key: 'donor_mean' shape must be [" + std::to_string(d_in) + "]");
    }
    key->donor_mean_ = decode_tensor_f32(buf, *dm);
    key->has_donor_mean_ = true;
  }

  key->target_mean_.assign(static_cast<std::size_t>(d_out), 0.0f);
  const RawTensorDesc* tm = buf.find("target_mean");
  if (!tm && tensor_layout == "affine_W_yue2") tm = buf.find("b_yue2");
  if (!tm && (tensor_layout == "affine_W" || tensor_layout == "affine_W_yue2")) tm = buf.find("b");
  if (tm) {
    if (tm->shape.size() != 1 || tm->shape[0] != d_out) {
      throw std::invalid_argument("platos_key: '" + tm->name + "' shape must be [" + std::to_string(d_out) + "]");
    }
    key->target_mean_ = decode_tensor_f32(buf, *tm);
    key->has_target_mean_ = true;
  }

  key->meta_.scale = 1.0f;
  if (const RawTensorDesc* sc = buf.find("scale")) {
    if (sc->numel() != 1) {
      throw std::invalid_argument("platos_key: 'scale' tensor must have exactly 1 element");
    }
    key->meta_.scale = decode_tensor_f32(buf, *sc)[0];
  }

  // Extract and reconcile ALL populated embedded safetensors tap & donor declarations BEFORE sidecar merge.
  std::vector<std::pair<std::string, std::string>> emb_tap_specs;
  for (const char* tkey : {"tap_spec", "target_tap"}) {
    if (auto it = buf.metadata.find(tkey); it != buf.metadata.end() && !it->second.empty()) {
      emb_tap_specs.emplace_back(tkey, it->second);
    }
  }
  std::optional<int> emb_tap_layer;
  if (const std::string raw_emb_tl = meta_get_first(buf.metadata, {"tap_layer"});
      !raw_emb_tl.empty() && to_lower_ascii(raw_emb_tl) != "none" && to_lower_ascii(raw_emb_tl) != "null") {
    emb_tap_layer = parse_strict_nonneg_int(raw_emb_tl);
    if (!emb_tap_layer.has_value()) {
      throw std::invalid_argument("platos_key: malformed embedded metadata tap_layer '" + raw_emb_tl + "'");
    }
  }
  const ParsedTapBinding emb_tap = parse_and_reconcile_source_taps(emb_tap_specs, emb_tap_layer, "embedded");

  std::vector<std::pair<std::string, std::string>> emb_donors;
  for (const char* dkey : {"donor_model_id", "donor_model", "source_model", "model_id"}) {
    if (auto it = buf.metadata.find(dkey); it != buf.metadata.end() && !it->second.empty()) {
      emb_donors.emplace_back(dkey, it->second);
    }
  }
  const std::string emb_donor = reconcile_donor_aliases(emb_donors, "embedded");
  const std::string emb_cartridge_id = meta_get_first(buf.metadata, {"cartridge_id"});

  // Validate sidecar JSON fields (including SHA-256) and reconcile ALL populated sidecar tap & donor declarations.
  const SidecarDeclaredFields sc_fields = merge_and_validate_json_metadata(metadata_json, key->meta_);
  const ParsedTapBinding sc_tap =
      parse_and_reconcile_source_taps(sc_fields.tap_specs, sc_fields.tap_layer, "sidecar");
  const std::string sc_donor = reconcile_donor_aliases(sc_fields.donor_models, "sidecar");

  key->meta_.format = meta_get_first(key->meta_.raw_metadata, {"format"});
  if (key->meta_.format.empty()) key->meta_.format = detected_format;
  key->meta_.schema_version = meta_get_first(key->meta_.raw_metadata, {"schema_version"});
  if (!emb_cartridge_id.empty() && !sc_fields.cartridge_id.empty() && emb_cartridge_id != sc_fields.cartridge_id) {
    throw std::invalid_argument("platos_key: sidecar cartridge_id '" + sc_fields.cartridge_id +
                                "' contradicts embedded cartridge_id '" + emb_cartridge_id + "'");
  }
  key->meta_.cartridge_id = !sc_fields.cartridge_id.empty() ? sc_fields.cartridge_id : emb_cartridge_id;

  // Cross-source donor reconciliation: if both embedded and sidecar declare a donor checkpoint, require agreement.
  if (!emb_donor.empty() && !sc_donor.empty()) {
    const std::string emb_ckpt = canonical_checkpoint_id(emb_donor);
    const std::string sc_ckpt = canonical_checkpoint_id(sc_donor);
    if (emb_ckpt != sc_ckpt) {
      throw std::invalid_argument("platos_key: sidecar donor_model_id '" + sc_donor +
                                  "' contradicts embedded donor_model '" + emb_donor + "'");
    }
  }
  key->meta_.donor_model = !sc_donor.empty() ? sc_donor : emb_donor;
  key->meta_.target_model = meta_get_first(key->meta_.raw_metadata, {"target_model"});
  key->meta_.method = meta_get_first(key->meta_.raw_metadata, {"method", "solver"});

  // Also honor forbidden_coord_d2027_zero_masked if declared in embedded safetensors metadata or deployment_spec.
  if (!key->meta_.forbidden_coord_d2027_zero_masked) {
    const std::string raw_mask = to_lower_ascii(
        meta_get_first(key->meta_.raw_metadata, {"forbidden_coord_d2027_zero_masked"}));
    if (raw_mask == "true" || raw_mask == "1") {
      key->meta_.forbidden_coord_d2027_zero_masked = true;
    } else if (const std::string ds = meta_get_first(key->meta_.raw_metadata, {"deployment_spec"}); !ds.empty()) {
      try {
        const auto ds_json = nlohmann::json::parse(ds);
        if (ds_json.is_object() && ds_json.contains("forbidden_coord_d2027_zero_masked") &&
            ds_json["forbidden_coord_d2027_zero_masked"].is_boolean() &&
            ds_json["forbidden_coord_d2027_zero_masked"].get<bool>()) {
          key->meta_.forbidden_coord_d2027_zero_masked = true;
        }
      } catch (...) {
      }
    }
  }

  reconcile_tap_bindings(emb_tap, sc_tap, key->meta_);
  return key;
}

void PlatosKey::validate_binding(std::string_view active_model_id, std::string_view active_model_family,
                                 int active_embedding_dim, int active_num_tap_layers) const {
  (void)active_model_family;
  if (meta_.is_multi_tap) {
    throw std::invalid_argument("platos_key: multi-tap concat spec '" + meta_.tap_spec +
                                "' is not supported by single-tap /v1/embeddings");
  }
  if (active_embedding_dim > 0 && meta_.d_in != active_embedding_dim) {
    throw std::invalid_argument("platos_key: donor dim mismatch (key d_in=" + std::to_string(meta_.d_in) +
                                ", active model embedding_dim=" + std::to_string(active_embedding_dim) + ")");
  }
  if (meta_.expected_tap_layer.has_value() && active_num_tap_layers > 0) {
    if (*meta_.expected_tap_layer < 0 || *meta_.expected_tap_layer >= active_num_tap_layers) {
      throw std::invalid_argument("platos_key: bound tap_layer=" + std::to_string(*meta_.expected_tap_layer) +
                                  " out of active model range [0, " + std::to_string(active_num_tap_layers - 1) + "]");
    }
  }
  if (!meta_.donor_model.empty()) {
    const std::string donor_ckpt = canonical_checkpoint_id(meta_.donor_model);
    const std::string active_ckpt = canonical_checkpoint_id(active_model_id);
    if (donor_ckpt.empty() || active_ckpt.empty() || donor_ckpt != active_ckpt) {
      throw std::invalid_argument("platos_key: donor_model mismatch (key binds '" + meta_.donor_model +
                                  "' [canonical='" + donor_ckpt + "'], active runtime is model='" +
                                  std::string(active_model_id) + "' [canonical='" + active_ckpt + "'])");
    }
  }
}

void PlatosKey::validate_request_tap(std::optional<int> requested_tap_layer) const {
  if (meta_.is_multi_tap) {
    throw std::invalid_argument("platos_key: multi-tap concat spec '" + meta_.tap_spec +
                                "' cannot be executed by single-tap /v1/embeddings");
  }
  if (meta_.expected_tap_layer.has_value()) {
    if (!requested_tap_layer.has_value()) {
      throw std::invalid_argument("platos_key is bound to tap_layer=" + std::to_string(*meta_.expected_tap_layer) +
                                  " (tap_spec='" + meta_.tap_spec + "'); request must specify tap_layer=" +
                                  std::to_string(*meta_.expected_tap_layer));
    }
    if (*requested_tap_layer != *meta_.expected_tap_layer) {
      throw std::invalid_argument("requested tap_layer=" + std::to_string(*requested_tap_layer) +
                                  " conflicts with loaded platos_key bound to tap_layer=" +
                                  std::to_string(*meta_.expected_tap_layer));
    }
  } else if (meta_.requires_final_layer && requested_tap_layer.has_value()) {
    throw std::invalid_argument("requested tap_layer=" + std::to_string(*requested_tap_layer) +
                                " conflicts with loaded platos_key bound to final layer (tap_spec='" + meta_.tap_spec +
                                "')");
  }
}

void PlatosKey::validate_request_identity(std::string_view requested_cartridge_id,
                                          std::string_view requested_cartridge_sha256) const {
  if (!requested_cartridge_sha256.empty()) {
    const std::string req_sha = normalize_sha256_hex_strict(requested_cartridge_sha256, "cartridge_sha256");
    if (req_sha != meta_.file_sha256) {
      throw std::invalid_argument("requested cartridge_sha256=" + req_sha +
                                  " does not match loaded platos_key sha256=" + meta_.file_sha256);
    }
  }
  if (!requested_cartridge_id.empty()) {
    if (meta_.cartridge_id.empty() || requested_cartridge_id != meta_.cartridge_id) {
      throw std::invalid_argument("requested cartridge_id='" + std::string(requested_cartridge_id) +
                                  "' does not match loaded platos_key cartridge_id='" + meta_.cartridge_id + "'");
    }
  }
}

void PlatosKey::project_cpu(const float* in_x, int batch, float* out_y, bool l2_normalize) const {
  if (batch <= 0) return;
  if (!in_x || !out_y) throw std::invalid_argument("platos_key: null buffer in project_cpu");
  const int d_in = meta_.d_in;
  const int d_out = meta_.d_out;
  const double sc = static_cast<double>(meta_.scale);

  std::vector<float> staged;
  const float* x_src = in_x;
  if (in_x == out_y) {
    staged.assign(in_x, in_x + static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in));
    x_src = staged.data();
  }

  const int mask_coord = meta_.forbidden_coord_d2027_zero_masked ? 2027 : -1;
  std::vector<double> centered(static_cast<std::size_t>(d_in));
  for (int b = 0; b < batch; ++b) {
    const float* x_row = x_src + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in);
    float* y_row = out_y + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_out);
    for (int i = 0; i < d_in; ++i) {
      const double xv = (i == mask_coord) ? 0.0 : static_cast<double>(x_row[i]);
      centered[static_cast<std::size_t>(i)] = xv - static_cast<double>(donor_mean_[static_cast<std::size_t>(i)]);
    }
    for (int j = 0; j < d_out; ++j) {
      double acc = 0.0;
      for (int i = 0; i < d_in; ++i) {
        acc += centered[static_cast<std::size_t>(i)] *
               static_cast<double>(r_f32_[static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out) +
                                          static_cast<std::size_t>(j)]);
      }
      y_row[j] = static_cast<float>(static_cast<double>(target_mean_[static_cast<std::size_t>(j)]) + sc * acc);
    }
  }
  if (l2_normalize) {
    l2_normalize_rows_cpu(out_y, batch, d_out);
  }
}

}  // namespace ingine
