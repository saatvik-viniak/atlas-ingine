#include "ingine/platos_key.h"

#include "core/act.h"
#include "core/safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ingine {
namespace {

std::vector<float> decode_tensor_f32(const SafetensorsFile& file, const SafetensorsTensor& t) {
  const std::vector<std::byte> raw = file.read(t.name);
  const std::size_t n = static_cast<std::size_t>(t.numel());
  std::vector<float> out(n);
  if (t.dtype == "F32") {
    if (raw.size() != n * 4) throw std::invalid_argument("platos_key: F32 byte size mismatch for " + t.name);
    std::memcpy(out.data(), raw.data(), raw.size());
  } else if (t.dtype == "F16") {
    if (raw.size() != n * 2) throw std::invalid_argument("platos_key: F16 byte size mismatch for " + t.name);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, raw.data() + i * 2, 2);
      out[i] = act::host_fp16_bits_to_float(bits);
    }
  } else if (t.dtype == "BF16") {
    if (raw.size() != n * 2) throw std::invalid_argument("platos_key: BF16 byte size mismatch for " + t.name);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, raw.data() + i * 2, 2);
      out[i] = act::host_bf16_bits_to_float(bits);
    }
  } else {
    throw std::invalid_argument("platos_key: unsupported tensor dtype '" + t.dtype + "' for '" + t.name +
                                "' (expected F16, BF16, or F32)");
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

void merge_and_validate_json_metadata(std::string_view json_text, PlatosKeyMetadata& meta) {
  if (json_text.empty()) return;
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

  auto check_dim = [&](const char* key, int actual) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return;
    int declared = -1;
    if (it->is_number_integer()) {
      declared = it->get<int>();
    } else if (it->is_string()) {
      try {
        declared = std::stoi(it->get<std::string>());
      } catch (...) {
        throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "' is not an integer");
      }
    } else {
      throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "' must be an integer");
    }
    if (declared != actual) {
      throw std::invalid_argument(std::string("platos_key: metadata JSON '") + key + "'=" + std::to_string(declared) +
                                  " contradicts safetensors shape (" + std::to_string(actual) + ")");
    }
  };
  check_dim("d_donor", meta.d_in);
  check_dim("d_in", meta.d_in);
  check_dim("d_target", meta.d_out);
  check_dim("d_out", meta.d_out);

  for (const auto& [k, v] : obj.items()) {
    if (v.is_string()) {
      meta.raw_metadata[k] = v.get<std::string>();
    } else if (v.is_number() || v.is_boolean()) {
      meta.raw_metadata[k] = v.dump();
    }
  }
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

  SafetensorsFile file(safetensors_path);
  const SafetensorsTensor* mat_t = file.find("R");
  std::string detected_format = "platos_key_v1";
  if (!mat_t) {
    mat_t = file.find("W");
    detected_format = "affine_wb";
  }
  if (!mat_t) {
    throw std::invalid_argument("platos_key: '" + safetensors_path.string() +
                                "' missing required 2D projection tensor 'R' (or 'W')");
  }
  if (mat_t->shape.size() != 2 || mat_t->shape[0] <= 0 || mat_t->shape[1] <= 0) {
    throw std::invalid_argument("platos_key: projection tensor '" + mat_t->name + "' must be 2D with positive dims");
  }
  const int d_in = static_cast<int>(mat_t->shape[0]);
  const int d_out = static_cast<int>(mat_t->shape[1]);
  if (expected_in_dim > 0 && d_in != expected_in_dim) {
    throw std::invalid_argument("platos_key: donor dim mismatch in '" + safetensors_path.string() + "' (key d_in=" +
                                std::to_string(d_in) + ", model hidden_dim=" + std::to_string(expected_in_dim) + ")");
  }

  std::shared_ptr<PlatosKey> key(new PlatosKey());
  key->source_path_ = safetensors_path;
  key->meta_.d_in = d_in;
  key->meta_.d_out = d_out;
  key->meta_.weight_dtype = mat_t->dtype;
  key->meta_.raw_metadata = file.metadata();

  key->r_f32_ = decode_tensor_f32(file, *mat_t);
  key->rt_f16_.resize(static_cast<std::size_t>(d_out) * static_cast<std::size_t>(d_in));
  if (mat_t->dtype == "F16") {
    // Preserve exact IEEE-754 binary16 bit patterns (including subnormals) without a float roundtrip.
    const std::vector<std::byte> raw_r = file.read(mat_t->name);
    for (int i = 0; i < d_in; ++i) {
      const std::size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
      for (int j = 0; j < d_out; ++j) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, raw_r.data() + (row_off + static_cast<std::size_t>(j)) * 2, 2);
        key->rt_f16_[static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in) + static_cast<std::size_t>(i)] = bits;
      }
    }
  } else {
    for (int i = 0; i < d_in; ++i) {
      const float* src_row = key->r_f32_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
      for (int j = 0; j < d_out; ++j) {
        key->rt_f16_[static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in) + static_cast<std::size_t>(i)] =
            act::host_float_to_fp16_bits(src_row[j]);
      }
    }
  }

  key->donor_mean_.assign(static_cast<std::size_t>(d_in), 0.0f);
  if (const SafetensorsTensor* dm = file.find("donor_mean")) {
    if (dm->shape.size() != 1 || dm->shape[0] != d_in) {
      throw std::invalid_argument("platos_key: 'donor_mean' shape must be [" + std::to_string(d_in) + "]");
    }
    key->donor_mean_ = decode_tensor_f32(file, *dm);
    key->has_donor_mean_ = true;
  }

  key->target_mean_.assign(static_cast<std::size_t>(d_out), 0.0f);
  const SafetensorsTensor* tm = file.find("target_mean");
  if (!tm && detected_format == "affine_wb") tm = file.find("b");
  if (tm) {
    if (tm->shape.size() != 1 || tm->shape[0] != d_out) {
      throw std::invalid_argument("platos_key: '" + tm->name + "' shape must be [" + std::to_string(d_out) + "]");
    }
    key->target_mean_ = decode_tensor_f32(file, *tm);
    key->has_target_mean_ = true;
  }

  key->meta_.scale = 1.0f;
  if (const SafetensorsTensor* sc = file.find("scale")) {
    if (sc->numel() != 1) {
      throw std::invalid_argument("platos_key: 'scale' tensor must have exactly 1 element");
    }
    key->meta_.scale = decode_tensor_f32(file, *sc)[0];
  }

  merge_and_validate_json_metadata(metadata_json, key->meta_);

  key->meta_.format = meta_get_first(key->meta_.raw_metadata, {"format"});
  if (key->meta_.format.empty()) key->meta_.format = detected_format;
  key->meta_.schema_version = meta_get_first(key->meta_.raw_metadata, {"schema_version"});
  key->meta_.donor_model = meta_get_first(key->meta_.raw_metadata, {"donor_model", "source_model", "model_id"});
  key->meta_.target_model = meta_get_first(key->meta_.raw_metadata, {"target_model"});
  key->meta_.tap_spec = meta_get_first(key->meta_.raw_metadata, {"tap_spec", "target_tap"});
  key->meta_.method = meta_get_first(key->meta_.raw_metadata, {"method", "solver"});
  return key;
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

  std::vector<double> centered(static_cast<std::size_t>(d_in));
  for (int b = 0; b < batch; ++b) {
    const float* x_row = x_src + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in);
    float* y_row = out_y + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_out);
    for (int i = 0; i < d_in; ++i) {
      centered[static_cast<std::size_t>(i)] =
          static_cast<double>(x_row[i]) - static_cast<double>(donor_mean_[static_cast<std::size_t>(i)]);
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
