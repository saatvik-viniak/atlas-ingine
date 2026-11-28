#include "backends/embedding_gemma2/eg2_config.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ingine::backends::embedding_gemma2 {
namespace {

std::string read_text_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

Eg2Config Eg2Config::load(const std::filesystem::path& model_dir) {
  const std::filesystem::path cfg_path = model_dir / "config.json";
  const std::string cfg_str = read_text_file(cfg_path);
  std::string st_str;
  const std::filesystem::path st_path = model_dir / "config_sentence_transformers.json";
  if (std::filesystem::is_regular_file(st_path)) {
    st_str = read_text_file(st_path);
  }
  return parse(cfg_str, st_str);
}

Eg2Config Eg2Config::parse(std::string_view config_json, std::string_view st_config_json) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(config_json);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("EmbeddingGemma2 config.json parse error: ") + e.what());
  }
  if (!root.is_object()) throw std::invalid_argument("EmbeddingGemma2 config.json must be an object");

  Eg2Config cfg;
  cfg.model_type = root.value("model_type", std::string{"embedding_gemma2"});
  if (auto it = root.find("architectures"); it != root.end() && it->is_array()) {
    for (const auto& a : *it) {
      if (a.is_string()) cfg.architectures.push_back(a.get<std::string>());
    }
  }

  cfg.boi_token_id = root.value("boi_token_id", cfg.boi_token_id);
  cfg.boa_token_id = root.value("boa_token_id", cfg.boa_token_id);
  cfg.image_token_id = root.value("image_token_id", cfg.image_token_id);
  cfg.audio_token_id = root.value("audio_token_id", cfg.audio_token_id);
  cfg.eoi_token_id = root.value("eoi_token_id", cfg.eoi_token_id);
  cfg.eoa_token_id = root.value("eoa_token_index", root.value("eoa_token_id", cfg.eoa_token_id));
  cfg.video_token_id = root.value("video_token_id", cfg.video_token_id);
  cfg.vision_soft_tokens_per_image = root.value("vision_soft_tokens_per_image", cfg.vision_soft_tokens_per_image);
  cfg.has_vision_config = root.contains("vision_config") && root["vision_config"].is_object();
  cfg.has_audio_config = root.contains("audio_config") && root["audio_config"].is_object();

  const nlohmann::json& tc = root.contains("text_config") && root["text_config"].is_object() ? root["text_config"] : root;
  cfg.vocab_size = tc.value("vocab_size", cfg.vocab_size);
  cfg.hidden_size = tc.value("hidden_size", cfg.hidden_size);
  cfg.intermediate_size = tc.value("intermediate_size", cfg.intermediate_size);
  cfg.hidden_size_per_layer_input = tc.value("hidden_size_per_layer_input", cfg.hidden_size);
  cfg.num_hidden_layers = tc.value("num_hidden_layers", cfg.num_hidden_layers);
  cfg.num_attention_heads = tc.value("num_attention_heads", cfg.num_attention_heads);
  cfg.num_key_value_heads = tc.value("num_key_value_heads", cfg.num_key_value_heads);
  cfg.head_dim = tc.value("head_dim", cfg.head_dim);
  cfg.sliding_window = tc.value("sliding_window", cfg.sliding_window);
  cfg.max_position_embeddings = tc.value("max_position_embeddings", cfg.max_position_embeddings);
  cfg.embedding_dim = tc.value("embedding_dim", cfg.embedding_dim);
  cfg.rms_norm_eps = tc.value("rms_norm_eps", cfg.rms_norm_eps);
  cfg.pad_token_id = tc.value("pad_token_id", cfg.pad_token_id);
  cfg.eos_token_id = tc.value("eos_token_id", cfg.eos_token_id);
  cfg.bos_token_id = tc.value("bos_token_id", cfg.bos_token_id);

  if (auto rp = tc.find("rope_parameters"); rp != tc.end() && rp->is_object()) {
    if (auto sl = rp->find("sliding_attention"); sl != rp->end() && sl->is_object()) {
      cfg.sliding_rope_theta = sl->value("rope_theta", cfg.sliding_rope_theta);
    }
    if (auto fl = rp->find("full_attention"); fl != rp->end() && fl->is_object()) {
      cfg.full_rope_theta = fl->value("rope_theta", cfg.full_rope_theta);
    }
  }

  // Build per-layer specs from layer_types and per_layer_config ("05", "11", "17", "23").
  cfg.layers.resize(static_cast<std::size_t>(std::max(0, cfg.num_hidden_layers)));
  std::vector<std::string> layer_types;
  if (auto lt = tc.find("layer_types"); lt != tc.end() && lt->is_array()) {
    for (const auto& item : *lt) {
      if (item.is_string()) layer_types.push_back(item.get<std::string>());
    }
  }
  const nlohmann::json* plc =
      (tc.contains("per_layer_config") && tc["per_layer_config"].is_object()) ? &tc["per_layer_config"] : nullptr;

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    LayerSpec spec;
    const std::string ltype = (i < static_cast<int>(layer_types.size())) ? layer_types[static_cast<std::size_t>(i)]
                                                                         : std::string("sliding_attention");
    if (ltype == "full_attention") {
      spec.kind = LayerAttnKind::Full;
      spec.rope_theta = cfg.full_rope_theta;
    } else if (ltype == "sliding_attention") {
      spec.kind = LayerAttnKind::Sliding;
      spec.rope_theta = cfg.sliding_rope_theta;
    } else {
      throw std::invalid_argument("EmbeddingGemma2: unsupported layer_type '" + ltype + "' at layer " +
                                  std::to_string(i));
    }
    spec.num_attention_heads = cfg.num_attention_heads;
    spec.num_key_value_heads = cfg.num_key_value_heads;
    spec.head_dim = cfg.head_dim;

    if (plc != nullptr) {
      char key2[16];
      std::snprintf(key2, sizeof(key2), "%02d", i);
      const std::string key1 = std::to_string(i);
      const nlohmann::json* ov = nullptr;
      if (auto it = plc->find(key2); it != plc->end() && it->is_object()) ov = &*it;
      else if (auto it2 = plc->find(key1); it2 != plc->end() && it2->is_object()) ov = &*it2;
      if (ov != nullptr) {
        spec.num_attention_heads = ov->value("num_attention_heads", spec.num_attention_heads);
        spec.num_key_value_heads = ov->value("num_key_value_heads", spec.num_key_value_heads);
        spec.head_dim = ov->value("head_dim", spec.head_dim);
      }
    }
    cfg.layers[static_cast<std::size_t>(i)] = spec;
  }

  // Standard Matryoshka dims clamped to embedding_dim.
  cfg.matryoshka_dims.clear();
  for (int d : {768, 512, 256, 128}) {
    if (d <= cfg.embedding_dim) cfg.matryoshka_dims.push_back(d);
  }
  if (cfg.matryoshka_dims.empty() || cfg.matryoshka_dims.front() != cfg.embedding_dim) {
    cfg.matryoshka_dims.insert(cfg.matryoshka_dims.begin(), cfg.embedding_dim);
  }

  if (!st_config_json.empty()) {
    nlohmann::json st;
    try {
      st = nlohmann::json::parse(st_config_json);
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument(std::string("EmbeddingGemma2 config_sentence_transformers.json parse error: ") +
                                  e.what());
    }
    if (st.is_object()) {
      if (auto dp = st.find("default_prompt_name"); dp != st.end() && dp->is_string()) {
        cfg.default_prompt_name = dp->get<std::string>();
      }
      if (auto pr = st.find("prompts"); pr != st.end() && pr->is_object()) {
        for (const auto& [k, v] : pr->items()) {
          if (v.is_string()) cfg.prompts.emplace(k, v.get<std::string>());
        }
      }
    }
  }

  cfg.validate();
  return cfg;
}

void Eg2Config::validate() const {
  if (vocab_size <= 0 || hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 || embedding_dim <= 0) {
    throw std::invalid_argument("EmbeddingGemma2 config: core dimensions must be positive");
  }
  if (hidden_size_per_layer_input != hidden_size) {
    throw std::invalid_argument("EmbeddingGemma2 config: hidden_size_per_layer_input must equal hidden_size");
  }
  if (sliding_window <= 0) {
    throw std::invalid_argument("EmbeddingGemma2 config: sliding_window must be positive");
  }
  if (static_cast<int>(layers.size()) != num_hidden_layers) {
    throw std::invalid_argument("EmbeddingGemma2 config: layers size mismatch");
  }
  for (int i = 0; i < num_hidden_layers; ++i) {
    const LayerSpec& s = layers[static_cast<std::size_t>(i)];
    if (s.num_attention_heads <= 0 || s.num_key_value_heads <= 0 || s.head_dim <= 0 || (s.head_dim % 2) != 0 ||
        (s.num_attention_heads % s.num_key_value_heads) != 0) {
      throw std::invalid_argument("EmbeddingGemma2 config: invalid attention geometry at layer " + std::to_string(i));
    }
  }
}

std::optional<std::string_view> Eg2Config::find_prompt_prefix(std::string_view prompt_name) const noexcept {
  const auto it = prompts.find(std::string(prompt_name));
  if (it == prompts.end()) return std::nullopt;
  return std::string_view(it->second);
}

}  // namespace ingine::backends::embedding_gemma2
