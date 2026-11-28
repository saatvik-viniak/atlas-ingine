// Configuration and task-prompt metadata parser for google/embeddinggemma-2 (#968).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::backends::embedding_gemma2 {

enum class LayerAttnKind : std::uint8_t {
  Sliding = 0,
  Full = 1,
};

struct LayerSpec {
  LayerAttnKind kind = LayerAttnKind::Sliding;
  int num_attention_heads = 4;
  int num_key_value_heads = 2;
  int head_dim = 256;
  double rope_theta = 10000.0;
};

struct Eg2Config {
  std::string model_type = "embedding_gemma2";
  std::vector<std::string> architectures;

  // Text encoder config
  int vocab_size = 262144;
  int hidden_size = 512;
  int intermediate_size = 2048;
  int hidden_size_per_layer_input = 512;
  int num_hidden_layers = 24;
  int num_attention_heads = 4;
  int num_key_value_heads = 2;
  int head_dim = 256;
  int sliding_window = 512;
  int max_position_embeddings = 262144;
  int embedding_dim = 768;
  float rms_norm_eps = 1e-6f;
  double sliding_rope_theta = 10000.0;
  double full_rope_theta = 1000000.0;

  int pad_token_id = 0;
  int eos_token_id = 1;
  int bos_token_id = 2;
  int boi_token_id = 255999;
  int boa_token_id = 256000;
  int image_token_id = 258880;
  int audio_token_id = 258881;
  int eoi_token_id = 258882;
  int eoa_token_id = 258883;
  int video_token_id = 258884;

  // Per-layer specs (size == num_hidden_layers)
  std::vector<LayerSpec> layers;

  // Matryoshka supported output dimensions (descending, native first)
  std::vector<int> matryoshka_dims = {768, 512, 256, 128};

  // Presence of multimodal towers in config.json
  bool has_vision_config = false;
  bool has_audio_config = false;
  int vision_soft_tokens_per_image = 280;

  // Named task prefixes from config_sentence_transformers.json (optional)
  std::map<std::string, std::string> prompts;
  std::optional<std::string> default_prompt_name;

  static Eg2Config load(const std::filesystem::path& model_dir);
  static Eg2Config parse(std::string_view config_json, std::string_view st_config_json = {});
  void validate() const;
  std::optional<std::string_view> find_prompt_prefix(std::string_view prompt_name) const noexcept;
};

}  // namespace ingine::backends::embedding_gemma2
