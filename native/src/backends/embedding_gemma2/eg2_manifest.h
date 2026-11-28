// Selective safetensors manifest and weight loader for google/embeddinggemma-2 (#968).
//
// Enforces EmbeddingGemma-2 precision invariants: weights are loaded as BF16 or upcast to FP32.
// FP16 weights and activations are strictly forbidden (norm weights reach 39.25, embed scale 22.625).
#pragma once

#include "backends/embedding_gemma2/eg2_config.h"
#include "core/safetensors.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ingine::backends::embedding_gemma2 {

struct Eg2LayerWeightsF32 {
  std::vector<float> input_layernorm;          // [H]
  std::vector<float> q_proj;                   // [n_heads * hd, H]
  std::vector<float> k_proj;                   // [n_kv_heads * hd, H]
  std::vector<float> v_proj;                   // [n_kv_heads * hd, H]
  std::vector<float> o_proj;                   // [H, n_heads * hd]
  std::vector<float> q_norm;                   // [hd]
  std::vector<float> k_norm;                   // [hd]
  std::vector<float> post_attention_layernorm; // [H]
  std::vector<float> pre_feedforward_layernorm;// [H]
  std::vector<float> gate_proj;                // [inter, H]
  std::vector<float> up_proj;                  // [inter, H]
  std::vector<float> down_proj;                // [H, inter]
  std::vector<float> post_feedforward_layernorm;// [H]
  std::vector<float> per_layer_input_gate;     // [H, H]
  std::vector<float> per_layer_projection;     // [H, H]
  std::vector<float> post_per_layer_input_norm;// [H]
  float layer_scalar = 1.0f;
};

struct Eg2TextWeightsF32 {
  Eg2Config config;
  std::vector<float> embed_tokens;                // [vocab, H]
  std::vector<float> per_layer_model_projection;  // [num_layers * H, H]
  std::vector<float> per_layer_projection_norm;   // [H]
  std::vector<Eg2LayerWeightsF32> layers;         // [num_layers]
  std::vector<float> norm;                        // [H]
  std::vector<float> embedding_projection;        // [emb_dim, H]
};

struct Eg2LayerWeightsBf16 {
  std::vector<std::uint16_t> input_layernorm;
  std::vector<std::uint16_t> q_proj;
  std::vector<std::uint16_t> k_proj;
  std::vector<std::uint16_t> v_proj;
  std::vector<std::uint16_t> o_proj;
  std::vector<std::uint16_t> q_norm;
  std::vector<std::uint16_t> k_norm;
  std::vector<std::uint16_t> post_attention_layernorm;
  std::vector<std::uint16_t> pre_feedforward_layernorm;
  std::vector<std::uint16_t> gate_proj;
  std::vector<std::uint16_t> up_proj;
  std::vector<std::uint16_t> down_proj;
  std::vector<std::uint16_t> post_feedforward_layernorm;
  std::vector<std::uint16_t> per_layer_input_gate;
  std::vector<std::uint16_t> per_layer_projection;
  std::vector<std::uint16_t> post_per_layer_input_norm;
  float layer_scalar = 1.0f;
};

struct Eg2TextWeightsBf16 {
  Eg2Config config;
  std::vector<std::uint16_t> embed_tokens;
  std::vector<std::uint16_t> per_layer_model_projection;
  std::vector<std::uint16_t> per_layer_projection_norm;
  std::vector<Eg2LayerWeightsBf16> layers;
  std::vector<std::uint16_t> norm;
  std::vector<std::uint16_t> embedding_projection;
};

struct Eg2ManifestSummary {
  std::filesystem::path model_dir;
  Eg2Config config;
  std::size_t total_tensors = 0;
  std::size_t text_tensors = 0;
  std::size_t vision_tensors = 0;
  std::size_t audio_tensors = 0;
  std::uint64_t text_bytes = 0;
  std::uint64_t vision_bytes = 0;
  std::uint64_t audio_bytes = 0;
  bool vision_present = false;
  bool audio_present = false;
};

// Inspects config.json + safetensors headers without reading weight payloads; validates every
// required language_model.* tensor name, shape, and BF16/F32 dtype fail-closed.
Eg2ManifestSummary inspect_eg2_manifest(const std::filesystem::path& model_dir);

// Selectively reads ONLY language_model.* tensors (skipping vision_tower.* and audio_tower.*)
// and returns FP32 host weights (used by CPU backend and sm_70 FP32 CUDA backend).
Eg2TextWeightsF32 load_eg2_text_weights_f32(const std::filesystem::path& model_dir);

struct Eg2VisionLayerWeightsF32 {
  std::vector<float> input_layernorm;            // [VH]
  std::vector<float> q_proj;                     // [VH, VH]
  std::vector<float> k_proj;                     // [VH, VH]
  std::vector<float> v_proj;                     // [VH, VH]
  std::vector<float> o_proj;                     // [VH, VH]
  std::vector<float> q_norm;                     // [VHD]
  std::vector<float> k_norm;                     // [VHD]
  std::vector<float> post_attention_layernorm;   // [VH]
  std::vector<float> pre_feedforward_layernorm;  // [VH]
  std::vector<float> gate_proj;                  // [VI, VH]
  std::vector<float> up_proj;                    // [VI, VH]
  std::vector<float> down_proj;                  // [VH, VI]
  std::vector<float> post_feedforward_layernorm; // [VH]
};

struct Eg2VisionWeightsF32 {
  Eg2VisionConfig config;
  int text_hidden_size = 512;
  std::vector<float> patch_input_proj;           // [VH, 3 * P * P]
  std::vector<float> position_embedding_table;   // [2, pos_emb_size, VH]
  std::vector<Eg2VisionLayerWeightsF32> layers;  // [num_vision_layers]
  std::vector<float> embed_vision_projection;    // [text_H, VH]
};

struct Eg2VisionLayerWeightsBf16 {
  std::vector<std::uint16_t> input_layernorm;
  std::vector<std::uint16_t> q_proj;
  std::vector<std::uint16_t> k_proj;
  std::vector<std::uint16_t> v_proj;
  std::vector<std::uint16_t> o_proj;
  std::vector<std::uint16_t> q_norm;
  std::vector<std::uint16_t> k_norm;
  std::vector<std::uint16_t> post_attention_layernorm;
  std::vector<std::uint16_t> pre_feedforward_layernorm;
  std::vector<std::uint16_t> gate_proj;
  std::vector<std::uint16_t> up_proj;
  std::vector<std::uint16_t> down_proj;
  std::vector<std::uint16_t> post_feedforward_layernorm;
};

struct Eg2VisionWeightsBf16 {
  Eg2VisionConfig config;
  int text_hidden_size = 512;
  std::vector<std::uint16_t> patch_input_proj;
  std::vector<std::uint16_t> position_embedding_table;
  std::vector<Eg2VisionLayerWeightsBf16> layers;
  std::vector<std::uint16_t> embed_vision_projection;
};

// Selectively reads ONLY language_model.* tensors as raw BF16 uint16_t words (used by sm_80+ CUDA).
Eg2TextWeightsBf16 load_eg2_text_weights_bf16(const std::filesystem::path& model_dir);

// Selectively reads ONLY vision_tower.* + embed_vision.* tensors (211 tensors; 319.97 MiB BF16 / 639.94 MiB FP32).
Eg2VisionWeightsF32 load_eg2_vision_weights_f32(const std::filesystem::path& model_dir);
Eg2VisionWeightsBf16 load_eg2_vision_weights_bf16(const std::filesystem::path& model_dir);

}  // namespace ingine::backends::embedding_gemma2
