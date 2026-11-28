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

struct Eg2ClippedLinearF32 {
  std::vector<float> weight;
  float input_min = -1e30f;
  float input_max = 1e30f;
  float output_min = -1e30f;
  float output_max = 1e30f;
};

struct Eg2ClippedLinearBf16 {
  std::vector<std::uint16_t> weight;
  float input_min = -1e30f;
  float input_max = 1e30f;
  float output_min = -1e30f;
  float output_max = 1e30f;
};

struct Eg2AudioFfwWeightsF32 {
  std::vector<float> pre_layer_norm;   // [AH]
  Eg2ClippedLinearF32 ffw_layer_1;     // [4*AH, AH]
  Eg2ClippedLinearF32 ffw_layer_2;     // [AH, 4*AH]
  std::vector<float> post_layer_norm;  // [AH]
};

struct Eg2AudioFfwWeightsBf16 {
  std::vector<std::uint16_t> pre_layer_norm;
  Eg2ClippedLinearBf16 ffw_layer_1;
  Eg2ClippedLinearBf16 ffw_layer_2;
  std::vector<std::uint16_t> post_layer_norm;
};

struct Eg2AudioLayerWeightsF32 {
  Eg2AudioFfwWeightsF32 feed_forward1;
  std::vector<float> norm_pre_attn;          // [AH]
  Eg2ClippedLinearF32 q_proj;                // [AH, AH]
  Eg2ClippedLinearF32 k_proj;                // [AH, AH]
  Eg2ClippedLinearF32 v_proj;                // [AH, AH]
  std::vector<float> relative_k_proj;        // [AH, AH]
  std::vector<float> per_dim_scale;          // [AHD]
  Eg2ClippedLinearF32 post;                  // [AH, AH]
  std::vector<float> norm_post_attn;         // [AH]
  std::vector<float> lconv_pre_layer_norm;   // [AH]
  Eg2ClippedLinearF32 lconv_linear_start;    // [2*AH, AH]
  std::vector<float> lconv_depthwise_conv1d; // [AH, 1, K]
  std::vector<float> lconv_conv_norm;        // [AH]
  Eg2ClippedLinearF32 lconv_linear_end;      // [AH, AH]
  Eg2AudioFfwWeightsF32 feed_forward2;
  std::vector<float> norm_out;               // [AH]
};

struct Eg2AudioLayerWeightsBf16 {
  Eg2AudioFfwWeightsBf16 feed_forward1;
  std::vector<std::uint16_t> norm_pre_attn;
  Eg2ClippedLinearBf16 q_proj;
  Eg2ClippedLinearBf16 k_proj;
  Eg2ClippedLinearBf16 v_proj;
  std::vector<std::uint16_t> relative_k_proj;
  std::vector<float> per_dim_scale;  // kept FP32 for exact softplus(per_dim_scale)
  Eg2ClippedLinearBf16 post;
  std::vector<std::uint16_t> norm_post_attn;
  std::vector<std::uint16_t> lconv_pre_layer_norm;
  Eg2ClippedLinearBf16 lconv_linear_start;
  std::vector<std::uint16_t> lconv_depthwise_conv1d;
  std::vector<std::uint16_t> lconv_conv_norm;
  Eg2ClippedLinearBf16 lconv_linear_end;
  Eg2AudioFfwWeightsBf16 feed_forward2;
  std::vector<std::uint16_t> norm_out;
};

struct Eg2AudioWeightsF32 {
  Eg2AudioConfig config;
  int text_hidden_size = 512;
  std::vector<float> sscp_layer0_conv;        // [C0, 1, 3, 3]
  std::vector<float> sscp_layer0_norm;        // [C0]
  std::vector<float> sscp_layer1_conv;        // [C1, C0, 3, 3]
  std::vector<float> sscp_layer1_norm;        // [C1]
  std::vector<float> sscp_input_proj_linear;  // [AH, (128/4)*C1]
  std::vector<Eg2AudioLayerWeightsF32> layers;
  std::vector<float> output_proj_weight;      // [output_proj_dims, AH]
  std::vector<float> output_proj_bias;        // [output_proj_dims]
  std::vector<float> embed_audio_projection;  // [text_H, output_proj_dims]
};

struct Eg2AudioWeightsBf16 {
  Eg2AudioConfig config;
  int text_hidden_size = 512;
  std::vector<std::uint16_t> sscp_layer0_conv;
  std::vector<std::uint16_t> sscp_layer0_norm;
  std::vector<std::uint16_t> sscp_layer1_conv;
  std::vector<std::uint16_t> sscp_layer1_norm;
  std::vector<std::uint16_t> sscp_input_proj_linear;
  std::vector<Eg2AudioLayerWeightsBf16> layers;
  std::vector<std::uint16_t> output_proj_weight;
  std::vector<std::uint16_t> output_proj_bias;
  std::vector<std::uint16_t> embed_audio_projection;
};

// Selectively reads ONLY language_model.* tensors as raw BF16 uint16_t words (used by sm_80+ CUDA).
Eg2TextWeightsBf16 load_eg2_text_weights_bf16(const std::filesystem::path& model_dir);

// Selectively reads ONLY vision_tower.* + embed_vision.* tensors (211 tensors; 319.97 MiB BF16 / 639.94 MiB FP32).
Eg2VisionWeightsF32 load_eg2_vision_weights_f32(const std::filesystem::path& model_dir);
Eg2VisionWeightsBf16 load_eg2_vision_weights_bf16(const std::filesystem::path& model_dir);

// Selectively reads ONLY audio_tower.* + embed_audio.* tensors (752 tensors; 582.91 MiB BF16 / 1165.82 MiB FP32).
Eg2AudioWeightsF32 load_eg2_audio_weights_f32(const std::filesystem::path& model_dir);
Eg2AudioWeightsBf16 load_eg2_audio_weights_bf16(const std::filesystem::path& model_dir);

}  // namespace ingine::backends::embedding_gemma2
