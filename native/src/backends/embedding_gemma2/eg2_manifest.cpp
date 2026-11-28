#include "backends/embedding_gemma2/eg2_manifest.h"

#include "core/act.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::backends::embedding_gemma2 {
namespace {

bool starts_with(std::string_view s, std::string_view prefix) noexcept {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

void require_tensor_shape_and_dtype(const SafetensorsModel& model, std::string_view name,
                                    const std::vector<std::int64_t>& expected_shape) {
  const SafetensorsRef ref = model.find(name);
  if (!ref) {
    throw std::invalid_argument("EmbeddingGemma2 safetensors missing required tensor: " + std::string(name));
  }
  if (ref->dtype != "BF16" && ref->dtype != "F32") {
    throw std::invalid_argument("EmbeddingGemma2 tensor '" + std::string(name) + "' has unsupported dtype '" +
                                ref->dtype + "' (only BF16/F32 permitted; FP16 is forbidden)");
  }
  if (ref->shape != expected_shape) {
    auto fmt_shape = [](const std::vector<std::int64_t>& sh) {
      std::string out = "[";
      for (std::size_t i = 0; i < sh.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(sh[i]);
      }
      return out + "]";
    };
    throw std::invalid_argument("EmbeddingGemma2 tensor '" + std::string(name) + "' shape mismatch: got " +
                                fmt_shape(ref->shape) + ", expected " + fmt_shape(expected_shape));
  }
}

void validate_text_tensors(const SafetensorsModel& model, const Eg2Config& cfg) {
  const std::int64_t V = cfg.vocab_size;
  const std::int64_t H = cfg.hidden_size;
  const std::int64_t I = cfg.intermediate_size;
  const std::int64_t L = cfg.num_hidden_layers;
  const std::int64_t E = cfg.embedding_dim;

  require_tensor_shape_and_dtype(model, "language_model.embed_tokens.weight", {V, H});
  require_tensor_shape_and_dtype(model, "language_model.ple.per_layer_model_projection.weight", {L * H, H});
  require_tensor_shape_and_dtype(model, "language_model.ple.per_layer_projection_norm.weight", {H});
  require_tensor_shape_and_dtype(model, "language_model.norm.weight", {H});
  require_tensor_shape_and_dtype(model, "language_model.embedding_projection.weight", {E, H});

  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    const LayerSpec& ls = cfg.layers[static_cast<std::size_t>(i)];
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    const std::int64_t q_dim = static_cast<std::int64_t>(ls.num_attention_heads) * ls.head_dim;
    const std::int64_t kv_dim = static_cast<std::int64_t>(ls.num_key_value_heads) * ls.head_dim;
    const std::int64_t hd = ls.head_dim;

    require_tensor_shape_and_dtype(model, p + "input_layernorm.weight", {H});
    require_tensor_shape_and_dtype(model, p + "self_attn.q_proj.weight", {q_dim, H});
    require_tensor_shape_and_dtype(model, p + "self_attn.k_proj.weight", {kv_dim, H});
    require_tensor_shape_and_dtype(model, p + "self_attn.v_proj.weight", {kv_dim, H});
    require_tensor_shape_and_dtype(model, p + "self_attn.o_proj.weight", {H, q_dim});
    require_tensor_shape_and_dtype(model, p + "self_attn.q_norm.weight", {hd});
    require_tensor_shape_and_dtype(model, p + "self_attn.k_norm.weight", {hd});
    require_tensor_shape_and_dtype(model, p + "post_attention_layernorm.weight", {H});
    require_tensor_shape_and_dtype(model, p + "pre_feedforward_layernorm.weight", {H});
    require_tensor_shape_and_dtype(model, p + "mlp.gate_proj.weight", {I, H});
    require_tensor_shape_and_dtype(model, p + "mlp.up_proj.weight", {I, H});
    require_tensor_shape_and_dtype(model, p + "mlp.down_proj.weight", {H, I});
    require_tensor_shape_and_dtype(model, p + "post_feedforward_layernorm.weight", {H});
    require_tensor_shape_and_dtype(model, p + "ple_block.per_layer_input_gate.weight", {H, H});
    require_tensor_shape_and_dtype(model, p + "ple_block.per_layer_projection.weight", {H, H});
    require_tensor_shape_and_dtype(model, p + "ple_block.post_per_layer_input_norm.weight", {H});
    require_tensor_shape_and_dtype(model, p + "layer_scalar", {1});
  }
}

std::vector<float> read_tensor_f32(const SafetensorsModel& model, std::string_view name) {
  const SafetensorsRef ref = model.at(name);
  const std::size_t n = static_cast<std::size_t>(ref->numel());
  std::vector<float> out(n);
  if (ref->dtype == "F32") {
    ref.file->read(*ref.tensor, out.data());
    return out;
  }
  if (ref->dtype == "BF16") {
    std::vector<std::uint16_t> raw(n);
    ref.file->read(*ref.tensor, raw.data());
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = act::host_bf16_bits_to_float(raw[i]);
    }
    return out;
  }
  throw std::invalid_argument("EmbeddingGemma2: unsupported dtype '" + ref->dtype + "' for " + std::string(name));
}

std::vector<std::uint16_t> read_tensor_bf16(const SafetensorsModel& model, std::string_view name) {
  const SafetensorsRef ref = model.at(name);
  const std::size_t n = static_cast<std::size_t>(ref->numel());
  std::vector<std::uint16_t> out(n);
  if (ref->dtype == "BF16") {
    ref.file->read(*ref.tensor, out.data());
    return out;
  }
  if (ref->dtype == "F32") {
    std::vector<float> raw(n);
    ref.file->read(*ref.tensor, raw.data());
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = act::host_float_to_bf16_bits(raw[i]);
    }
    return out;
  }
  throw std::invalid_argument("EmbeddingGemma2: unsupported dtype '" + ref->dtype + "' for " + std::string(name));
}

float read_scalar_f32(const SafetensorsModel& model, std::string_view name) {
  const auto v = read_tensor_f32(model, name);
  if (v.size() != 1) throw std::invalid_argument("EmbeddingGemma2: expected 1-element scalar for " + std::string(name));
  return v[0];
}

}  // namespace

Eg2ManifestSummary inspect_eg2_manifest(const std::filesystem::path& model_dir) {
  Eg2ManifestSummary summary;
  summary.model_dir = model_dir;
  summary.config = Eg2Config::load(model_dir);

  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_text_tensors(model, summary.config);

  for (const auto& file : model.files()) {
    for (const auto& t : file->tensors()) {
      ++summary.total_tensors;
      if (starts_with(t.name, "language_model.")) {
        ++summary.text_tensors;
        summary.text_bytes += t.bytes;
      } else if (starts_with(t.name, "vision_tower.") || starts_with(t.name, "embed_vision.")) {
        ++summary.vision_tensors;
        summary.vision_bytes += t.bytes;
      } else if (starts_with(t.name, "audio_tower.") || starts_with(t.name, "embed_audio.")) {
        ++summary.audio_tensors;
        summary.audio_bytes += t.bytes;
      }
    }
  }
  summary.vision_present = summary.vision_tensors > 0;
  summary.audio_present = summary.audio_tensors > 0;
  return summary;
}

Eg2TextWeightsF32 load_eg2_text_weights_f32(const std::filesystem::path& model_dir) {
  Eg2TextWeightsF32 w;
  w.config = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_text_tensors(model, w.config);

  w.embed_tokens = read_tensor_f32(model, "language_model.embed_tokens.weight");
  w.per_layer_model_projection = read_tensor_f32(model, "language_model.ple.per_layer_model_projection.weight");
  w.per_layer_projection_norm = read_tensor_f32(model, "language_model.ple.per_layer_projection_norm.weight");
  w.norm = read_tensor_f32(model, "language_model.norm.weight");
  w.embedding_projection = read_tensor_f32(model, "language_model.embedding_projection.weight");

  w.layers.resize(static_cast<std::size_t>(w.config.num_hidden_layers));
  for (int i = 0; i < w.config.num_hidden_layers; ++i) {
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.input_layernorm = read_tensor_f32(model, p + "input_layernorm.weight");
    lw.q_proj = read_tensor_f32(model, p + "self_attn.q_proj.weight");
    lw.k_proj = read_tensor_f32(model, p + "self_attn.k_proj.weight");
    lw.v_proj = read_tensor_f32(model, p + "self_attn.v_proj.weight");
    lw.o_proj = read_tensor_f32(model, p + "self_attn.o_proj.weight");
    lw.q_norm = read_tensor_f32(model, p + "self_attn.q_norm.weight");
    lw.k_norm = read_tensor_f32(model, p + "self_attn.k_norm.weight");
    lw.post_attention_layernorm = read_tensor_f32(model, p + "post_attention_layernorm.weight");
    lw.pre_feedforward_layernorm = read_tensor_f32(model, p + "pre_feedforward_layernorm.weight");
    lw.gate_proj = read_tensor_f32(model, p + "mlp.gate_proj.weight");
    lw.up_proj = read_tensor_f32(model, p + "mlp.up_proj.weight");
    lw.down_proj = read_tensor_f32(model, p + "mlp.down_proj.weight");
    lw.post_feedforward_layernorm = read_tensor_f32(model, p + "post_feedforward_layernorm.weight");
    lw.per_layer_input_gate = read_tensor_f32(model, p + "ple_block.per_layer_input_gate.weight");
    lw.per_layer_projection = read_tensor_f32(model, p + "ple_block.per_layer_projection.weight");
    lw.post_per_layer_input_norm = read_tensor_f32(model, p + "ple_block.post_per_layer_input_norm.weight");
    lw.layer_scalar = read_scalar_f32(model, p + "layer_scalar");
  }
  return w;
}

Eg2TextWeightsBf16 load_eg2_text_weights_bf16(const std::filesystem::path& model_dir) {
  Eg2TextWeightsBf16 w;
  w.config = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_text_tensors(model, w.config);

  w.embed_tokens = read_tensor_bf16(model, "language_model.embed_tokens.weight");
  w.per_layer_model_projection = read_tensor_bf16(model, "language_model.ple.per_layer_model_projection.weight");
  w.per_layer_projection_norm = read_tensor_bf16(model, "language_model.ple.per_layer_projection_norm.weight");
  w.norm = read_tensor_bf16(model, "language_model.norm.weight");
  w.embedding_projection = read_tensor_bf16(model, "language_model.embedding_projection.weight");

  w.layers.resize(static_cast<std::size_t>(w.config.num_hidden_layers));
  for (int i = 0; i < w.config.num_hidden_layers; ++i) {
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.input_layernorm = read_tensor_bf16(model, p + "input_layernorm.weight");
    lw.q_proj = read_tensor_bf16(model, p + "self_attn.q_proj.weight");
    lw.k_proj = read_tensor_bf16(model, p + "self_attn.k_proj.weight");
    lw.v_proj = read_tensor_bf16(model, p + "self_attn.v_proj.weight");
    lw.o_proj = read_tensor_bf16(model, p + "self_attn.o_proj.weight");
    lw.q_norm = read_tensor_bf16(model, p + "self_attn.q_norm.weight");
    lw.k_norm = read_tensor_bf16(model, p + "self_attn.k_norm.weight");
    lw.post_attention_layernorm = read_tensor_bf16(model, p + "post_attention_layernorm.weight");
    lw.pre_feedforward_layernorm = read_tensor_bf16(model, p + "pre_feedforward_layernorm.weight");
    lw.gate_proj = read_tensor_bf16(model, p + "mlp.gate_proj.weight");
    lw.up_proj = read_tensor_bf16(model, p + "mlp.up_proj.weight");
    lw.down_proj = read_tensor_bf16(model, p + "mlp.down_proj.weight");
    lw.post_feedforward_layernorm = read_tensor_bf16(model, p + "post_feedforward_layernorm.weight");
    lw.per_layer_input_gate = read_tensor_bf16(model, p + "ple_block.per_layer_input_gate.weight");
    lw.per_layer_projection = read_tensor_bf16(model, p + "ple_block.per_layer_projection.weight");
    lw.post_per_layer_input_norm = read_tensor_bf16(model, p + "ple_block.post_per_layer_input_norm.weight");
    lw.layer_scalar = read_scalar_f32(model, p + "layer_scalar");
  }
  return w;
}

namespace {

void validate_vision_tensors(const SafetensorsModel& model, const Eg2Config& cfg) {
  if (!cfg.has_vision_config) {
    throw std::invalid_argument("EmbeddingGemma2: model config has no vision_config");
  }
  const Eg2VisionConfig& vc = cfg.vision;
  if (vc.use_clipped_linears || vc.standardize) {
    throw std::invalid_argument("EmbeddingGemma2: clipped vision linears / standardize are not supported");
  }
  const std::int64_t VH = vc.hidden_size;
  const std::int64_t VI = vc.intermediate_size;
  const std::int64_t VHD = vc.head_dim;
  const std::int64_t VQ = static_cast<std::int64_t>(vc.num_attention_heads) * VHD;
  const std::int64_t VKV = static_cast<std::int64_t>(vc.num_key_value_heads) * VHD;
  const std::int64_t patch_in = 3 * static_cast<std::int64_t>(vc.patch_size) * vc.patch_size;

  require_tensor_shape_and_dtype(model, "vision_tower.patch_embedder.input_proj.weight", {VH, patch_in});
  require_tensor_shape_and_dtype(model, "vision_tower.patch_embedder.position_embedding_table",
                                 {2, vc.position_embedding_size, VH});
  require_tensor_shape_and_dtype(model, "embed_vision.embedding_projection.weight", {cfg.hidden_size, VH});

  for (int i = 0; i < vc.num_hidden_layers; ++i) {
    const std::string p = "vision_tower.encoder.layers." + std::to_string(i) + ".";
    require_tensor_shape_and_dtype(model, p + "input_layernorm.weight", {VH});
    require_tensor_shape_and_dtype(model, p + "self_attn.q_proj.linear.weight", {VQ, VH});
    require_tensor_shape_and_dtype(model, p + "self_attn.k_proj.linear.weight", {VKV, VH});
    require_tensor_shape_and_dtype(model, p + "self_attn.v_proj.linear.weight", {VKV, VH});
    require_tensor_shape_and_dtype(model, p + "self_attn.o_proj.linear.weight", {VH, VQ});
    require_tensor_shape_and_dtype(model, p + "self_attn.q_norm.weight", {VHD});
    require_tensor_shape_and_dtype(model, p + "self_attn.k_norm.weight", {VHD});
    require_tensor_shape_and_dtype(model, p + "post_attention_layernorm.weight", {VH});
    require_tensor_shape_and_dtype(model, p + "pre_feedforward_layernorm.weight", {VH});
    require_tensor_shape_and_dtype(model, p + "mlp.gate_proj.linear.weight", {VI, VH});
    require_tensor_shape_and_dtype(model, p + "mlp.up_proj.linear.weight", {VI, VH});
    require_tensor_shape_and_dtype(model, p + "mlp.down_proj.linear.weight", {VH, VI});
    require_tensor_shape_and_dtype(model, p + "post_feedforward_layernorm.weight", {VH});
  }
}

}  // namespace

Eg2VisionWeightsF32 load_eg2_vision_weights_f32(const std::filesystem::path& model_dir) {
  const Eg2Config cfg = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_vision_tensors(model, cfg);

  Eg2VisionWeightsF32 w;
  w.config = cfg.vision;
  w.text_hidden_size = cfg.hidden_size;
  w.patch_input_proj = read_tensor_f32(model, "vision_tower.patch_embedder.input_proj.weight");
  w.position_embedding_table = read_tensor_f32(model, "vision_tower.patch_embedder.position_embedding_table");
  w.embed_vision_projection = read_tensor_f32(model, "embed_vision.embedding_projection.weight");
  w.layers.resize(static_cast<std::size_t>(cfg.vision.num_hidden_layers));
  for (int i = 0; i < cfg.vision.num_hidden_layers; ++i) {
    const std::string p = "vision_tower.encoder.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.input_layernorm = read_tensor_f32(model, p + "input_layernorm.weight");
    lw.q_proj = read_tensor_f32(model, p + "self_attn.q_proj.linear.weight");
    lw.k_proj = read_tensor_f32(model, p + "self_attn.k_proj.linear.weight");
    lw.v_proj = read_tensor_f32(model, p + "self_attn.v_proj.linear.weight");
    lw.o_proj = read_tensor_f32(model, p + "self_attn.o_proj.linear.weight");
    lw.q_norm = read_tensor_f32(model, p + "self_attn.q_norm.weight");
    lw.k_norm = read_tensor_f32(model, p + "self_attn.k_norm.weight");
    lw.post_attention_layernorm = read_tensor_f32(model, p + "post_attention_layernorm.weight");
    lw.pre_feedforward_layernorm = read_tensor_f32(model, p + "pre_feedforward_layernorm.weight");
    lw.gate_proj = read_tensor_f32(model, p + "mlp.gate_proj.linear.weight");
    lw.up_proj = read_tensor_f32(model, p + "mlp.up_proj.linear.weight");
    lw.down_proj = read_tensor_f32(model, p + "mlp.down_proj.linear.weight");
    lw.post_feedforward_layernorm = read_tensor_f32(model, p + "post_feedforward_layernorm.weight");
  }
  return w;
}

Eg2VisionWeightsBf16 load_eg2_vision_weights_bf16(const std::filesystem::path& model_dir) {
  const Eg2Config cfg = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_vision_tensors(model, cfg);

  Eg2VisionWeightsBf16 w;
  w.config = cfg.vision;
  w.text_hidden_size = cfg.hidden_size;
  w.patch_input_proj = read_tensor_bf16(model, "vision_tower.patch_embedder.input_proj.weight");
  w.position_embedding_table = read_tensor_bf16(model, "vision_tower.patch_embedder.position_embedding_table");
  w.embed_vision_projection = read_tensor_bf16(model, "embed_vision.embedding_projection.weight");
  w.layers.resize(static_cast<std::size_t>(cfg.vision.num_hidden_layers));
  for (int i = 0; i < cfg.vision.num_hidden_layers; ++i) {
    const std::string p = "vision_tower.encoder.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.input_layernorm = read_tensor_bf16(model, p + "input_layernorm.weight");
    lw.q_proj = read_tensor_bf16(model, p + "self_attn.q_proj.linear.weight");
    lw.k_proj = read_tensor_bf16(model, p + "self_attn.k_proj.linear.weight");
    lw.v_proj = read_tensor_bf16(model, p + "self_attn.v_proj.linear.weight");
    lw.o_proj = read_tensor_bf16(model, p + "self_attn.o_proj.linear.weight");
    lw.q_norm = read_tensor_bf16(model, p + "self_attn.q_norm.weight");
    lw.k_norm = read_tensor_bf16(model, p + "self_attn.k_norm.weight");
    lw.post_attention_layernorm = read_tensor_bf16(model, p + "post_attention_layernorm.weight");
    lw.pre_feedforward_layernorm = read_tensor_bf16(model, p + "pre_feedforward_layernorm.weight");
    lw.gate_proj = read_tensor_bf16(model, p + "mlp.gate_proj.linear.weight");
    lw.up_proj = read_tensor_bf16(model, p + "mlp.up_proj.linear.weight");
    lw.down_proj = read_tensor_bf16(model, p + "mlp.down_proj.linear.weight");
    lw.post_feedforward_layernorm = read_tensor_bf16(model, p + "post_feedforward_layernorm.weight");
  }
  return w;
}

namespace {

void require_scalar_tensor(const SafetensorsModel& model, std::string_view name) {
  const SafetensorsRef ref = model.find(name);
  if (!ref) {
    throw std::invalid_argument("EmbeddingGemma2 safetensors missing required scalar tensor: " + std::string(name));
  }
  if (ref->dtype != "BF16" && ref->dtype != "F32") {
    throw std::invalid_argument("EmbeddingGemma2 tensor '" + std::string(name) + "' has unsupported dtype '" +
                                ref->dtype + "' (only BF16/F32 permitted; FP16 is forbidden)");
  }
  if (ref->numel() != 1) {
    throw std::invalid_argument("EmbeddingGemma2 scalar tensor '" + std::string(name) + "' expected 1 element");
  }
}

void validate_clipped_linear(const SafetensorsModel& model, const std::string& prefix, std::int64_t out_dim,
                             std::int64_t in_dim, bool clipped) {
  require_tensor_shape_and_dtype(model, prefix + ".linear.weight", {out_dim, in_dim});
  if (clipped) {
    require_scalar_tensor(model, prefix + ".input_min");
    require_scalar_tensor(model, prefix + ".input_max");
    require_scalar_tensor(model, prefix + ".output_min");
    require_scalar_tensor(model, prefix + ".output_max");
  }
}

void validate_audio_tensors(const SafetensorsModel& model, const Eg2Config& cfg) {
  if (!cfg.has_audio_config) {
    throw std::invalid_argument("EmbeddingGemma2: model config has no audio_config");
  }
  const Eg2AudioConfig& ac = cfg.audio;
  const std::int64_t AH = ac.hidden_size;
  const std::int64_t AHD = AH / ac.num_attention_heads;
  const std::int64_t C0 = ac.sscp_conv_channels_0;
  const std::int64_t C1 = ac.sscp_conv_channels_1;
  const std::int64_t sscp_in = (128 / 4) * C1;
  const std::int64_t OP = ac.output_proj_dims;
  const std::int64_t K = ac.conv_kernel_size;

  require_tensor_shape_and_dtype(model, "audio_tower.subsample_conv_projection.layer0.conv.weight", {C0, 1, 3, 3});
  require_tensor_shape_and_dtype(model, "audio_tower.subsample_conv_projection.layer0.norm.weight", {C0});
  require_tensor_shape_and_dtype(model, "audio_tower.subsample_conv_projection.layer1.conv.weight", {C1, C0, 3, 3});
  require_tensor_shape_and_dtype(model, "audio_tower.subsample_conv_projection.layer1.norm.weight", {C1});
  require_tensor_shape_and_dtype(model, "audio_tower.subsample_conv_projection.input_proj_linear.weight", {AH, sscp_in});
  require_tensor_shape_and_dtype(model, "audio_tower.output_proj.weight", {OP, AH});
  require_tensor_shape_and_dtype(model, "audio_tower.output_proj.bias", {OP});
  require_tensor_shape_and_dtype(model, "embed_audio.embedding_projection.weight", {cfg.hidden_size, OP});

  for (int i = 0; i < ac.num_hidden_layers; ++i) {
    const std::string p = "audio_tower.layers." + std::to_string(i) + ".";
    require_tensor_shape_and_dtype(model, p + "feed_forward1.pre_layer_norm.weight", {AH});
    validate_clipped_linear(model, p + "feed_forward1.ffw_layer_1", 4 * AH, AH, ac.use_clipped_linears);
    validate_clipped_linear(model, p + "feed_forward1.ffw_layer_2", AH, 4 * AH, ac.use_clipped_linears);
    require_tensor_shape_and_dtype(model, p + "feed_forward1.post_layer_norm.weight", {AH});

    require_tensor_shape_and_dtype(model, p + "norm_pre_attn.weight", {AH});
    validate_clipped_linear(model, p + "self_attn.q_proj", AH, AH, ac.use_clipped_linears);
    validate_clipped_linear(model, p + "self_attn.k_proj", AH, AH, ac.use_clipped_linears);
    validate_clipped_linear(model, p + "self_attn.v_proj", AH, AH, ac.use_clipped_linears);
    require_tensor_shape_and_dtype(model, p + "self_attn.relative_k_proj.weight", {AH, AH});
    require_tensor_shape_and_dtype(model, p + "self_attn.per_dim_scale", {AHD});
    validate_clipped_linear(model, p + "self_attn.post", AH, AH, ac.use_clipped_linears);
    require_tensor_shape_and_dtype(model, p + "norm_post_attn.weight", {AH});

    require_tensor_shape_and_dtype(model, p + "lconv1d.pre_layer_norm.weight", {AH});
    validate_clipped_linear(model, p + "lconv1d.linear_start", 2 * AH, AH, ac.use_clipped_linears);
    require_tensor_shape_and_dtype(model, p + "lconv1d.depthwise_conv1d.weight", {AH, 1, K});
    require_tensor_shape_and_dtype(model, p + "lconv1d.conv_norm.weight", {AH});
    validate_clipped_linear(model, p + "lconv1d.linear_end", AH, AH, ac.use_clipped_linears);

    require_tensor_shape_and_dtype(model, p + "feed_forward2.pre_layer_norm.weight", {AH});
    validate_clipped_linear(model, p + "feed_forward2.ffw_layer_1", 4 * AH, AH, ac.use_clipped_linears);
    validate_clipped_linear(model, p + "feed_forward2.ffw_layer_2", AH, 4 * AH, ac.use_clipped_linears);
    require_tensor_shape_and_dtype(model, p + "feed_forward2.post_layer_norm.weight", {AH});

    require_tensor_shape_and_dtype(model, p + "norm_out.weight", {AH});
  }
}

Eg2ClippedLinearF32 read_clipped_f32(const SafetensorsModel& model, const std::string& prefix, bool clipped) {
  Eg2ClippedLinearF32 c;
  c.weight = read_tensor_f32(model, prefix + ".linear.weight");
  if (clipped) {
    c.input_min = read_scalar_f32(model, prefix + ".input_min");
    c.input_max = read_scalar_f32(model, prefix + ".input_max");
    c.output_min = read_scalar_f32(model, prefix + ".output_min");
    c.output_max = read_scalar_f32(model, prefix + ".output_max");
  }
  return c;
}

Eg2ClippedLinearBf16 read_clipped_bf16(const SafetensorsModel& model, const std::string& prefix, bool clipped) {
  Eg2ClippedLinearBf16 c;
  c.weight = read_tensor_bf16(model, prefix + ".linear.weight");
  if (clipped) {
    c.input_min = read_scalar_f32(model, prefix + ".input_min");
    c.input_max = read_scalar_f32(model, prefix + ".input_max");
    c.output_min = read_scalar_f32(model, prefix + ".output_min");
    c.output_max = read_scalar_f32(model, prefix + ".output_max");
  }
  return c;
}

}  // namespace

Eg2AudioWeightsF32 load_eg2_audio_weights_f32(const std::filesystem::path& model_dir) {
  const Eg2Config cfg = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_audio_tensors(model, cfg);

  Eg2AudioWeightsF32 w;
  w.config = cfg.audio;
  w.text_hidden_size = cfg.hidden_size;
  w.sscp_layer0_conv = read_tensor_f32(model, "audio_tower.subsample_conv_projection.layer0.conv.weight");
  w.sscp_layer0_norm = read_tensor_f32(model, "audio_tower.subsample_conv_projection.layer0.norm.weight");
  w.sscp_layer1_conv = read_tensor_f32(model, "audio_tower.subsample_conv_projection.layer1.conv.weight");
  w.sscp_layer1_norm = read_tensor_f32(model, "audio_tower.subsample_conv_projection.layer1.norm.weight");
  w.sscp_input_proj_linear = read_tensor_f32(model, "audio_tower.subsample_conv_projection.input_proj_linear.weight");
  w.output_proj_weight = read_tensor_f32(model, "audio_tower.output_proj.weight");
  w.output_proj_bias = read_tensor_f32(model, "audio_tower.output_proj.bias");
  w.embed_audio_projection = read_tensor_f32(model, "embed_audio.embedding_projection.weight");

  const bool cl = cfg.audio.use_clipped_linears;
  w.layers.resize(static_cast<std::size_t>(cfg.audio.num_hidden_layers));
  for (int i = 0; i < cfg.audio.num_hidden_layers; ++i) {
    const std::string p = "audio_tower.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.feed_forward1.pre_layer_norm = read_tensor_f32(model, p + "feed_forward1.pre_layer_norm.weight");
    lw.feed_forward1.ffw_layer_1 = read_clipped_f32(model, p + "feed_forward1.ffw_layer_1", cl);
    lw.feed_forward1.ffw_layer_2 = read_clipped_f32(model, p + "feed_forward1.ffw_layer_2", cl);
    lw.feed_forward1.post_layer_norm = read_tensor_f32(model, p + "feed_forward1.post_layer_norm.weight");

    lw.norm_pre_attn = read_tensor_f32(model, p + "norm_pre_attn.weight");
    lw.q_proj = read_clipped_f32(model, p + "self_attn.q_proj", cl);
    lw.k_proj = read_clipped_f32(model, p + "self_attn.k_proj", cl);
    lw.v_proj = read_clipped_f32(model, p + "self_attn.v_proj", cl);
    lw.relative_k_proj = read_tensor_f32(model, p + "self_attn.relative_k_proj.weight");
    lw.per_dim_scale = read_tensor_f32(model, p + "self_attn.per_dim_scale");
    lw.post = read_clipped_f32(model, p + "self_attn.post", cl);
    lw.norm_post_attn = read_tensor_f32(model, p + "norm_post_attn.weight");

    lw.lconv_pre_layer_norm = read_tensor_f32(model, p + "lconv1d.pre_layer_norm.weight");
    lw.lconv_linear_start = read_clipped_f32(model, p + "lconv1d.linear_start", cl);
    lw.lconv_depthwise_conv1d = read_tensor_f32(model, p + "lconv1d.depthwise_conv1d.weight");
    lw.lconv_conv_norm = read_tensor_f32(model, p + "lconv1d.conv_norm.weight");
    lw.lconv_linear_end = read_clipped_f32(model, p + "lconv1d.linear_end", cl);

    lw.feed_forward2.pre_layer_norm = read_tensor_f32(model, p + "feed_forward2.pre_layer_norm.weight");
    lw.feed_forward2.ffw_layer_1 = read_clipped_f32(model, p + "feed_forward2.ffw_layer_1", cl);
    lw.feed_forward2.ffw_layer_2 = read_clipped_f32(model, p + "feed_forward2.ffw_layer_2", cl);
    lw.feed_forward2.post_layer_norm = read_tensor_f32(model, p + "feed_forward2.post_layer_norm.weight");

    lw.norm_out = read_tensor_f32(model, p + "norm_out.weight");
  }
  return w;
}

Eg2AudioWeightsBf16 load_eg2_audio_weights_bf16(const std::filesystem::path& model_dir) {
  const Eg2Config cfg = Eg2Config::load(model_dir);
  SafetensorsModel model(model_dir, SafetensorsOpenMode::SafetensorsOnly);
  validate_audio_tensors(model, cfg);

  Eg2AudioWeightsBf16 w;
  w.config = cfg.audio;
  w.text_hidden_size = cfg.hidden_size;
  w.sscp_layer0_conv = read_tensor_bf16(model, "audio_tower.subsample_conv_projection.layer0.conv.weight");
  w.sscp_layer0_norm = read_tensor_bf16(model, "audio_tower.subsample_conv_projection.layer0.norm.weight");
  w.sscp_layer1_conv = read_tensor_bf16(model, "audio_tower.subsample_conv_projection.layer1.conv.weight");
  w.sscp_layer1_norm = read_tensor_bf16(model, "audio_tower.subsample_conv_projection.layer1.norm.weight");
  w.sscp_input_proj_linear = read_tensor_bf16(model, "audio_tower.subsample_conv_projection.input_proj_linear.weight");
  w.output_proj_weight = read_tensor_bf16(model, "audio_tower.output_proj.weight");
  w.output_proj_bias = read_tensor_bf16(model, "audio_tower.output_proj.bias");
  w.embed_audio_projection = read_tensor_bf16(model, "embed_audio.embedding_projection.weight");

  const bool cl = cfg.audio.use_clipped_linears;
  w.layers.resize(static_cast<std::size_t>(cfg.audio.num_hidden_layers));
  for (int i = 0; i < cfg.audio.num_hidden_layers; ++i) {
    const std::string p = "audio_tower.layers." + std::to_string(i) + ".";
    auto& lw = w.layers[static_cast<std::size_t>(i)];
    lw.feed_forward1.pre_layer_norm = read_tensor_bf16(model, p + "feed_forward1.pre_layer_norm.weight");
    lw.feed_forward1.ffw_layer_1 = read_clipped_bf16(model, p + "feed_forward1.ffw_layer_1", cl);
    lw.feed_forward1.ffw_layer_2 = read_clipped_bf16(model, p + "feed_forward1.ffw_layer_2", cl);
    lw.feed_forward1.post_layer_norm = read_tensor_bf16(model, p + "feed_forward1.post_layer_norm.weight");

    lw.norm_pre_attn = read_tensor_bf16(model, p + "norm_pre_attn.weight");
    lw.q_proj = read_clipped_bf16(model, p + "self_attn.q_proj", cl);
    lw.k_proj = read_clipped_bf16(model, p + "self_attn.k_proj", cl);
    lw.v_proj = read_clipped_bf16(model, p + "self_attn.v_proj", cl);
    lw.relative_k_proj = read_tensor_bf16(model, p + "self_attn.relative_k_proj.weight");
    lw.per_dim_scale = read_tensor_f32(model, p + "self_attn.per_dim_scale");
    lw.post = read_clipped_bf16(model, p + "self_attn.post", cl);
    lw.norm_post_attn = read_tensor_bf16(model, p + "norm_post_attn.weight");

    lw.lconv_pre_layer_norm = read_tensor_bf16(model, p + "lconv1d.pre_layer_norm.weight");
    lw.lconv_linear_start = read_clipped_bf16(model, p + "lconv1d.linear_start", cl);
    lw.lconv_depthwise_conv1d = read_tensor_bf16(model, p + "lconv1d.depthwise_conv1d.weight");
    lw.lconv_conv_norm = read_tensor_bf16(model, p + "lconv1d.conv_norm.weight");
    lw.lconv_linear_end = read_clipped_bf16(model, p + "lconv1d.linear_end", cl);

    lw.feed_forward2.pre_layer_norm = read_tensor_bf16(model, p + "feed_forward2.pre_layer_norm.weight");
    lw.feed_forward2.ffw_layer_1 = read_clipped_bf16(model, p + "feed_forward2.ffw_layer_1", cl);
    lw.feed_forward2.ffw_layer_2 = read_clipped_bf16(model, p + "feed_forward2.ffw_layer_2", cl);
    lw.feed_forward2.post_layer_norm = read_tensor_bf16(model, p + "feed_forward2.post_layer_norm.weight");

    lw.norm_out = read_tensor_bf16(model, p + "norm_out.weight");
  }
  return w;
}

}  // namespace ingine::backends::embedding_gemma2
