#include "backends/embedding_gemma2/eg2_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::backends::embedding_gemma2 {
namespace {

constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

inline float gelu_pytorch_tanh(float x) noexcept {
  const float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// Direct multiplicative RMSNorm (EmbeddingGemma2RMSNorm: NO +1.0 offset).
// When weight == nullptr, runs unscaled RMSNorm (with_scale=False, used for v_norm).
void rms_norm_rows(const float* x, const float* weight, int rows, int dim, float eps, float* out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = static_cast<double>(xr[d]);
      sum_sq += v * v;
    }
    const float inv_rms = static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + static_cast<double>(eps)));
    if (weight != nullptr) {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv_rms * weight[d];
    } else {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv_rms;
    }
  }
}

// Y[rows, out_dim] = X[rows, in_dim] @ W[out_dim, in_dim]^T  (with optional scalar post-scale).
void linear_f32(const float* x, const float* w, int rows, int in_dim, int out_dim, float* out, float scale = 1.0f) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim);
    for (int o = 0; o < out_dim; ++o) {
      const float* wo = w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim);
      float acc = 0.0f;
      for (int k = 0; k < in_dim; ++k) {
        acc += xr[k] * wo[k];
      }
      yr[o] = acc * scale;
    }
  }
}

// Applies standard rotate_half RoPE in-place to tensor of shape [T, n_heads, head_dim].
void apply_rope_inplace(float* x, int T, int n_heads, int head_dim, double rope_theta) {
  const int half = head_dim / 2;
  std::vector<double> inv_freq(static_cast<std::size_t>(half));
  for (int j = 0; j < half; ++j) {
    inv_freq[static_cast<std::size_t>(j)] = 1.0 / std::pow(rope_theta, (2.0 * static_cast<double>(j)) / static_cast<double>(head_dim));
  }
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < half; ++j) {
      const double angle = static_cast<double>(t) * inv_freq[static_cast<std::size_t>(j)];
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      for (int h = 0; h < n_heads; ++h) {
        float* ptr = x + (static_cast<std::size_t>(t) * static_cast<std::size_t>(n_heads) + static_cast<std::size_t>(h)) *
                             static_cast<std::size_t>(head_dim);
        const float x1 = ptr[j];
        const float x2 = ptr[j + half];
        ptr[j] = x1 * c - x2 * s;
        ptr[j + half] = x2 * c + x1 * s;
      }
    }
  }
}

void validate_dimensions(const Eg2Config& cfg, std::optional<int> dimensions, int base_dim) {
  if (!dimensions.has_value()) return;
  const int d = *dimensions;
  if (d <= 0 || d > base_dim) {
    throw std::invalid_argument("EmbeddingGemma2: requested dimensions " + std::to_string(d) +
                                " out of range [1, " + std::to_string(base_dim) + "]");
  }
  if (base_dim == cfg.embedding_dim && !cfg.matryoshka_dims.empty()) {
    if (std::find(cfg.matryoshka_dims.begin(), cfg.matryoshka_dims.end(), d) == cfg.matryoshka_dims.end()) {
      throw std::invalid_argument("EmbeddingGemma2: requested dimensions " + std::to_string(d) +
                                  " is not in supported Matryoshka dims {768, 512, 256, 128}");
    }
  }
}

}  // namespace

Eg2CpuEncoder::Eg2CpuEncoder(Eg2TextWeightsF32 weights) : weights_(std::move(weights)) {
  weights_.config.validate();
}

std::vector<float> Eg2CpuEncoder::encode_hidden(std::span<const std::int32_t> token_ids,
                                                std::optional<int> tap_layer,
                                                std::span<const float> vision_soft_tokens,
                                                std::span<const float> audio_soft_tokens) const {
  const Eg2Config& cfg = weights_.config;
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("Eg2CpuEncoder::encode_hidden: token_ids must not be empty");
  if (tap_layer.has_value() && (*tap_layer < 0 || *tap_layer >= cfg.num_hidden_layers)) {
    throw std::invalid_argument("Eg2CpuEncoder::encode_hidden: tap_layer out of range");
  }

  const int H = cfg.hidden_size;
  const int I = cfg.intermediate_size;
  const int L = cfg.num_hidden_layers;
  const float embed_scale = std::sqrt(static_cast<float>(H));
  const float ple_scale = 1.0f / std::sqrt(static_cast<float>(H));

  if ((vision_soft_tokens.size() % static_cast<std::size_t>(H)) != 0) {
    throw std::invalid_argument("Eg2CpuEncoder::encode_hidden: vision_soft_tokens length not divisible by hidden_size");
  }
  if ((audio_soft_tokens.size() % static_cast<std::size_t>(H)) != 0) {
    throw std::invalid_argument("Eg2CpuEncoder::encode_hidden: audio_soft_tokens length not divisible by hidden_size");
  }
  const std::size_t n_vision_soft = vision_soft_tokens.size() / static_cast<std::size_t>(H);
  const std::size_t n_audio_soft = audio_soft_tokens.size() / static_cast<std::size_t>(H);

  // 1. Scaled word embeddings + multimodal soft-token scatter onto <|image|>/<|video|>/<|audio|> BEFORE PLE.
  std::vector<float> hidden(static_cast<std::size_t>(T) * static_cast<std::size_t>(H));
  std::size_t vision_cursor = 0;
  std::size_t audio_cursor = 0;
  for (int t = 0; t < T; ++t) {
    const std::int32_t tid = token_ids[static_cast<std::size_t>(t)];
    if (tid < 0 || tid >= cfg.vocab_size) {
      throw std::invalid_argument("Eg2CpuEncoder: token_id " + std::to_string(tid) + " out of vocab range [0, " +
                                  std::to_string(cfg.vocab_size - 1) + "]");
    }
    float* dst = hidden.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(H);
    if (tid == cfg.image_token_id || tid == cfg.video_token_id) {
      if (vision_cursor >= n_vision_soft) {
        throw std::invalid_argument("Eg2CpuEncoder: more image/video placeholder tokens than vision soft tokens");
      }
      const float* vsrc = vision_soft_tokens.data() + vision_cursor * static_cast<std::size_t>(H);
      std::memcpy(dst, vsrc, static_cast<std::size_t>(H) * sizeof(float));
      ++vision_cursor;
    } else if (tid == cfg.audio_token_id) {
      if (audio_cursor >= n_audio_soft) {
        throw std::invalid_argument("Eg2CpuEncoder: more audio placeholder tokens than audio soft tokens");
      }
      const float* asrc = audio_soft_tokens.data() + audio_cursor * static_cast<std::size_t>(H);
      std::memcpy(dst, asrc, static_cast<std::size_t>(H) * sizeof(float));
      ++audio_cursor;
    } else {
      const float* src = weights_.embed_tokens.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(H);
      for (int d = 0; d < H; ++d) dst[d] = src[d] * embed_scale;
    }
  }
  if (vision_cursor != n_vision_soft) {
    throw std::invalid_argument("Eg2CpuEncoder: vision soft token count (" + std::to_string(n_vision_soft) +
                                ") does not match image/video placeholder tokens (" + std::to_string(vision_cursor) + ")");
  }
  if (audio_cursor != n_audio_soft) {
    throw std::invalid_argument("Eg2CpuEncoder: audio soft token count (" + std::to_string(n_audio_soft) +
                                ") does not match audio placeholder tokens (" + std::to_string(audio_cursor) + ")");
  }

  // 2. Projection-only PLE: [T, L * H] -> RMSNorm across each [H] slice
  const int stop_layer = tap_layer.value_or(L - 1);
  std::vector<float> ple(static_cast<std::size_t>(T) * static_cast<std::size_t>(L) * static_cast<std::size_t>(H));
  linear_f32(hidden.data(), weights_.per_layer_model_projection.data(), T, H, L * H, ple.data(), ple_scale);
  rms_norm_rows(ple.data(), weights_.per_layer_projection_norm.data(), T * L, H, cfg.rms_norm_eps, ple.data());

  // Scratch buffers reused across layers
  std::vector<float> norm_buf(static_cast<std::size_t>(T) * static_cast<std::size_t>(H));
  std::vector<float> sub_out(static_cast<std::size_t>(T) * static_cast<std::size_t>(H));
  std::vector<float> gate_buf(static_cast<std::size_t>(T) * static_cast<std::size_t>(I));
  std::vector<float> up_buf(static_cast<std::size_t>(T) * static_cast<std::size_t>(I));
  std::vector<float> ple_gate(static_cast<std::size_t>(T) * static_cast<std::size_t>(H));
  std::vector<float> q_buf, k_buf, v_buf, attn_ctx;
  std::vector<float> scores(static_cast<std::size_t>(T));

  for (int li = 0; li <= stop_layer; ++li) {
    const LayerSpec& ls = cfg.layers[static_cast<std::size_t>(li)];
    const Eg2LayerWeightsF32& lw = weights_.layers[static_cast<std::size_t>(li)];
    const int n_heads = ls.num_attention_heads;
    const int n_kv = ls.num_key_value_heads;
    const int hd = ls.head_dim;
    const int q_dim = n_heads * hd;
    const int kv_dim = n_kv * hd;
    const int kv_groups = n_heads / n_kv;
    const bool is_sliding = (ls.kind == LayerAttnKind::Sliding);

    // ---- Attention sub-block ----
    rms_norm_rows(hidden.data(), lw.input_layernorm.data(), T, H, cfg.rms_norm_eps, norm_buf.data());

    q_buf.resize(static_cast<std::size_t>(T) * static_cast<std::size_t>(q_dim));
    k_buf.resize(static_cast<std::size_t>(T) * static_cast<std::size_t>(kv_dim));
    v_buf.resize(static_cast<std::size_t>(T) * static_cast<std::size_t>(kv_dim));
    attn_ctx.resize(static_cast<std::size_t>(T) * static_cast<std::size_t>(q_dim));

    linear_f32(norm_buf.data(), lw.q_proj.data(), T, H, q_dim, q_buf.data());
    linear_f32(norm_buf.data(), lw.k_proj.data(), T, H, kv_dim, k_buf.data());
    linear_f32(norm_buf.data(), lw.v_proj.data(), T, H, kv_dim, v_buf.data());

    rms_norm_rows(q_buf.data(), lw.q_norm.data(), T * n_heads, hd, cfg.rms_norm_eps, q_buf.data());
    rms_norm_rows(k_buf.data(), lw.k_norm.data(), T * n_kv, hd, cfg.rms_norm_eps, k_buf.data());
    rms_norm_rows(v_buf.data(), nullptr, T * n_kv, hd, cfg.rms_norm_eps, v_buf.data());

    apply_rope_inplace(q_buf.data(), T, n_heads, hd, ls.rope_theta);
    apply_rope_inplace(k_buf.data(), T, n_kv, hd, ls.rope_theta);

    // Bidirectional attention with scaling = 1.0; inclusive sliding window |q - k| <= sliding_window.
    for (int h = 0; h < n_heads; ++h) {
      const int kvh = h / kv_groups;
      for (int qi = 0; qi < T; ++qi) {
        const int k_begin = is_sliding ? std::max(0, qi - cfg.sliding_window) : 0;
        const int k_end = is_sliding ? std::min(T - 1, qi + cfg.sliding_window) : (T - 1);
        const float* qp = q_buf.data() + (static_cast<std::size_t>(qi) * n_heads + h) * hd;

        float max_s = -std::numeric_limits<float>::infinity();
        for (int ki = k_begin; ki <= k_end; ++ki) {
          const float* kp = k_buf.data() + (static_cast<std::size_t>(ki) * n_kv + kvh) * hd;
          float dot = 0.0f;
          for (int d = 0; d < hd; ++d) dot += qp[d] * kp[d];
          scores[static_cast<std::size_t>(ki)] = dot;
          if (dot > max_s) max_s = dot;
        }
        double sum_exp = 0.0;
        for (int ki = k_begin; ki <= k_end; ++ki) {
          const double e = std::exp(static_cast<double>(scores[static_cast<std::size_t>(ki)] - max_s));
          scores[static_cast<std::size_t>(ki)] = static_cast<float>(e);
          sum_exp += e;
        }
        const float inv_sum = static_cast<float>(1.0 / sum_exp);
        float* op = attn_ctx.data() + (static_cast<std::size_t>(qi) * n_heads + h) * hd;
        std::fill(op, op + hd, 0.0f);
        for (int ki = k_begin; ki <= k_end; ++ki) {
          const float w_attn = scores[static_cast<std::size_t>(ki)] * inv_sum;
          const float* vp = v_buf.data() + (static_cast<std::size_t>(ki) * n_kv + kvh) * hd;
          for (int d = 0; d < hd; ++d) op[d] += w_attn * vp[d];
        }
      }
    }

    linear_f32(attn_ctx.data(), lw.o_proj.data(), T, q_dim, H, sub_out.data());
    rms_norm_rows(sub_out.data(), lw.post_attention_layernorm.data(), T, H, cfg.rms_norm_eps, sub_out.data());
    for (std::size_t idx = 0; idx < hidden.size(); ++idx) hidden[idx] += sub_out[idx];

    // ---- GeGLU MLP sub-block ----
    rms_norm_rows(hidden.data(), lw.pre_feedforward_layernorm.data(), T, H, cfg.rms_norm_eps, norm_buf.data());
    linear_f32(norm_buf.data(), lw.gate_proj.data(), T, H, I, gate_buf.data());
    linear_f32(norm_buf.data(), lw.up_proj.data(), T, H, I, up_buf.data());
    for (std::size_t idx = 0; idx < gate_buf.size(); ++idx) {
      gate_buf[idx] = gelu_pytorch_tanh(gate_buf[idx]) * up_buf[idx];
    }
    linear_f32(gate_buf.data(), lw.down_proj.data(), T, I, H, sub_out.data());
    rms_norm_rows(sub_out.data(), lw.post_feedforward_layernorm.data(), T, H, cfg.rms_norm_eps, sub_out.data());
    for (std::size_t idx = 0; idx < hidden.size(); ++idx) hidden[idx] += sub_out[idx];

    // ---- PLE residual sub-block ----
    linear_f32(hidden.data(), lw.per_layer_input_gate.data(), T, H, H, ple_gate.data());
    for (int t = 0; t < T; ++t) {
      float* gp = ple_gate.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(H);
      const float* pp =
          ple.data() + (static_cast<std::size_t>(t) * static_cast<std::size_t>(L) + static_cast<std::size_t>(li)) *
                           static_cast<std::size_t>(H);
      for (int d = 0; d < H; ++d) gp[d] = gelu_pytorch_tanh(gp[d]) * pp[d];
    }
    linear_f32(ple_gate.data(), lw.per_layer_projection.data(), T, H, H, sub_out.data());
    rms_norm_rows(sub_out.data(), lw.post_per_layer_input_norm.data(), T, H, cfg.rms_norm_eps, sub_out.data());
    for (std::size_t idx = 0; idx < hidden.size(); ++idx) hidden[idx] += sub_out[idx];

    // ---- Learned per-layer scalar ----
    const float sc = lw.layer_scalar;
    for (float& v : hidden) v *= sc;
  }

  // Final RMSNorm only at full depth (tap_layer == nullopt).
  if (!tap_layer.has_value()) {
    rms_norm_rows(hidden.data(), weights_.norm.data(), T, H, cfg.rms_norm_eps, hidden.data());
  }
  return hidden;
}

ingine::model::EmbeddingResult Eg2CpuEncoder::embed(const ingine::model::EmbeddingRequest& req,
                                                    const PlatosKey* platos_key,
                                                    const Eg2CpuVisionEncoder* vision_encoder,
                                                    const Eg2CpuAudioEncoder* audio_encoder) const {
  const Eg2Config& cfg = weights_.config;
  if (req.inputs.empty()) throw std::invalid_argument("Eg2CpuEncoder::embed: inputs must not be empty");
  if (!req.images.empty() && req.images.size() != req.inputs.size()) {
    throw std::invalid_argument("Eg2CpuEncoder::embed: images batch size must match inputs batch size");
  }
  if (!req.audios.empty() && req.audios.size() != req.inputs.size()) {
    throw std::invalid_argument("Eg2CpuEncoder::embed: audios batch size must match inputs batch size");
  }
  if (req.tap_layer.has_value() && (*req.tap_layer < 0 || *req.tap_layer >= cfg.num_hidden_layers)) {
    throw std::invalid_argument("Eg2CpuEncoder::embed: tap_layer out of range");
  }
  if (req.use_platos_key && platos_key == nullptr) {
    throw std::invalid_argument("Eg2CpuEncoder::embed: platos_key requested but none loaded");
  }

  const auto t0 = std::chrono::steady_clock::now();
  const int B = static_cast<int>(req.inputs.size());
  const int H = cfg.hidden_size;
  const int E = cfg.embedding_dim;
  const bool full_depth = !req.tap_layer.has_value();

  int total_tokens = 0;
  std::vector<float> pooled_h(static_cast<std::size_t>(B) * static_cast<std::size_t>(H), 0.0f);
  for (int b = 0; b < B; ++b) {
    const auto& seq = req.inputs[static_cast<std::size_t>(b)];
    if (seq.empty()) throw std::invalid_argument("Eg2CpuEncoder::embed: input sequence must not be empty");
    total_tokens += static_cast<int>(seq.size());
    std::vector<float> item_vision_soft;
    if (!req.images.empty() && !req.images[static_cast<std::size_t>(b)].empty()) {
      if (vision_encoder == nullptr) {
        throw std::invalid_argument("EmbeddingGemma2: image/video input rejected because vision tower is disabled (--no-vision)");
      }
      for (const auto& img : req.images[static_cast<std::size_t>(b)]) {
        const std::vector<float> soft = vision_encoder->encode_image(img);
        item_vision_soft.insert(item_vision_soft.end(), soft.begin(), soft.end());
      }
    }
    std::vector<float> item_audio_soft;
    if (!req.audios.empty() && !req.audios[static_cast<std::size_t>(b)].empty()) {
      if (audio_encoder == nullptr) {
        throw std::invalid_argument("EmbeddingGemma2: audio input rejected because audio tower is disabled (--no-audio)");
      }
      for (const auto& aud : req.audios[static_cast<std::size_t>(b)]) {
        const std::vector<float> soft = audio_encoder->encode_audio(aud);
        item_audio_soft.insert(item_audio_soft.end(), soft.begin(), soft.end());
      }
    }
    const std::vector<float> h = encode_hidden(seq, req.tap_layer, item_vision_soft, item_audio_soft);
    const int T = static_cast<int>(seq.size());
    float* dst = pooled_h.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(H);
    if (req.pooling == model::EmbeddingPooling::Cls) {
      std::memcpy(dst, h.data(), static_cast<std::size_t>(H) * sizeof(float));
    } else if (req.pooling == model::EmbeddingPooling::LastToken) {
      std::memcpy(dst, h.data() + static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(H),
                  static_cast<std::size_t>(H) * sizeof(float));
    } else {
      for (int d = 0; d < H; ++d) {
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += static_cast<double>(h[static_cast<std::size_t>(t) * H + d]);
        dst[d] = static_cast<float>(acc / static_cast<double>(T));
      }
    }
  }

  // At full depth, project pooled [B, H=512] -> [B, E=768] via embedding_projection.
  // Because embedding_projection is bias-free, mean_t(W @ h_t) == W @ mean_t(h_t).
  int cur_dim = full_depth ? E : H;
  std::vector<float> cur_vecs;
  if (full_depth) {
    cur_vecs.resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(E));
    linear_f32(pooled_h.data(), weights_.embedding_projection.data(), B, H, E, cur_vecs.data());
  } else {
    cur_vecs = std::move(pooled_h);
  }
  const auto t1 = std::chrono::steady_clock::now();

  double pk_seconds = 0.0;
  int out_dim = cur_dim;
  std::vector<float> final_flat;
  if (req.use_platos_key) {
    if (req.dimensions.has_value()) {
      throw std::invalid_argument("EmbeddingGemma2: 'dimensions' cannot be combined with 'platos_key'");
    }
    platos_key->validate_request_tap(req.tap_layer);
    platos_key->validate_request_identity(req.cartridge_id.value_or(""), req.cartridge_sha256.value_or(""));
    if (platos_key->in_dim() != cur_dim) {
      throw std::invalid_argument("EmbeddingGemma2: platos_key d_in (" + std::to_string(platos_key->in_dim()) +
                                  ") does not match embedding input dim (" + std::to_string(cur_dim) + ")");
    }
    const auto tp0 = std::chrono::steady_clock::now();
    out_dim = platos_key->out_dim();
    final_flat.resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(out_dim));
    platos_key->project_cpu(cur_vecs.data(), B, final_flat.data(), req.normalize);
    pk_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
  } else {
    validate_dimensions(cfg, req.dimensions, cur_dim);
    out_dim = req.dimensions.value_or(cur_dim);
    if (out_dim == cur_dim) {
      final_flat = std::move(cur_vecs);
    } else {
      final_flat.resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(out_dim));
      for (int b = 0; b < B; ++b) {
        std::memcpy(final_flat.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim),
                    cur_vecs.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(cur_dim),
                    static_cast<std::size_t>(out_dim) * sizeof(float));
      }
    }
    if (req.normalize) l2_normalize_rows_cpu(final_flat.data(), B, out_dim);
  }

  model::EmbeddingResult res;
  res.embeddings.resize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    const float* row = final_flat.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim);
    res.embeddings[static_cast<std::size_t>(b)].assign(row, row + out_dim);
  }
  res.stats.total_prompt_tokens = total_tokens;
  res.stats.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
  res.stats.platos_key_seconds = pk_seconds;
  res.stats.effective_tap_layer = req.tap_layer.value_or(-1);
  res.stats.output_dim = out_dim;
  res.stats.kv_slots_allocated = 0;
  res.stats.lm_head_skipped = true;
  return res;
}

// ---- Eg2CpuBackend -------------------------------------------------------------------------------------------

Eg2CpuBackend::Eg2CpuBackend(const ingine::model::RuntimeOptions& options)
    : encoder_(load_eg2_text_weights_f32(options.model_dir)),
      max_ctx_(options.max_ctx > 0 ? std::min(options.max_ctx, 8192) : 8192) {
  if (!options.model_dir.filename().empty()) {
    model_id_ = options.model_dir.filename().string();
  }
  if (options.vision && encoder_.config().has_vision_config) {
    vision_encoder_ = std::make_unique<Eg2CpuVisionEncoder>(load_eg2_vision_weights_f32(options.model_dir));
  }
  if (options.audio && encoder_.config().has_audio_config) {
    audio_encoder_ = std::make_unique<Eg2CpuAudioEncoder>(load_eg2_audio_weights_f32(options.model_dir));
  }
  if (!options.platos_key.empty()) {
    platos_key_ = PlatosKey::load(options.platos_key, options.platos_key_metadata, 0);
    const Eg2Config& cfg = encoder_.config();
    const int expected_dim = platos_key_->metadata().expected_tap_layer.has_value()
                                 ? cfg.hidden_size
                                 : (platos_key_->metadata().requires_final_layer ? cfg.embedding_dim : 0);
    if (expected_dim == 0 && platos_key_->in_dim() != cfg.hidden_size && platos_key_->in_dim() != cfg.embedding_dim) {
      throw std::invalid_argument("EmbeddingGemma2: platos_key d_in (" + std::to_string(platos_key_->in_dim()) +
                                  ") matches neither hidden_size (" + std::to_string(cfg.hidden_size) +
                                  ") nor embedding_dim (" + std::to_string(cfg.embedding_dim) + ")");
    }
    platos_key_->validate_binding(model_id_, model_family(), expected_dim, cfg.num_hidden_layers);
  }
  init_metadata();
}

Eg2CpuBackend::Eg2CpuBackend(Eg2TextWeightsF32 weights, int max_ctx, std::shared_ptr<const PlatosKey> platos_key)
    : encoder_(std::move(weights)), platos_key_(std::move(platos_key)), max_ctx_(max_ctx) {
  init_metadata();
}

void Eg2CpuBackend::init_metadata() {
  const Eg2Config& cfg = encoder_.config();
  caps_.family = model::ModelFamily::EmbeddingGemma2;
  caps_.backend_name = "embeddinggemma2-cpu-f32";
  caps_.architecture = !cfg.architectures.empty() ? cfg.architectures.front() : "EmbeddingGemma2Model";
  caps_.supports_text = true;
  caps_.supports_images = (vision_encoder_ != nullptr);
  caps_.supports_audio = (audio_encoder_ != nullptr);
  caps_.max_supported_batch = 64;
  caps_.supported_gears = {"encoder_f32"};
  caps_.supported_kv_cache_modes = {"none"};
  caps_.supported_quant_layouts = {"bf16_to_f32"};
  caps_.media_token_ids = {cfg.boi_token_id, cfg.boa_token_id, cfg.image_token_id,
                           cfg.audio_token_id, cfg.eoi_token_id, cfg.eoa_token_id, cfg.video_token_id};
  caps_.vocab_size = cfg.vocab_size;
  caps_.max_position_embeddings = max_ctx_;

  profile_.family = caps_.family;
  profile_.model_id = model_id_;
  profile_.gear = "encoder_f32";
  profile_.kv_cache_mode = "none";
  profile_.tokenizer_format = "gemma_bpe";
  profile_.chat_template_family = "embedding_gemma2";
  profile_.max_ctx = max_ctx_;
  profile_.max_batch = 64;
  profile_.capabilities = caps_;

  identity_.family = caps_.family;
  identity_.backend_name = caps_.backend_name;
  identity_.state_layout_id = "eg2_stateless_encoder_v1";
  identity_.act_dtype = model::ActivationDtype::Fp32;
  identity_.numeric_abi = "act=f32;weights=bf16_to_f32;attn=bidir_swa_mqa_f32";
  identity_.weight_precision = "f32";
  identity_.kv_format = "none";
  identity_.kv_cache_mode = "none";
  identity_.gear = "encoder_f32";
  identity_.max_ctx = max_ctx_;
  identity_.prefill_chunk = max_ctx_;
}

ingine::model::GenerateStats Eg2CpuBackend::generate(std::span<const std::int32_t>, int,
                                                     const ingine::model::Sampling&, std::span<const std::int32_t>,
                                                     const std::function<bool(std::span<const std::int32_t>)>&) {
  throw std::invalid_argument("EmbeddingGemma2 is an embeddings-only encoder; use POST /v1/embeddings");
}

ingine::model::EmbeddingResult Eg2CpuBackend::embed(const ingine::model::EmbeddingRequest& req) {
  for (const auto& seq : req.inputs) {
    if (static_cast<int>(seq.size()) > max_ctx_) {
      throw std::invalid_argument("EmbeddingGemma2: input length " + std::to_string(seq.size()) +
                                  " exceeds max_ctx " + std::to_string(max_ctx_));
    }
  }
  return encoder_.embed(req, platos_key_.get(), vision_encoder_.get(), audio_encoder_.get());
}

void register_eg2_cpu_backend() {
  auto& reg = model::BackendRegistry::instance();
  if (reg.find(model::ModelFamily::EmbeddingGemma2).has_value()) return;
  reg.register_backend(model::BackendRegistration{
      model::ModelFamily::EmbeddingGemma2,
      "embeddinggemma2-cpu-f32",
      [](const model::RuntimeOptions& options) -> std::unique_ptr<model::ModelBackend> {
        return std::make_unique<Eg2CpuBackend>(options);
      },
  });
}

}  // namespace ingine::backends::embedding_gemma2
