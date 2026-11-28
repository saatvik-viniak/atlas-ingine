#include "backends/embedding_gemma2/eg2_vision.h"

#include "image/image.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ingine::backends::embedding_gemma2 {
namespace {

constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

inline float gelu_pytorch_tanh(float x) noexcept {
  const float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

void rms_norm_rows(const float* x, const float* weight, int rows, int dim, float eps, float* out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    double sum_sq = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = static_cast<double>(xr[d]);
      sum_sq += v * v;
    }
    const float inv_rms =
        static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + static_cast<double>(eps)));
    if (weight != nullptr) {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv_rms * weight[d];
    } else {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv_rms;
    }
  }
}

void linear_f32(const float* x, const float* w, int rows, int in_dim, int out_dim, float* out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim);
    for (int o = 0; o < out_dim; ++o) {
      const float* wo = w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim);
      float acc = 0.0f;
      for (int k = 0; k < in_dim; ++k) acc += xr[k] * wo[k];
      yr[o] = acc;
    }
  }
}

// 2D axial RoPE matching Gemma4VisionRotaryEmbedding + apply_multidimensional_rope:
// Splits head_dim into two equal halves [0..half-1] (rotated by pos_x) and [half..head_dim-1] (rotated by pos_y).
// Each half uses standard rotate_half over quarter = head_dim / 4 pairs.
void apply_2d_axial_rope_inplace(float* x, int N, int n_heads, int head_dim, std::span<const std::int32_t> pos_x,
                                 std::span<const std::int32_t> pos_y, double rope_theta) {
  const int spatial_dim = head_dim / 2;
  const int quarter = spatial_dim / 2;
  std::vector<double> inv_freq(static_cast<std::size_t>(quarter));
  for (int j = 0; j < quarter; ++j) {
    inv_freq[static_cast<std::size_t>(j)] =
        1.0 / std::pow(rope_theta, (2.0 * static_cast<double>(j)) / static_cast<double>(spatial_dim));
  }
  for (int n = 0; n < N; ++n) {
    const double px = static_cast<double>(pos_x[static_cast<std::size_t>(n)]);
    const double py = static_cast<double>(pos_y[static_cast<std::size_t>(n)]);
    for (int j = 0; j < quarter; ++j) {
      const double ax = px * inv_freq[static_cast<std::size_t>(j)];
      const double ay = py * inv_freq[static_cast<std::size_t>(j)];
      const float cx = static_cast<float>(std::cos(ax));
      const float sx = static_cast<float>(std::sin(ax));
      const float cy = static_cast<float>(std::cos(ay));
      const float sy = static_cast<float>(std::sin(ay));
      for (int h = 0; h < n_heads; ++h) {
        float* ptr = x + (static_cast<std::size_t>(n) * static_cast<std::size_t>(n_heads) + static_cast<std::size_t>(h)) *
                             static_cast<std::size_t>(head_dim);
        // X-axis half: [0 .. spatial_dim - 1]
        const float x1 = ptr[j];
        const float x2 = ptr[j + quarter];
        ptr[j] = x1 * cx - x2 * sx;
        ptr[j + quarter] = x2 * cx + x1 * sx;
        // Y-axis half: [spatial_dim .. head_dim - 1]
        const float y1 = ptr[spatial_dim + j];
        const float y2 = ptr[spatial_dim + j + quarter];
        ptr[spatial_dim + j] = y1 * cy - y2 * sy;
        ptr[spatial_dim + j + quarter] = y2 * cy + y1 * sy;
      }
    }
  }
}

}  // namespace

bool eg2_valid_max_soft_tokens(int max_soft_tokens) noexcept {
  return max_soft_tokens == 70 || max_soft_tokens == 140 || max_soft_tokens == 280 || max_soft_tokens == 560 ||
         max_soft_tokens == 1120;
}

Eg2VisionSize eg2_aspect_ratio_preserving_size(int height, int width, int max_soft_tokens, int patch_size,
                                               int pooling_kernel_size) {
  if (height <= 0 || width <= 0) {
    throw std::invalid_argument("eg2_vision: image height and width must be positive");
  }
  if (!eg2_valid_max_soft_tokens(max_soft_tokens)) {
    throw std::invalid_argument("eg2_vision: max_soft_tokens must be one of {70, 140, 280, 560, 1120}, got " +
                                std::to_string(max_soft_tokens));
  }
  const int k2 = pooling_kernel_size * pooling_kernel_size;
  const int max_patches = max_soft_tokens * k2;
  const double total_px = static_cast<double>(height) * static_cast<double>(width);
  const double target_px = static_cast<double>(max_patches) * static_cast<double>(patch_size * patch_size);
  const double factor = std::sqrt(target_px / total_px);
  const double ideal_h = factor * static_cast<double>(height);
  const double ideal_w = factor * static_cast<double>(width);
  const int side_mult = pooling_kernel_size * patch_size;

  int target_h = static_cast<int>(std::floor(ideal_h / static_cast<double>(side_mult))) * side_mult;
  int target_w = static_cast<int>(std::floor(ideal_w / static_cast<double>(side_mult))) * side_mult;

  if (target_h == 0 && target_w == 0) {
    throw std::invalid_argument("eg2_vision: image aspect ratio collapses to 0x0 under side_mult=" +
                                std::to_string(side_mult));
  }
  const int max_side_len = (max_patches / k2) * side_mult;
  if (target_h == 0) {
    target_h = side_mult;
    target_w = std::min(static_cast<int>(std::floor(static_cast<double>(width) / static_cast<double>(height))) * side_mult,
                        max_side_len);
    if (target_w == 0) target_w = side_mult;
  } else if (target_w == 0) {
    target_w = side_mult;
    target_h = std::min(static_cast<int>(std::floor(static_cast<double>(height) / static_cast<double>(width))) * side_mult,
                        max_side_len);
    if (target_h == 0) target_h = side_mult;
  }

  if (static_cast<std::int64_t>(target_h) * target_w > static_cast<std::int64_t>(max_patches) * patch_size * patch_size) {
    throw std::invalid_argument("eg2_vision: computed target size exceeds patch budget");
  }

  Eg2VisionSize sz;
  sz.target_height = target_h;
  sz.target_width = target_w;
  sz.patch_height = target_h / patch_size;
  sz.patch_width = target_w / patch_size;
  sz.num_patches = sz.patch_height * sz.patch_width;
  sz.num_soft_tokens = sz.num_patches / k2;
  return sz;
}

Eg2PreprocessedImage eg2_preprocess_image(const ingine::model::EmbeddingImage& image, const Eg2VisionConfig& vcfg) {
  if (image.height <= 0 || image.width <= 0 ||
      image.rgb.size() != static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.width) * 3u) {
    throw std::invalid_argument("eg2_preprocess_image: invalid RGB buffer dimensions");
  }
  const int max_st = image.max_soft_tokens > 0
                         ? image.max_soft_tokens
                         : (image.is_video_frame ? vcfg.video_max_soft_tokens : vcfg.default_output_length);
  const Eg2VisionSize sz =
      eg2_aspect_ratio_preserving_size(image.height, image.width, max_st, vcfg.patch_size, vcfg.pooling_kernel_size);

  std::vector<std::uint8_t> resized_storage;
  const std::uint8_t* pixels = image.rgb.data();
  if (sz.target_height != image.height || sz.target_width != image.width) {
    resized_storage = ingine::image::resize_bicubic_antialias(image.rgb, image.height, image.width, sz.target_height,
                                                              sz.target_width);
    pixels = resized_storage.data();
  }

  const int P = vcfg.patch_size;
  const int patch_dim = 3 * P * P;
  Eg2PreprocessedImage out;
  out.size = sz;
  out.patches_f32.resize(static_cast<std::size_t>(sz.num_patches) * static_cast<std::size_t>(patch_dim));
  out.pos_x.resize(static_cast<std::size_t>(sz.num_patches));
  out.pos_y.resize(static_cast<std::size_t>(sz.num_patches));

  constexpr float kInv255 = 1.0f / 255.0f;
  for (int py = 0; py < sz.patch_height; ++py) {
    for (int px = 0; px < sz.patch_width; ++px) {
      const int idx = py * sz.patch_width + px;
      out.pos_x[static_cast<std::size_t>(idx)] = px;
      out.pos_y[static_cast<std::size_t>(idx)] = py;
      float* dst = out.patches_f32.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(patch_dim);
      int off = 0;
      for (int dy = 0; dy < P; ++dy) {
        const int y = py * P + dy;
        const std::uint8_t* row = pixels + (static_cast<std::size_t>(y) * static_cast<std::size_t>(sz.target_width) +
                                            static_cast<std::size_t>(px * P)) *
                                               3u;
        for (int dx = 0; dx < P; ++dx) {
          dst[off++] = static_cast<float>(row[dx * 3 + 0]) * kInv255;
          dst[off++] = static_cast<float>(row[dx * 3 + 1]) * kInv255;
          dst[off++] = static_cast<float>(row[dx * 3 + 2]) * kInv255;
        }
      }
    }
  }
  return out;
}

Eg2CpuVisionEncoder::Eg2CpuVisionEncoder(Eg2VisionWeightsF32 weights) : weights_(std::move(weights)) {}

std::vector<float> Eg2CpuVisionEncoder::encode_preprocessed(const Eg2PreprocessedImage& prep) const {
  const Eg2VisionConfig& vc = weights_.config;
  const int N = prep.size.num_patches;
  const int VH = vc.hidden_size;
  const int VI = vc.intermediate_size;
  const int n_heads = vc.num_attention_heads;
  const int n_kv = vc.num_key_value_heads;
  const int hd = vc.head_dim;
  const int q_dim = n_heads * hd;
  const int kv_dim = n_kv * hd;
  const int kv_groups = n_heads / n_kv;
  const int patch_in = 3 * vc.patch_size * vc.patch_size;

  // 1. PatchEmbedder: pixel_values = 2 * (pv - 0.5) -> Linear -> + pos_table[0, x] + pos_table[1, y]
  std::vector<float> pv_scaled(prep.patches_f32.size());
  for (std::size_t i = 0; i < pv_scaled.size(); ++i) {
    pv_scaled[i] = 2.0f * (prep.patches_f32[i] - 0.5f);
  }
  std::vector<float> hidden(static_cast<std::size_t>(N) * static_cast<std::size_t>(VH));
  linear_f32(pv_scaled.data(), weights_.patch_input_proj.data(), N, patch_in, VH, hidden.data());

  const std::size_t pos_stride = static_cast<std::size_t>(vc.position_embedding_size) * static_cast<std::size_t>(VH);
  const float* tbl_x = weights_.position_embedding_table.data();
  const float* tbl_y = tbl_x + pos_stride;
  for (int n = 0; n < N; ++n) {
    const int px = prep.pos_x[static_cast<std::size_t>(n)];
    const int py = prep.pos_y[static_cast<std::size_t>(n)];
    if (px < 0 || px >= vc.position_embedding_size || py < 0 || py >= vc.position_embedding_size) {
      throw std::invalid_argument("Eg2CpuVisionEncoder: patch coordinate out of position_embedding_table range");
    }
    const float* ex = tbl_x + static_cast<std::size_t>(px) * static_cast<std::size_t>(VH);
    const float* ey = tbl_y + static_cast<std::size_t>(py) * static_cast<std::size_t>(VH);
    float* dst = hidden.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(VH);
    for (int d = 0; d < VH; ++d) dst[d] += ex[d] + ey[d];
  }

  // 2. 16 bidirectional ViT encoder layers
  std::vector<float> norm_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(VH));
  std::vector<float> sub_out(static_cast<std::size_t>(N) * static_cast<std::size_t>(VH));
  std::vector<float> q_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(q_dim));
  std::vector<float> k_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(kv_dim));
  std::vector<float> v_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(kv_dim));
  std::vector<float> attn_ctx(static_cast<std::size_t>(N) * static_cast<std::size_t>(q_dim));
  std::vector<float> gate_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(VI));
  std::vector<float> up_buf(static_cast<std::size_t>(N) * static_cast<std::size_t>(VI));
  std::vector<float> scores(static_cast<std::size_t>(N));

  for (int li = 0; li < vc.num_hidden_layers; ++li) {
    const auto& lw = weights_.layers[static_cast<std::size_t>(li)];

    // Self-attention
    rms_norm_rows(hidden.data(), lw.input_layernorm.data(), N, VH, vc.rms_norm_eps, norm_buf.data());
    linear_f32(norm_buf.data(), lw.q_proj.data(), N, VH, q_dim, q_buf.data());
    linear_f32(norm_buf.data(), lw.k_proj.data(), N, VH, kv_dim, k_buf.data());
    linear_f32(norm_buf.data(), lw.v_proj.data(), N, VH, kv_dim, v_buf.data());

    rms_norm_rows(q_buf.data(), lw.q_norm.data(), N * n_heads, hd, vc.rms_norm_eps, q_buf.data());
    rms_norm_rows(k_buf.data(), lw.k_norm.data(), N * n_kv, hd, vc.rms_norm_eps, k_buf.data());
    rms_norm_rows(v_buf.data(), nullptr, N * n_kv, hd, vc.rms_norm_eps, v_buf.data());

    apply_2d_axial_rope_inplace(q_buf.data(), N, n_heads, hd, prep.pos_x, prep.pos_y, vc.rope_theta);
    apply_2d_axial_rope_inplace(k_buf.data(), N, n_kv, hd, prep.pos_x, prep.pos_y, vc.rope_theta);

    // Full bidirectional attention (scaling = 1.0)
    for (int h = 0; h < n_heads; ++h) {
      const int kvh = h / kv_groups;
      for (int qi = 0; qi < N; ++qi) {
        const float* qp = q_buf.data() + (static_cast<std::size_t>(qi) * n_heads + h) * hd;
        float max_s = -std::numeric_limits<float>::infinity();
        for (int ki = 0; ki < N; ++ki) {
          const float* kp = k_buf.data() + (static_cast<std::size_t>(ki) * n_kv + kvh) * hd;
          float dot = 0.0f;
          for (int d = 0; d < hd; ++d) dot += qp[d] * kp[d];
          scores[static_cast<std::size_t>(ki)] = dot;
          if (dot > max_s) max_s = dot;
        }
        double sum_exp = 0.0;
        for (int ki = 0; ki < N; ++ki) {
          const double e = std::exp(static_cast<double>(scores[static_cast<std::size_t>(ki)] - max_s));
          scores[static_cast<std::size_t>(ki)] = static_cast<float>(e);
          sum_exp += e;
        }
        const float inv_sum = static_cast<float>(1.0 / sum_exp);
        float* op = attn_ctx.data() + (static_cast<std::size_t>(qi) * n_heads + h) * hd;
        std::fill(op, op + hd, 0.0f);
        for (int ki = 0; ki < N; ++ki) {
          const float w_attn = scores[static_cast<std::size_t>(ki)] * inv_sum;
          const float* vp = v_buf.data() + (static_cast<std::size_t>(ki) * n_kv + kvh) * hd;
          for (int d = 0; d < hd; ++d) op[d] += w_attn * vp[d];
        }
      }
    }

    linear_f32(attn_ctx.data(), lw.o_proj.data(), N, q_dim, VH, sub_out.data());
    rms_norm_rows(sub_out.data(), lw.post_attention_layernorm.data(), N, VH, vc.rms_norm_eps, sub_out.data());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += sub_out[i];

    // GeGLU MLP
    rms_norm_rows(hidden.data(), lw.pre_feedforward_layernorm.data(), N, VH, vc.rms_norm_eps, norm_buf.data());
    linear_f32(norm_buf.data(), lw.gate_proj.data(), N, VH, VI, gate_buf.data());
    linear_f32(norm_buf.data(), lw.up_proj.data(), N, VH, VI, up_buf.data());
    for (std::size_t i = 0; i < gate_buf.size(); ++i) {
      gate_buf[i] = gelu_pytorch_tanh(gate_buf[i]) * up_buf[i];
    }
    linear_f32(gate_buf.data(), lw.down_proj.data(), N, VI, VH, sub_out.data());
    rms_norm_rows(sub_out.data(), lw.post_feedforward_layernorm.data(), N, VH, vc.rms_norm_eps, sub_out.data());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += sub_out[i];
  }

  // 3. Spatial k x k average pool -> * sqrt(VH) in FP32 -> unscaled RMSNorm -> embed_vision.embedding_projection
  const int k = vc.pooling_kernel_size;
  const int k2 = k * k;
  const int oh = prep.size.patch_height / k;
  const int ow = prep.size.patch_width / k;
  const int n_soft = oh * ow;
  const float scale = std::sqrt(static_cast<float>(VH)) / static_cast<float>(k2);

  std::vector<float> pooled(static_cast<std::size_t>(n_soft) * static_cast<std::size_t>(VH), 0.0f);
  for (int oy = 0; oy < oh; ++oy) {
    for (int ox = 0; ox < ow; ++ox) {
      float* dst = pooled.data() + static_cast<std::size_t>(oy * ow + ox) * static_cast<std::size_t>(VH);
      for (int dy = 0; dy < k; ++dy) {
        for (int dx = 0; dx < k; ++dx) {
          const int p_idx = (oy * k + dy) * prep.size.patch_width + (ox * k + dx);
          const float* src = hidden.data() + static_cast<std::size_t>(p_idx) * static_cast<std::size_t>(VH);
          for (int d = 0; d < VH; ++d) dst[d] += src[d];
        }
      }
      for (int d = 0; d < VH; ++d) dst[d] *= scale;
    }
  }

  rms_norm_rows(pooled.data(), nullptr, n_soft, VH, vc.rms_norm_eps, pooled.data());

  const int TH = weights_.text_hidden_size;
  std::vector<float> soft_tokens(static_cast<std::size_t>(n_soft) * static_cast<std::size_t>(TH));
  linear_f32(pooled.data(), weights_.embed_vision_projection.data(), n_soft, VH, TH, soft_tokens.data());
  return soft_tokens;
}

std::vector<float> Eg2CpuVisionEncoder::encode_image(const ingine::model::EmbeddingImage& image) const {
  return encode_preprocessed(eg2_preprocess_image(image, weights_.config));
}

}  // namespace ingine::backends::embedding_gemma2
