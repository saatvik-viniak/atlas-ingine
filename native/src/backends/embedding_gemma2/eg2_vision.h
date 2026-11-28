// Gemma-4 / EmbeddingGemma-2 Vision Tower preprocessing & pure-CPU FP32 encoder (#968 Phase 2).
//
// Implements:
//   - Aspect-ratio-preserving resize sizing to exact multiples of `patch_size * pooling_kernel_size` (48)
//     for supported soft-token budgets {70, 140, 280, 560, 1120}.
//   - Separable Catmull-Rom bicubic antialiased resize + [0, 1] rescale + HWC patchify [N_patches, 768] + (x, y) coords.
//   - Pure-CPU FP32 Gemma4VisionModel: PatchEmbedder (2*(x-0.5) -> Linear + 2D pos table) -> 16L bidirectional ViT
//     (12 heads, head_dim=64, 2D axial RoPE theta=100, q/k/unscaled-v RMSNorm, scaling=1.0, GeGLU MLP) ->
//     3x3 spatial average pool -> sqrt(768) -> unscaled RMSNorm -> embed_vision.embedding_projection [N_soft, 512].
#pragma once

#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "ingine/generator.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ingine::backends::embedding_gemma2 {

struct Eg2VisionSize {
  int target_height = 0;
  int target_width = 0;
  int patch_height = 0;
  int patch_width = 0;
  int num_patches = 0;
  int num_soft_tokens = 0;
};

struct Eg2PreprocessedImage {
  Eg2VisionSize size;
  std::vector<float> patches_f32;  // [num_patches, 3 * patch_size * patch_size] in [0, 1]
  std::vector<std::int32_t> pos_x; // [num_patches] in [0, patch_width - 1]
  std::vector<std::int32_t> pos_y; // [num_patches] in [0, patch_height - 1]
};

// Returns true if `max_soft_tokens` is in Gemma-4's supported set {70, 140, 280, 560, 1120}.
bool eg2_valid_max_soft_tokens(int max_soft_tokens) noexcept;

// Exact port of `transformers.models.gemma4.image_processing_gemma4.get_aspect_ratio_preserving_size`.
Eg2VisionSize eg2_aspect_ratio_preserving_size(int height, int width, int max_soft_tokens = 280,
                                               int patch_size = 16, int pooling_kernel_size = 3);

// Resizes (if needed) via bicubic antialias, rescales to [0, 1], and patchifies into [N_patches, 768] + (x, y) coords.
Eg2PreprocessedImage eg2_preprocess_image(const ingine::model::EmbeddingImage& image,
                                          const Eg2VisionConfig& vcfg);

// Pure CPU FP32 Gemma4VisionModel + embed_vision projector.
class Eg2CpuVisionEncoder {
 public:
  explicit Eg2CpuVisionEncoder(Eg2VisionWeightsF32 weights);

  const Eg2VisionConfig& config() const noexcept { return weights_.config; }
  int text_hidden_size() const noexcept { return weights_.text_hidden_size; }

  // Encodes one preprocessed image -> [num_soft_tokens, text_hidden_size] FP32 soft tokens.
  std::vector<float> encode_preprocessed(const Eg2PreprocessedImage& prep) const;

  // Preprocesses + encodes one raw RGB EmbeddingImage -> [num_soft_tokens, text_hidden_size] FP32 soft tokens.
  std::vector<float> encode_image(const ingine::model::EmbeddingImage& image) const;

 private:
  Eg2VisionWeightsF32 weights_;
};

}  // namespace ingine::backends::embedding_gemma2
