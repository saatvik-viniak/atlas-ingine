// Image input for Inkling-2's vision tower (CPU, CUDA-free): data: URIs -> the RGB pixels the reference
// image processor (transformers Inkling2ImageProcessor) produces, their token count and a content
// identity for the prompt cache.
//
// The model sees an image as a grid of 40x40-pixel tiles, one language-model token each, in raster order:
// ceil(H/40) x ceil(W/40) tokens. Pixels are never resized by the model; an image above `max_tokens` is
// reduced here (area average, aspect ratio kept) so that it fits. Transparency is composited onto the
// processor's 8-pixel checkerboard of grey 204 / 153.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ingine::image {

inline constexpr int TILE = 40;                 // pixels per side of one image token
inline constexpr int MAX_TOKENS = 10609;        // 103 x 103 tiles: 4096 x 4096 pixels, the model's range

struct Limits {
  int max_tokens = 4096;                        // per image; larger images are reduced to fit
  int max_side = 16384;                         // source pixels per side
  std::int64_t max_pixels = std::int64_t(1) << 26;  // source pixels (decoded before any reduction)
};

// A malformed or unsupported image (HTTP 400). what() is the client-facing message.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Image {
  int width = 0, height = 0;                    // pixels in `rgb`
  int source_width = 0, source_height = 0;      // as decoded; differ from width/height after a reduction
  std::vector<std::uint8_t> rgb;                // [height, width, 3]
  std::uint64_t hash = 0;                       // of width, height and rgb

  [[nodiscard]] int rows() const { return (height + TILE - 1) / TILE; }
  [[nodiscard]] int cols() const { return (width + TILE - 1) / TILE; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// Tokens of a height x width image.
[[nodiscard]] int tokens(int height, int width);

// The largest (height, width) with the aspect ratio of the source and at most max_tokens tokens; the source
// size when it already fits. Never enlarges.
[[nodiscard]] std::pair<int, int> fit(int height, int width, int max_tokens);

// Standard base64 (RFC 4648 alphabet); ASCII whitespace is skipped and padding is optional.
[[nodiscard]] std::string decode_base64(std::string_view text);

// PNG, JPEG, WebP, GIF or BMP bytes (the first frame of an animation) -> Image. Throws Error.
[[nodiscard]] Image decode(std::span<const std::uint8_t> bytes, const Limits& limits = {});

// "data:[<media type>];base64,<data>" -> Image. The media type is ignored: the bytes say what they are.
[[nodiscard]] Image decode_data_uri(std::string_view uri, const Limits& limits = {});

// Decodes PNG, JPEG, WebP, GIF or BMP bytes directly to unresized RGB [height, width, 3] (Pillow RGB semantics,
// no 40x40 tile reduction or checkerboard compositing). Used by multimodal encoders such as EmbeddingGemma 2.
[[nodiscard]] Image decode_raw_rgb(std::span<const std::uint8_t> bytes, const Limits& limits = {});
[[nodiscard]] Image decode_raw_rgb_data_uri(std::string_view uri, const Limits& limits = {});

// Area-average reduction of [height, width, 3] bytes to [out_height, out_width, 3].
[[nodiscard]] std::vector<std::uint8_t> reduce(std::span<const std::uint8_t> rgb, int height, int width,
                                               int out_height, int out_width);

// Separable Catmull-Rom bicubic (a = -0.5) antialiased resize matching PIL.Image.Resampling.BICUBIC /
// torchvision.transforms.v2.functional.resize(..., BICUBIC, antialias=True). Supports both downscaling & upscaling.
[[nodiscard]] std::vector<std::uint8_t> resize_bicubic_antialias(std::span<const std::uint8_t> rgb, int height,
                                                                 int width, int out_height, int out_width);

[[nodiscard]] std::uint64_t hash_pixels(int width, int height, std::span<const std::uint8_t> rgb);

// Prompt id of image token k (0-based, raster order) of the image with this hash: negative, so it is
// never a vocabulary id, and a function of the image content and k, so the id-based prompt caches tell
// images apart and never match the middle of one image against another. In [-2^31, -1].
[[nodiscard]] std::int32_t token_id(std::uint64_t hash, int k);

}  // namespace ingine::image
