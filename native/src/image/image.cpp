#include "image/image.h"

#include <stb/stb_image.h>
#include <webp/decode.h>
#include <webp/demux.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace ingine::image {

namespace {

constexpr std::uint8_t HASH_VERSION = 1;  // of the pixel pipeline (compositing, reduction): part of the identity

std::uint64_t mix(std::uint64_t z) {  // splitmix64 finaliser
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Decoded pixels as the file has them: 1 grey, 2 grey + alpha, 3 RGB or 4 RGBA bytes per pixel.
struct Raw {
  int width = 0, height = 0, channels = 0;
  std::unique_ptr<std::uint8_t, void (*)(void*)> pixels{nullptr, +[](void*) {}};
};

void check_size(int width, int height, const Limits& limits) {
  if (width < 1 || height < 1) throw Error("image has no pixels");
  if (width > limits.max_side || height > limits.max_side || std::int64_t(width) * height > limits.max_pixels) {
    throw Error("image of " + std::to_string(width) + "x" + std::to_string(height) + " pixels is too large (at most " +
                std::to_string(limits.max_side) + " per side and " + std::to_string(limits.max_pixels) + " in all)");
  }
}

bool is_webp(std::span<const std::uint8_t> bytes) {
  return bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 && std::memcmp(bytes.data() + 8, "WEBP", 4) == 0;
}

// WebP (libwebp): RGBA when the file has an alpha channel, else RGB; of an animation, its first frame on the
// canvas (as the reference's decoder, Pillow, which also sits on libwebp).
Raw decode_webp(std::span<const std::uint8_t> bytes, const Limits& limits) {
  WebPBitstreamFeatures features;
  if (WebPGetFeatures(bytes.data(), bytes.size(), &features) != VP8_STATUS_OK) {
    throw Error("image is not a valid WebP file");
  }
  check_size(features.width, features.height, limits);
  Raw raw;
  raw.width = features.width;
  raw.height = features.height;
  raw.channels = features.has_alpha ? 4 : 3;
  if (features.has_animation) {
    WebPAnimDecoderOptions options;
    if (!WebPAnimDecoderOptionsInit(&options)) throw Error("WebP image could not be decoded");
    options.color_mode = MODE_RGBA;
    options.use_threads = 0;
    const WebPData data{bytes.data(), bytes.size()};
    const std::unique_ptr<WebPAnimDecoder, void (*)(WebPAnimDecoder*)> decoder(WebPAnimDecoderNew(&data, &options),
                                                                               &WebPAnimDecoderDelete);
    std::uint8_t* canvas = nullptr;  // owned by the decoder
    int timestamp = 0;
    if (!decoder || !WebPAnimDecoderGetNext(decoder.get(), &canvas, &timestamp)) {
      throw Error("WebP image could not be decoded");
    }
    const std::size_t pixels = std::size_t(raw.width) * raw.height;
    auto* copy = static_cast<std::uint8_t*>(std::malloc(pixels * std::size_t(raw.channels)));
    if (!copy) throw Error("image is too large");
    for (std::size_t i = 0; i < pixels; ++i) std::memcpy(copy + i * raw.channels, canvas + i * 4, std::size_t(raw.channels));
    raw.pixels = {copy, &std::free};
  } else {
    int width = 0, height = 0;
    std::uint8_t* pixels = features.has_alpha ? WebPDecodeRGBA(bytes.data(), bytes.size(), &width, &height)
                                              : WebPDecodeRGB(bytes.data(), bytes.size(), &width, &height);
    if (!pixels) throw Error("WebP image could not be decoded");
    raw.pixels = {pixels, &WebPFree};
    if (width != raw.width || height != raw.height) throw Error("WebP image could not be decoded");
  }
  return raw;
}

// PNG, JPEG, GIF (first frame) and BMP (stb_image), in the file's own channel count (palettes are expanded).
Raw decode_stb(std::span<const std::uint8_t> bytes, const Limits& limits) {
  const int length = int(bytes.size());
  Raw raw;
  if (!stbi_info_from_memory(bytes.data(), length, &raw.width, &raw.height, &raw.channels)) {
    throw Error("image is not a supported PNG, JPEG, WebP, GIF or BMP file");
  }
  check_size(raw.width, raw.height, limits);
  stbi_uc* pixels = stbi_load_from_memory(bytes.data(), length, &raw.width, &raw.height, &raw.channels, 0);
  if (!pixels) {
    const char* reason = stbi_failure_reason();
    throw Error(std::string("image could not be decoded") + (reason ? std::string(": ") + reason : std::string()));
  }
  raw.pixels = {pixels, +[](void* p) { stbi_image_free(p); }};
  if (raw.channels < 1 || raw.channels > 4) throw Error("image has an unsupported number of channels");
  return raw;
}

bool starts_with_nocase(std::string_view text, std::string_view prefix) {
  if (text.size() < prefix.size()) return false;
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    char c = text[i];
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    if (c != prefix[i]) return false;
  }
  return true;
}

}  // namespace

int tokens(int height, int width) { return ((height + TILE - 1) / TILE) * ((width + TILE - 1) / TILE); }

std::pair<int, int> fit(int height, int width, int max_tokens) {
  if (tokens(height, width) <= max_tokens) return {height, width};
  // Start from the scale that gives max_tokens tiles of area, then shrink the longer side a pixel at a time
  // (the other follows the aspect ratio) until the tile grid, which rounds up, fits.
  const double scale = std::sqrt(double(max_tokens) * TILE * TILE / (double(height) * double(width)));
  const bool wide = width >= height;
  int longer = std::max(1, int(std::floor((wide ? width : height) * scale)));
  for (;; --longer) {
    const double ratio = double(wide ? height : width) / double(wide ? width : height);
    const int shorter = std::max(1, int(std::lround(longer * ratio)));
    const int h = wide ? shorter : longer, w = wide ? longer : shorter;
    if (tokens(h, w) <= max_tokens || longer == 1) return {h, w};
  }
}

std::string decode_base64(std::string_view text) {
  static const std::array<std::int8_t, 256> table = [] {
    std::array<std::int8_t, 256> t;
    t.fill(-1);
    const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(alphabet[i])] = std::int8_t(i);
    return t;
  }();
  std::string out;
  out.reserve(text.size() / 4 * 3 + 3);
  std::uint32_t buffer = 0;
  int bits = 0;
  std::size_t symbols = 0;
  bool padded = false;
  for (const char ch : text) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (c == '=') {
      padded = true;
      continue;
    }
    const int v = table[c];
    if (v < 0 || padded) throw Error("image data is not valid base64");
    buffer = (buffer << 6) | std::uint32_t(v);
    bits += 6;
    ++symbols;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(char((buffer >> bits) & 0xFF));
    }
  }
  if (symbols % 4 == 1) throw Error("image data is not valid base64");
  return out;
}

std::uint64_t hash_pixels(int width, int height, std::span<const std::uint8_t> rgb) {
  std::uint64_t h = mix((std::uint64_t(std::uint32_t(width)) << 32) | std::uint32_t(height)) ^ HASH_VERSION;
  const std::uint8_t* p = rgb.data();
  std::size_t n = rgb.size();
  for (; n >= 8; p += 8, n -= 8) {
    std::uint64_t word;
    std::memcpy(&word, p, 8);
    h = mix(h + 0x9E3779B97F4A7C15ull + word);
  }
  if (n) {
    std::uint64_t word = 0;
    std::memcpy(&word, p, n);
    h = mix(h + 0x9E3779B97F4A7C15ull + word);
  }
  return mix(h ^ std::uint64_t(rgb.size()));
}

std::int32_t token_id(std::uint64_t hash, int k) {
  const std::uint64_t v = mix(hash ^ (std::uint64_t(std::uint32_t(k)) + 1) * 0x9E3779B97F4A7C15ull) >> 33;  // 31 bits
  return std::int32_t(-1 - std::int64_t(v));
}

Image decode(std::span<const std::uint8_t> bytes, const Limits& limits) {
  if (bytes.empty()) throw Error("image is empty");
  if (bytes.size() > std::size_t(0x7FFFFFFF)) throw Error("image is too large");
  Raw raw = is_webp(bytes) ? decode_webp(bytes, limits) : decode_stb(bytes, limits);
  const int width = raw.width, height = raw.height, channels = raw.channels;

  Image image;
  image.source_width = width;
  image.source_height = height;
  std::vector<std::uint8_t> rgb(std::size_t(width) * height * 3);
  const std::uint8_t* src = raw.pixels.get();
  const bool grey = channels <= 2, alpha = channels == 2 || channels == 4;
  for (int y = 0; y < height; ++y) {
    std::uint8_t* dst = rgb.data() + std::size_t(y) * width * 3;
    for (int x = 0; x < width; ++x, src += channels, dst += 3) {
      int c[3] = {src[0], grey ? src[0] : src[1], grey ? src[0] : src[2]};
      if (alpha) {  // the processor's checkerboard: 8-pixel squares, 204 where (y/8 + x/8) is even, else 153
        const int a = src[channels - 1];
        const int background = ((y >> 3) + (x >> 3)) % 2 == 0 ? 204 : 153;
        for (int& v : c) v = (v * a + background * (255 - a) + 127) / 255;
      }
      dst[0] = std::uint8_t(c[0]);
      dst[1] = std::uint8_t(c[1]);
      dst[2] = std::uint8_t(c[2]);
    }
  }
  raw.pixels.reset();

  const auto [out_height, out_width] = fit(height, width, std::clamp(limits.max_tokens, 1, MAX_TOKENS));
  if (out_height != height || out_width != width) rgb = reduce(rgb, height, width, out_height, out_width);
  image.width = out_width;
  image.height = out_height;
  image.rgb = std::move(rgb);
  image.hash = hash_pixels(image.width, image.height, image.rgb);
  return image;
}

Image decode_data_uri(std::string_view uri, const Limits& limits) {
  if (!starts_with_nocase(uri, "data:")) throw Error("only data: URIs are supported for images");
  const std::size_t comma = uri.find(',');
  if (comma == std::string_view::npos) throw Error("image data: URI has no data");
  const std::string_view header = uri.substr(5, comma - 5);
  constexpr std::string_view marker = ";base64";
  if (header.size() < marker.size() || !starts_with_nocase(header.substr(header.size() - marker.size()), marker)) {
    throw Error("image data: URI must be base64-encoded (data:image/png;base64,...)");
  }
  const std::string bytes = decode_base64(uri.substr(comma + 1));
  return decode(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()), limits);
}

Image decode_raw_rgb(std::span<const std::uint8_t> bytes, const Limits& limits) {
  if (bytes.empty()) throw Error("image is empty");
  if (bytes.size() > std::size_t(0x7FFFFFFF)) throw Error("image is too large");
  Raw raw = is_webp(bytes) ? decode_webp(bytes, limits) : decode_stb(bytes, limits);
  const int width = raw.width, height = raw.height, channels = raw.channels;

  Image image;
  image.source_width = width;
  image.source_height = height;
  image.width = width;
  image.height = height;
  image.rgb.resize(std::size_t(width) * height * 3);
  const std::uint8_t* src = raw.pixels.get();
  const bool grey = channels <= 2;
  for (int y = 0; y < height; ++y) {
    std::uint8_t* dst = image.rgb.data() + std::size_t(y) * width * 3;
    for (int x = 0; x < width; ++x, src += channels, dst += 3) {
      dst[0] = src[0];
      dst[1] = grey ? src[0] : src[1];
      dst[2] = grey ? src[0] : src[2];
    }
  }
  image.hash = hash_pixels(image.width, image.height, image.rgb);
  return image;
}

Image decode_raw_rgb_data_uri(std::string_view uri, const Limits& limits) {
  if (!starts_with_nocase(uri, "data:")) throw Error("only data: URIs are supported for images");
  const std::size_t comma = uri.find(',');
  if (comma == std::string_view::npos) throw Error("image data: URI has no data");
  const std::string_view header = uri.substr(5, comma - 5);
  constexpr std::string_view marker = ";base64";
  if (header.size() < marker.size() || !starts_with_nocase(header.substr(header.size() - marker.size()), marker)) {
    throw Error("image data: URI must be base64-encoded (data:image/png;base64,...)");
  }
  const std::string bytes = decode_base64(uri.substr(comma + 1));
  return decode_raw_rgb(
      std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()), limits);
}

}  // namespace ingine::image
