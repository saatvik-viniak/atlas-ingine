// Area-average (box filter) reduction: every output pixel is the mean of the source area it covers, with
// partial pixels weighted by their overlap. Separable: columns first, then rows.
#include "image/image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ingine::image {

namespace {

struct Tap {
  int first = 0;               // first source index
  std::vector<float> weights;  // overlap of each source cell with the output cell, normalised to sum 1
};

std::vector<Tap> taps(int in, int out) {
  std::vector<Tap> result(static_cast<std::size_t>(out));
  const double step = double(in) / double(out);
  for (int o = 0; o < out; ++o) {
    const double lo = o * step, hi = (o + 1) * step;
    const int first = std::min(in - 1, int(std::floor(lo)));
    const int last = std::min(in - 1, std::max(first, int(std::ceil(hi)) - 1));
    Tap& tap = result[static_cast<std::size_t>(o)];
    tap.first = first;
    double sum = 0;
    for (int i = first; i <= last; ++i) {
      const double w = std::max(0.0, std::min(hi, double(i + 1)) - std::max(lo, double(i)));
      tap.weights.push_back(float(w));
      sum += w;
    }
    for (float& w : tap.weights) w = float(w / sum);
  }
  return result;
}

double bicubic_filter(double x) noexcept {
  constexpr double a = -0.5;  // Catmull-Rom (Pillow ImagingResample BICUBIC / torchvision antialias=True)
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

std::vector<Tap> bicubic_taps(int in, int out) {
  std::vector<Tap> result(static_cast<std::size_t>(out));
  const double ratio = double(in) / double(out);
  const double scale = std::max(1.0, ratio);
  const double support = 2.0 * scale;
  const double inv_scale = 1.0 / scale;
  for (int o = 0; o < out; ++o) {
    const double center = (double(o) + 0.5) * ratio;
    const int xmin = std::max(0, int(std::floor(center - support)));
    const int xmax = std::min(in, int(std::ceil(center + support)));
    Tap& tap = result[static_cast<std::size_t>(o)];
    tap.first = xmin;
    tap.weights.reserve(static_cast<std::size_t>(std::max(1, xmax - xmin)));
    double sum = 0.0;
    for (int i = xmin; i < xmax; ++i) {
      const double w = bicubic_filter((double(i) + 0.5 - center) * inv_scale);
      tap.weights.push_back(float(w));
      sum += w;
    }
    if (sum != 0.0) {
      for (float& w : tap.weights) w = float(double(w) / sum);
    }
  }
  return result;
}

}  // namespace

std::vector<std::uint8_t> reduce(std::span<const std::uint8_t> rgb, int height, int width, int out_height,
                                 int out_width) {
  if (height < 1 || width < 1 || out_height < 1 || out_width < 1 || out_height > height || out_width > width ||
      rgb.size() != std::size_t(height) * width * 3) {
    throw std::invalid_argument("image::reduce: bad sizes");
  }
  const std::vector<Tap> tx = taps(width, out_width), ty = taps(height, out_height);
  std::vector<float> rows(std::size_t(height) * out_width * 3);  // [height, out_width, 3]
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* src = rgb.data() + std::size_t(y) * width * 3;
    float* dst = rows.data() + std::size_t(y) * out_width * 3;
    for (int x = 0; x < out_width; ++x, dst += 3) {
      const Tap& tap = tx[static_cast<std::size_t>(x)];
      float acc[3] = {0, 0, 0};
      const std::uint8_t* p = src + std::size_t(tap.first) * 3;
      for (const float w : tap.weights) {
        acc[0] += w * p[0];
        acc[1] += w * p[1];
        acc[2] += w * p[2];
        p += 3;
      }
      dst[0] = acc[0];
      dst[1] = acc[1];
      dst[2] = acc[2];
    }
  }
  std::vector<std::uint8_t> out(std::size_t(out_height) * out_width * 3);
  std::vector<float> acc(std::size_t(out_width) * 3);
  for (int y = 0; y < out_height; ++y) {
    const Tap& tap = ty[static_cast<std::size_t>(y)];
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (std::size_t i = 0; i < tap.weights.size(); ++i) {
      const float w = tap.weights[i];
      const float* src = rows.data() + (std::size_t(tap.first) + i) * out_width * 3;
      for (std::size_t j = 0; j < acc.size(); ++j) acc[j] += w * src[j];
    }
    std::uint8_t* dst = out.data() + std::size_t(y) * out_width * 3;
    for (std::size_t j = 0; j < acc.size(); ++j) {
      dst[j] = std::uint8_t(std::clamp(std::lround(acc[j]), 0l, 255l));
    }
  }
  return out;
}

std::vector<std::uint8_t> resize_bicubic_antialias(std::span<const std::uint8_t> rgb, int height, int width,
                                                   int out_height, int out_width) {
  if (height < 1 || width < 1 || out_height < 1 || out_width < 1 ||
      rgb.size() != std::size_t(height) * width * 3) {
    throw std::invalid_argument("image::resize_bicubic_antialias: bad sizes");
  }
  if (height == out_height && width == out_width) {
    return std::vector<std::uint8_t>(rgb.begin(), rgb.end());
  }
  const std::vector<Tap> tx = bicubic_taps(width, out_width);
  const std::vector<Tap> ty = bicubic_taps(height, out_height);
  std::vector<float> rows(std::size_t(height) * out_width * 3);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* src = rgb.data() + std::size_t(y) * width * 3;
    float* dst = rows.data() + std::size_t(y) * out_width * 3;
    for (int x = 0; x < out_width; ++x, dst += 3) {
      const Tap& tap = tx[static_cast<std::size_t>(x)];
      float acc[3] = {0.0f, 0.0f, 0.0f};
      const std::uint8_t* p = src + std::size_t(tap.first) * 3;
      for (const float w : tap.weights) {
        acc[0] += w * float(p[0]);
        acc[1] += w * float(p[1]);
        acc[2] += w * float(p[2]);
        p += 3;
      }
      dst[0] = acc[0];
      dst[1] = acc[1];
      dst[2] = acc[2];
    }
  }
  std::vector<std::uint8_t> out(std::size_t(out_height) * out_width * 3);
  std::vector<float> acc(std::size_t(out_width) * 3);
  for (int y = 0; y < out_height; ++y) {
    const Tap& tap = ty[static_cast<std::size_t>(y)];
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (std::size_t i = 0; i < tap.weights.size(); ++i) {
      const float w = tap.weights[i];
      const float* src = rows.data() + (std::size_t(tap.first) + i) * out_width * 3;
      for (std::size_t j = 0; j < acc.size(); ++j) acc[j] += w * src[j];
    }
    std::uint8_t* dst = out.data() + std::size_t(y) * out_width * 3;
    for (std::size_t j = 0; j < acc.size(); ++j) {
      dst[j] = std::uint8_t(std::clamp(std::lround(acc[j]), 0l, 255l));
    }
  }
  return out;
}

}  // namespace ingine::image
