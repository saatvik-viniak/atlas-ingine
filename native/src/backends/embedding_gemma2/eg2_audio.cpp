#include "backends/embedding_gemma2/eg2_audio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::backends::embedding_gemma2 {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr int kTargetSampleRate = 16000;
constexpr int kMaxSamples = 480000;  // 30s @ 16kHz
constexpr int kPadMultiple = 128;
constexpr int kFrameLength = 320;
constexpr int kHopLength = 160;
constexpr int kFftLength = 512;
constexpr int kNumFreqBins = kFftLength / 2 + 1;  // 257
constexpr int kNumMels = 128;
constexpr double kMelFloor = 0.001;

std::uint16_t read_le16(const std::uint8_t* p) noexcept {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t read_le32(const std::uint8_t* p) noexcept {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::vector<std::uint8_t> decode_base64_bytes(std::string_view b64) {
  static constexpr std::array<int, 256> kDec = []() {
    std::array<int, 256> t{};
    t.fill(-1);
    for (int i = 0; i < 26; ++i) t[static_cast<std::size_t>('A' + i)] = i;
    for (int i = 0; i < 26; ++i) t[static_cast<std::size_t>('a' + i)] = 26 + i;
    for (int i = 0; i < 10; ++i) t[static_cast<std::size_t>('0' + i)] = 52 + i;
    t[static_cast<std::size_t>('+')] = 62;
    t[static_cast<std::size_t>('/')] = 63;
    t[static_cast<std::size_t>('-')] = 62;
    t[static_cast<std::size_t>('_')] = 63;
    return t;
  }();

  std::vector<std::uint8_t> out;
  out.reserve((b64.size() * 3) / 4);
  int val = 0;
  int valb = -8;
  for (unsigned char c : b64) {
    if (c == '=' || c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
    const int d = kDec[c];
    if (d < 0) throw std::invalid_argument("EmbeddingGemma2 audio: invalid base64 character");
    val = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xff));
      valb -= 8;
    }
  }
  return out;
}

// In-place radix-2 Cooley-Tukey complex FFT of power-of-two length N (double precision).
void fft_radix2(double* re, double* im, int N) noexcept {
  // Bit-reversal permutation
  for (int i = 1, j = 0; i < N; ++i) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (int len = 2; len <= N; len <<= 1) {
    const double ang = -2.0 * kPi / static_cast<double>(len);
    const double wlen_r = std::cos(ang);
    const double wlen_i = std::sin(ang);
    const int half = len >> 1;
    for (int i = 0; i < N; i += len) {
      double wr = 1.0, wi = 0.0;
      for (int j = 0; j < half; ++j) {
        const int u = i + j;
        const int v = u + half;
        const double vr = re[v] * wr - im[v] * wi;
        const double vi = re[v] * wi + im[v] * wr;
        re[v] = re[u] - vr;
        im[v] = im[u] - vi;
        re[u] += vr;
        im[u] += vi;
        const double nwr = wr * wlen_r - wi * wlen_i;
        wi = wr * wlen_i + wi * wlen_r;
        wr = nwr;
      }
    }
  }
}

struct MelTables {
  std::array<float, kFrameLength> window{};
  // [kNumFreqBins, kNumMels] matching transformers.audio_utils.mel_filter_bank(..., norm=None, mel_scale="htk")
  std::vector<double> filterbank;

  MelTables() : filterbank(static_cast<std::size_t>(kNumFreqBins) * static_cast<std::size_t>(kNumMels), 0.0) {
    // Periodic Hann window matching window_function(320, "hann", periodic=True).astype(np.float32)
    for (int n = 0; n < kFrameLength; ++n) {
      window[static_cast<std::size_t>(n)] =
          static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(n) / static_cast<double>(kFrameLength)));
    }
    auto hz_to_mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel_to_hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };

    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(8000.0);
    std::array<double, kNumMels + 2> filter_freqs{};
    for (int i = 0; i < kNumMels + 2; ++i) {
      const double m = mel_min + (mel_max - mel_min) * (static_cast<double>(i) / static_cast<double>(kNumMels + 1));
      filter_freqs[static_cast<std::size_t>(i)] = mel_to_hz(m);
    }
    for (int b = 0; b < kNumFreqBins; ++b) {
      const double f = 8000.0 * (static_cast<double>(b) / static_cast<double>(kNumFreqBins - 1));
      for (int m = 0; m < kNumMels; ++m) {
        const double f_left = filter_freqs[static_cast<std::size_t>(m)];
        const double f_mid = filter_freqs[static_cast<std::size_t>(m + 1)];
        const double f_right = filter_freqs[static_cast<std::size_t>(m + 2)];
        const double down = (f - f_left) / (f_mid - f_left);
        const double up = (f_right - f) / (f_right - f_mid);
        const double w = std::max(0.0, std::min(down, up));
        filterbank[static_cast<std::size_t>(b) * static_cast<std::size_t>(kNumMels) + static_cast<std::size_t>(m)] = w;
      }
    }
  }
};

const MelTables& get_mel_tables() {
  static const MelTables kTables;
  return kTables;
}

// ---- CPU Conformer math primitives ----

inline float clamp_f32(float v, float lo, float hi) noexcept {
  return std::max(lo, std::min(hi, v));
}

inline float silu_f32(float x) noexcept {
  return x / (1.0f + std::exp(-x));
}

inline float softplus_f32(float x) noexcept {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

void rms_norm_rows(const float* x, const float* weight, int rows, int dim, float eps, float* out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double v = static_cast<double>(xr[d]);
      ss += v * v;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(dim) + static_cast<double>(eps)));
    if (weight != nullptr) {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv * weight[d];
    } else {
      for (int d = 0; d < dim; ++d) yr[d] = xr[d] * inv;
    }
  }
}

// Mean-centered LayerNorm without bias (nn.LayerNorm(C, eps, elementwise_affine=True, bias=False)) used by SSCP.
void layer_norm_no_bias(float* x, const float* weight, int rows, int dim, float eps) {
  for (int r = 0; r < rows; ++r) {
    float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
    double mean = 0.0;
    for (int d = 0; d < dim; ++d) mean += static_cast<double>(xr[d]);
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double diff = static_cast<double>(xr[d]) - mean;
      var += diff * diff;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(var / static_cast<double>(dim) + static_cast<double>(eps)));
    for (int d = 0; d < dim; ++d) {
      xr[d] = static_cast<float>(static_cast<double>(xr[d]) - mean) * inv * weight[d];
    }
  }
}

void linear_f32(const float* x, const float* w, const float* bias, int rows, int in_dim, int out_dim, float* out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim);
    float* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim);
    for (int o = 0; o < out_dim; ++o) {
      const float* wo = w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim);
      float acc = bias ? bias[o] : 0.0f;
      for (int k = 0; k < in_dim; ++k) acc += xr[k] * wo[k];
      yr[o] = acc;
    }
  }
}

void clipped_linear_f32(const float* x, const Eg2ClippedLinearF32& lin, bool use_clip, int rows, int in_dim, int out_dim,
                        float* out, std::vector<float>& scratch) {
  const float* src = x;
  if (use_clip) {
    const std::size_t n_in = static_cast<std::size_t>(rows) * static_cast<std::size_t>(in_dim);
    scratch.resize(n_in);
    for (std::size_t i = 0; i < n_in; ++i) scratch[i] = clamp_f32(x[i], lin.input_min, lin.input_max);
    src = scratch.data();
  }
  linear_f32(src, lin.weight.data(), nullptr, rows, in_dim, out_dim, out);
  if (use_clip) {
    const std::size_t n_out = static_cast<std::size_t>(rows) * static_cast<std::size_t>(out_dim);
    for (std::size_t i = 0; i < n_out; ++i) out[i] = clamp_f32(out[i], lin.output_min, lin.output_max);
  }
}

// Conv2d(k=3, stride=2, pad=1, bias=False) over NCHW input [C_in, H_in, W_in] -> NHWC output [H_out, W_out, C_out]
// followed by LayerNorm(C_out, no bias) + ReLU, returned as NCHW [C_out, H_out, W_out].
std::vector<float> sscp_conv_ln_relu(const float* in_chw, int C_in, int H_in, int W_in, const float* conv_w,
                                     const float* norm_w, int C_out, float eps, int& H_out, int& W_out) {
  H_out = (H_in + 2 - 3) / 2 + 1;
  W_out = (W_in + 2 - 3) / 2 + 1;
  std::vector<float> nhwc(static_cast<std::size_t>(H_out) * static_cast<std::size_t>(W_out) *
                              static_cast<std::size_t>(C_out),
                          0.0f);
  for (int oh = 0; oh < H_out; ++oh) {
    for (int ow = 0; ow < W_out; ++ow) {
      float* dst = nhwc.data() +
                   (static_cast<std::size_t>(oh) * static_cast<std::size_t>(W_out) + static_cast<std::size_t>(ow)) *
                       static_cast<std::size_t>(C_out);
      for (int oc = 0; oc < C_out; ++oc) {
        const float* w_oc = conv_w + static_cast<std::size_t>(oc) * static_cast<std::size_t>(C_in) * 9u;
        float acc = 0.0f;
        for (int ic = 0; ic < C_in; ++ic) {
          const float* in_ic = in_chw + static_cast<std::size_t>(ic) * static_cast<std::size_t>(H_in) *
                                            static_cast<std::size_t>(W_in);
          const float* w_ic = w_oc + static_cast<std::size_t>(ic) * 9u;
          for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * 2 + kh - 1;
            if (ih < 0 || ih >= H_in) continue;
            const float* in_row = in_ic + static_cast<std::size_t>(ih) * static_cast<std::size_t>(W_in);
            for (int kw = 0; kw < 3; ++kw) {
              const int iw = ow * 2 + kw - 1;
              if (iw < 0 || iw >= W_in) continue;
              acc += in_row[iw] * w_ic[kh * 3 + kw];
            }
          }
        }
        dst[oc] = acc;
      }
    }
  }
  layer_norm_no_bias(nhwc.data(), norm_w, H_out * W_out, C_out, eps);
  for (float& v : nhwc) v = std::max(0.0f, v);

  std::vector<float> out_chw(nhwc.size());
  for (int oh = 0; oh < H_out; ++oh) {
    for (int ow = 0; ow < W_out; ++ow) {
      const float* src = nhwc.data() +
                         (static_cast<std::size_t>(oh) * static_cast<std::size_t>(W_out) + static_cast<std::size_t>(ow)) *
                             static_cast<std::size_t>(C_out);
      for (int oc = 0; oc < C_out; ++oc) {
        out_chw[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(H_out) + static_cast<std::size_t>(oh)) *
                    static_cast<std::size_t>(W_out) +
                static_cast<std::size_t>(ow)] = src[oc];
      }
    }
  }
  return out_chw;
}

}  // namespace

ingine::model::EmbeddingAudio eg2_decode_wav_bytes(std::span<const std::uint8_t> wav_bytes) {
  if (wav_bytes.size() < 44 || std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::invalid_argument("EmbeddingGemma2 audio: input is not a valid RIFF/WAVE file");
  }

  bool have_fmt = false;
  bool have_data = false;
  std::uint16_t audio_format = 0;
  std::uint16_t num_channels = 0;
  std::uint32_t sample_rate = 0;
  std::uint16_t bits_per_sample = 0;
  const std::uint8_t* data_ptr = nullptr;
  std::size_t data_size = 0;

  std::size_t pos = 12;
  while (pos + 8 <= wav_bytes.size()) {
    const std::uint8_t* cid = wav_bytes.data() + pos;
    const std::uint32_t csz = read_le32(wav_bytes.data() + pos + 4);
    pos += 8;
    if (pos + csz > wav_bytes.size()) {
      throw std::invalid_argument("EmbeddingGemma2 audio: truncated WAV chunk");
    }
    if (std::memcmp(cid, "fmt ", 4) == 0) {
      if (csz < 16) throw std::invalid_argument("EmbeddingGemma2 audio: truncated WAV fmt chunk");
      audio_format = read_le16(wav_bytes.data() + pos + 0);
      num_channels = read_le16(wav_bytes.data() + pos + 2);
      sample_rate = read_le32(wav_bytes.data() + pos + 4);
      bits_per_sample = read_le16(wav_bytes.data() + pos + 14);
      have_fmt = true;
    } else if (std::memcmp(cid, "data", 4) == 0) {
      data_ptr = wav_bytes.data() + pos;
      data_size = csz;
      have_data = true;
    }
    pos += csz + (csz & 1u);
  }
  if (!have_fmt || !have_data) {
    throw std::invalid_argument("EmbeddingGemma2 audio: WAV missing fmt or data chunk");
  }
  if (num_channels == 0 || sample_rate == 0 || sample_rate > 192000) {
    throw std::invalid_argument("EmbeddingGemma2 audio: invalid WAV channels or sample rate");
  }
  if (!((audio_format == 1 && (bits_per_sample == 16 || bits_per_sample == 24 || bits_per_sample == 32)) ||
        (audio_format == 3 && bits_per_sample == 32))) {
    throw std::invalid_argument("EmbeddingGemma2 audio: unsupported WAV format (expected PCM 16/24/32 or IEEE float32)");
  }

  const std::size_t bytes_per_sample = bits_per_sample / 8;
  const std::size_t frame_bytes = bytes_per_sample * num_channels;
  if (frame_bytes == 0 || data_size < frame_bytes) {
    throw std::invalid_argument("EmbeddingGemma2 audio: WAV contains no audio frames");
  }
  const std::size_t n_frames = data_size / frame_bytes;
  std::vector<float> mono(n_frames, 0.0f);
  const float inv_ch = 1.0f / static_cast<float>(num_channels);

  for (std::size_t f = 0; f < n_frames; ++f) {
    const std::uint8_t* fp = data_ptr + f * frame_bytes;
    float acc = 0.0f;
    for (std::uint16_t ch = 0; ch < num_channels; ++ch) {
      const std::uint8_t* sp = fp + ch * bytes_per_sample;
      float s = 0.0f;
      if (audio_format == 1 && bits_per_sample == 16) {
        const std::int16_t v = static_cast<std::int16_t>(read_le16(sp));
        s = static_cast<float>(v) / 32768.0f;
      } else if (audio_format == 1 && bits_per_sample == 24) {
        std::int32_t v = static_cast<std::int32_t>(sp[0]) | (static_cast<std::int32_t>(sp[1]) << 8) |
                         (static_cast<std::int32_t>(sp[2]) << 16);
        if (v & 0x800000) v |= ~0xffffff;
        s = static_cast<float>(v) / 8388608.0f;
      } else if (audio_format == 1 && bits_per_sample == 32) {
        const std::int32_t v = static_cast<std::int32_t>(read_le32(sp));
        s = static_cast<float>(static_cast<double>(v) / 2147483648.0);
      } else {
        std::memcpy(&s, sp, 4);
        if (!std::isfinite(s)) s = 0.0f;
      }
      acc += s;
    }
    mono[f] = acc * inv_ch;
  }

  // Resample linearly to 16 kHz if needed (exact pass-through when already 16 kHz).
  ingine::model::EmbeddingAudio out;
  out.sample_rate = kTargetSampleRate;
  if (sample_rate == static_cast<std::uint32_t>(kTargetSampleRate)) {
    if (mono.size() > static_cast<std::size_t>(kMaxSamples)) mono.resize(kMaxSamples);
    out.samples = std::move(mono);
  } else {
    const double ratio = static_cast<double>(kTargetSampleRate) / static_cast<double>(sample_rate);
    const std::size_t out_n = std::min<std::size_t>(
        static_cast<std::size_t>(kMaxSamples),
        std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(static_cast<double>(mono.size()) * ratio))));
    out.samples.resize(out_n);
    const double inv_ratio = static_cast<double>(sample_rate) / static_cast<double>(kTargetSampleRate);
    for (std::size_t i = 0; i < out_n; ++i) {
      const double src_pos = static_cast<double>(i) * inv_ratio;
      const std::size_t idx0 = std::min(mono.size() - 1, static_cast<std::size_t>(src_pos));
      const std::size_t idx1 = std::min(mono.size() - 1, idx0 + 1);
      const float frac = static_cast<float>(src_pos - static_cast<double>(idx0));
      out.samples[i] = mono[idx0] * (1.0f - frac) + mono[idx1] * frac;
    }
  }
  return out;
}

ingine::model::EmbeddingAudio eg2_decode_wav_base64_or_data_uri(std::string_view input) {
  if (input.empty()) {
    throw std::invalid_argument("EmbeddingGemma2 audio: empty audio payload");
  }
  std::string_view b64 = input;
  if (input.rfind("data:", 0) == 0) {
    const std::size_t comma = input.find(',');
    if (comma == std::string_view::npos) {
      throw std::invalid_argument("EmbeddingGemma2 audio: malformed data: URI (missing ',')");
    }
    const std::string_view hdr = input.substr(0, comma);
    if (hdr.find(";base64") == std::string_view::npos) {
      throw std::invalid_argument("EmbeddingGemma2 audio: data: URI must be base64-encoded");
    }
    b64 = input.substr(comma + 1);
  }
  const std::vector<std::uint8_t> raw = decode_base64_bytes(b64);
  return eg2_decode_wav_bytes(raw);
}

Eg2PreprocessedAudio eg2_preprocess_audio(const ingine::model::EmbeddingAudio& audio) {
  if (audio.samples.empty()) {
    throw std::invalid_argument("EmbeddingGemma2 audio: waveform must not be empty");
  }
  const int orig_len = std::min<int>(static_cast<int>(audio.samples.size()), kMaxSamples);
  const int pad_right = (kPadMultiple - (orig_len % kPadMultiple)) % kPadMultiple;
  const int pad_left = kFrameLength / 2;  // 160
  const int total_len = pad_left + orig_len + pad_right;

  std::vector<double> w_pad(static_cast<std::size_t>(total_len), 0.0);
  std::vector<std::uint8_t> attn_mask(static_cast<std::size_t>(total_len), 0);
  for (int i = 0; i < orig_len; ++i) {
    w_pad[static_cast<std::size_t>(pad_left + i)] = static_cast<double>(audio.samples[static_cast<std::size_t>(i)]);
    attn_mask[static_cast<std::size_t>(pad_left + i)] = 1;
  }

  const int frame_size_for_unfold = kFrameLength + 1;  // 321
  if (total_len < frame_size_for_unfold) {
    throw std::invalid_argument("EmbeddingGemma2 audio: clip too short (<11ms / 161 samples at 16 kHz)");
  }
  const int num_frames = (total_len - frame_size_for_unfold) / kHopLength + 1;
  if (num_frames <= 0) {
    throw std::invalid_argument("EmbeddingGemma2 audio: clip produced 0 mel frames");
  }

  const MelTables& tbl = get_mel_tables();
  Eg2PreprocessedAudio out;
  out.mel_frames = num_frames;
  out.num_mels = kNumMels;
  out.mel_f32.assign(static_cast<std::size_t>(num_frames) * static_cast<std::size_t>(kNumMels), 0.0f);
  out.mel_mask.assign(static_cast<std::size_t>(num_frames), 0);

  std::array<double, kFftLength> fft_re{};
  std::array<double, kFftLength> fft_im{};
  std::array<double, kNumFreqBins> mag{};

  for (int f = 0; f < num_frames; ++f) {
    const int end_idx = f * kHopLength + frame_size_for_unfold - 1;
    const std::uint8_t valid = attn_mask[static_cast<std::size_t>(end_idx)];
    out.mel_mask[static_cast<std::size_t>(f)] = valid;
    if (!valid) continue;  // prepared_speech = speech * mask[..., None]

    const int base = f * kHopLength;
    fft_re.fill(0.0);
    fft_im.fill(0.0);
    for (int n = 0; n < kFrameLength; ++n) {
      fft_re[static_cast<std::size_t>(n)] =
          w_pad[static_cast<std::size_t>(base + n)] * static_cast<double>(tbl.window[static_cast<std::size_t>(n)]);
    }
    fft_radix2(fft_re.data(), fft_im.data(), kFftLength);
    for (int b = 0; b < kNumFreqBins; ++b) {
      mag[static_cast<std::size_t>(b)] =
          std::hypot(fft_re[static_cast<std::size_t>(b)], fft_im[static_cast<std::size_t>(b)]);
    }
    float* dst = out.mel_f32.data() + static_cast<std::size_t>(f) * static_cast<std::size_t>(kNumMels);
    for (int m = 0; m < kNumMels; ++m) {
      double acc = 0.0;
      for (int b = 0; b < kNumFreqBins; ++b) {
        acc += mag[static_cast<std::size_t>(b)] *
               tbl.filterbank[static_cast<std::size_t>(b) * static_cast<std::size_t>(kNumMels) +
                              static_cast<std::size_t>(m)];
      }
      dst[m] = static_cast<float>(std::log(acc + kMelFloor));
    }
  }

  // Two stride-2 Conv2d layers (kernel=3, pad=1): track exact subsampled mask.
  std::vector<std::uint8_t> m_cur = out.mel_mask;
  for (int s = 0; s < 2; ++s) {
    const int out_len = (static_cast<int>(m_cur.size()) + 2 - 3) / 2 + 1;
    std::vector<std::uint8_t> m_next(static_cast<std::size_t>(out_len), 0);
    for (int i = 0; i < out_len; ++i) {
      m_next[static_cast<std::size_t>(i)] = m_cur[static_cast<std::size_t>(i * 2)];
    }
    m_cur = std::move(m_next);
  }
  out.sscp_frames = static_cast<int>(m_cur.size());
  out.valid_mask = std::move(m_cur);
  int n_valid = 0;
  for (std::uint8_t v : out.valid_mask) n_valid += (v != 0) ? 1 : 0;
  if (n_valid <= 0) {
    throw std::invalid_argument("EmbeddingGemma2 audio: clip produced 0 valid audio soft tokens");
  }
  out.num_valid_tokens = n_valid;
  return out;
}

std::vector<float> eg2_build_rel_pos_embed_f32(const Eg2AudioConfig& cfg) {
  const int H = cfg.hidden_size;
  const int max_past = cfg.attention_context_left - 1;
  const int ctx_size = cfg.attention_chunk_size + max_past + cfg.attention_context_right;
  const int pos_len = ctx_size / 2 + 1;  // 13
  const int num_ts = H / 2;              // 512
  const double log_inc = std::log(10000.0) / static_cast<double>(std::max(num_ts - 1, 1));

  std::vector<double> inv_ts(static_cast<std::size_t>(num_ts));
  for (int i = 0; i < num_ts; ++i) {
    inv_ts[static_cast<std::size_t>(i)] = std::exp(-static_cast<double>(i) * log_inc);
  }
  std::vector<float> out(static_cast<std::size_t>(pos_len) * static_cast<std::size_t>(H));
  for (int p = 0; p < pos_len; ++p) {
    const double pid = static_cast<double>(ctx_size / 2 - p);  // 12, 11, ..., 0
    float* row = out.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(H);
    for (int i = 0; i < num_ts; ++i) {
      const double a = pid * inv_ts[static_cast<std::size_t>(i)];
      row[i] = static_cast<float>(std::sin(a));
      row[num_ts + i] = static_cast<float>(std::cos(a));
    }
  }
  return out;
}

Eg2CpuAudioEncoder::Eg2CpuAudioEncoder(Eg2AudioWeightsF32 weights)
    : weights_(std::move(weights)), rel_pos_embed_(eg2_build_rel_pos_embed_f32(weights_.config)) {}

std::vector<float> Eg2CpuAudioEncoder::encode_audio(const ingine::model::EmbeddingAudio& audio) const {
  const Eg2PreprocessedAudio prep = eg2_preprocess_audio(audio);
  const Eg2AudioConfig& ac = weights_.config;
  const int AH = ac.hidden_size;           // 1024
  const int n_heads = ac.num_attention_heads;  // 8
  const int hd = AH / n_heads;             // 128
  const int max_past = ac.attention_context_left - 1;  // 12
  const int ctx_size = ac.attention_chunk_size + max_past + ac.attention_context_right;
  const int pos_len = ctx_size / 2 + 1;    // 13
  const int OP = ac.output_proj_dims;      // 1536
  const int TH = weights_.text_hidden_size;  // 512
  const int K = ac.conv_kernel_size;       // 5
  const float eps = ac.rms_norm_eps;
  const float grad_clip = ac.gradient_clipping;
  const float res_w = ac.residual_weight;
  const float softcap = ac.attention_logit_cap;
  const bool use_clip = ac.use_clipped_linears;

  // 1. SSCP Conv2d layer0 [1, T_mel, 128] -> [128, T1, 64] -> mask -> layer1 -> [32, T2, 32]
  int T1 = 0, F1 = 0;
  std::vector<float> s0 =
      sscp_conv_ln_relu(prep.mel_f32.data(), 1, prep.mel_frames, prep.num_mels, weights_.sscp_layer0_conv.data(),
                        weights_.sscp_layer0_norm.data(), ac.sscp_conv_channels_0, eps, T1, F1);
  // Mask layer0 output by mel_mask[::2] before layer1
  for (int c = 0; c < ac.sscp_conv_channels_0; ++c) {
    for (int t = 0; t < T1; ++t) {
      if (prep.mel_mask[static_cast<std::size_t>(t * 2)] == 0) {
        float* row = s0.data() + (static_cast<std::size_t>(c) * T1 + t) * F1;
        std::fill(row, row + F1, 0.0f);
      }
    }
  }
  int T_seq = 0, F2 = 0;
  std::vector<float> s1 =
      sscp_conv_ln_relu(s0.data(), ac.sscp_conv_channels_0, T1, F1, weights_.sscp_layer1_conv.data(),
                        weights_.sscp_layer1_norm.data(), ac.sscp_conv_channels_1, eps, T_seq, F2);

  // Permute [C1, T_seq, F2] -> [T_seq, F2, C1] flattened to [T_seq, F2*C1=1024] -> input_proj_linear
  const int sscp_flat_dim = F2 * ac.sscp_conv_channels_1;
  std::vector<float> s1_tfc(static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(sscp_flat_dim));
  for (int t = 0; t < T_seq; ++t) {
    float* dst = s1_tfc.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(sscp_flat_dim);
    for (int f = 0; f < F2; ++f) {
      for (int c = 0; c < ac.sscp_conv_channels_1; ++c) {
        dst[f * ac.sscp_conv_channels_1 + c] =
            s1[(static_cast<std::size_t>(c) * T_seq + t) * F2 + f];
      }
    }
  }
  std::vector<float> hidden(static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(AH));
  linear_f32(s1_tfc.data(), weights_.sscp_input_proj_linear.data(), nullptr, T_seq, sscp_flat_dim, AH, hidden.data());

  // 2. 12 Conformer layers
  std::vector<float> norm_buf(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> ffw_mid(static_cast<std::size_t>(T_seq) * (4 * AH));
  std::vector<float> sub_out(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> q_buf(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> k_buf(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> v_buf(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> rel_k(static_cast<std::size_t>(pos_len) * AH);
  std::vector<float> attn_ctx(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> lconv_start(static_cast<std::size_t>(T_seq) * (2 * AH));
  std::vector<float> lconv_glu(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> lconv_dw(static_cast<std::size_t>(T_seq) * AH);
  std::vector<float> clip_scratch;

  const float q_scale = static_cast<float>((1.0 / std::sqrt(static_cast<double>(hd))) / std::log(2.0));
  const float k_scale = static_cast<float>(std::log(1.0 + std::exp(1.0)) / std::log(2.0));

  auto run_ffw = [&](const Eg2AudioFfwWeightsF32& ffw) {
    for (std::size_t i = 0; i < hidden.size(); ++i) norm_buf[i] = clamp_f32(hidden[i], -grad_clip, grad_clip);
    rms_norm_rows(norm_buf.data(), ffw.pre_layer_norm.data(), T_seq, AH, eps, norm_buf.data());
    clipped_linear_f32(norm_buf.data(), ffw.ffw_layer_1, use_clip, T_seq, AH, 4 * AH, ffw_mid.data(), clip_scratch);
    for (float& v : ffw_mid) v = silu_f32(v);
    clipped_linear_f32(ffw_mid.data(), ffw.ffw_layer_2, use_clip, T_seq, 4 * AH, AH, sub_out.data(), clip_scratch);
    for (float& v : sub_out) v = clamp_f32(v, -grad_clip, grad_clip);
    rms_norm_rows(sub_out.data(), ffw.post_layer_norm.data(), T_seq, AH, eps, sub_out.data());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += res_w * sub_out[i];
  };

  for (int li = 0; li < ac.num_hidden_layers; ++li) {
    const auto& lw = weights_.layers[static_cast<std::size_t>(li)];

    // FFW1
    run_ffw(lw.feed_forward1);

    // Self-Attention (causal local sliding window left=13, right=0 with relative pos bias & softcap=50.0)
    for (std::size_t i = 0; i < hidden.size(); ++i) norm_buf[i] = clamp_f32(hidden[i], -grad_clip, grad_clip);
    rms_norm_rows(norm_buf.data(), lw.norm_pre_attn.data(), T_seq, AH, eps, norm_buf.data());
    clipped_linear_f32(norm_buf.data(), lw.q_proj, use_clip, T_seq, AH, AH, q_buf.data(), clip_scratch);
    clipped_linear_f32(norm_buf.data(), lw.k_proj, use_clip, T_seq, AH, AH, k_buf.data(), clip_scratch);
    clipped_linear_f32(norm_buf.data(), lw.v_proj, use_clip, T_seq, AH, AH, v_buf.data(), clip_scratch);
    linear_f32(rel_pos_embed_.data(), lw.relative_k_proj.data(), nullptr, pos_len, AH, AH, rel_k.data());

    std::vector<float> q_dim_scale(static_cast<std::size_t>(hd));
    for (int d = 0; d < hd; ++d) q_dim_scale[static_cast<std::size_t>(d)] = q_scale * softplus_f32(lw.per_dim_scale[d]);
    for (int t = 0; t < T_seq; ++t) {
      float* qr = q_buf.data() + static_cast<std::size_t>(t) * AH;
      float* kr = k_buf.data() + static_cast<std::size_t>(t) * AH;
      for (int h = 0; h < n_heads; ++h) {
        for (int d = 0; d < hd; ++d) {
          qr[h * hd + d] *= q_dim_scale[static_cast<std::size_t>(d)];
          kr[h * hd + d] *= k_scale;
        }
      }
    }

    std::fill(attn_ctx.begin(), attn_ctx.end(), 0.0f);
    std::vector<float> logits(static_cast<std::size_t>(max_past + 1));
    std::vector<int> k_indices(static_cast<std::size_t>(max_past + 1));
    for (int t = 0; t < T_seq; ++t) {
      if (prep.valid_mask[static_cast<std::size_t>(t)] == 0) continue;
      const int k_lo = std::max(0, t - max_past);
      for (int h = 0; h < n_heads; ++h) {
        const float* qp = q_buf.data() + static_cast<std::size_t>(t) * AH + h * hd;
        int n_k = 0;
        float max_s = -std::numeric_limits<float>::infinity();
        for (int kk = k_lo; kk <= t; ++kk) {
          if (prep.valid_mask[static_cast<std::size_t>(kk)] == 0) continue;
          const int rel_idx = max_past - (t - kk);
          const float* kp = k_buf.data() + static_cast<std::size_t>(kk) * AH + h * hd;
          const float* rp = rel_k.data() + static_cast<std::size_t>(rel_idx) * AH + h * hd;
          float dot = 0.0f;
          for (int d = 0; d < hd; ++d) dot += qp[d] * (kp[d] + rp[d]);
          const float s = std::tanh(dot / softcap) * softcap;
          logits[static_cast<std::size_t>(n_k)] = s;
          k_indices[static_cast<std::size_t>(n_k)] = kk;
          if (s > max_s) max_s = s;
          ++n_k;
        }
        double sum_e = 0.0;
        for (int j = 0; j < n_k; ++j) {
          const double e = std::exp(static_cast<double>(logits[static_cast<std::size_t>(j)] - max_s));
          logits[static_cast<std::size_t>(j)] = static_cast<float>(e);
          sum_e += e;
        }
        const float inv_e = static_cast<float>(1.0 / sum_e);
        float* op = attn_ctx.data() + static_cast<std::size_t>(t) * AH + h * hd;
        for (int j = 0; j < n_k; ++j) {
          const float w_a = logits[static_cast<std::size_t>(j)] * inv_e;
          const float* vp = v_buf.data() + static_cast<std::size_t>(k_indices[static_cast<std::size_t>(j)]) * AH + h * hd;
          for (int d = 0; d < hd; ++d) op[d] += w_a * vp[d];
        }
      }
    }
    clipped_linear_f32(attn_ctx.data(), lw.post, use_clip, T_seq, AH, AH, sub_out.data(), clip_scratch);
    for (float& v : sub_out) v = clamp_f32(v, -grad_clip, grad_clip);
    rms_norm_rows(sub_out.data(), lw.norm_post_attn.data(), T_seq, AH, eps, sub_out.data());
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += sub_out[i];

    // LightConv1d (pre_layer_norm -> linear_start -> GLU -> causal depthwise Conv1d k=5 -> conv_norm -> SiLU -> linear_end)
    rms_norm_rows(hidden.data(), lw.lconv_pre_layer_norm.data(), T_seq, AH, eps, norm_buf.data());
    clipped_linear_f32(norm_buf.data(), lw.lconv_linear_start, use_clip, T_seq, AH, 2 * AH, lconv_start.data(),
                       clip_scratch);
    for (int t = 0; t < T_seq; ++t) {
      const float* sr = lconv_start.data() + static_cast<std::size_t>(t) * (2 * AH);
      float* dr = lconv_glu.data() + static_cast<std::size_t>(t) * AH;
      for (int d = 0; d < AH; ++d) {
        const float sig = 1.0f / (1.0f + std::exp(-sr[AH + d]));
        dr[d] = sr[d] * sig;
      }
    }
    const int left_pad = K - 1;
    for (int t = 0; t < T_seq; ++t) {
      float* dr = lconv_dw.data() + static_cast<std::size_t>(t) * AH;
      for (int d = 0; d < AH; ++d) {
        const float* dw_w = lw.lconv_depthwise_conv1d.data() + static_cast<std::size_t>(d) * K;
        float acc = 0.0f;
        for (int ki = 0; ki < K; ++ki) {
          const int ti = t - left_pad + ki;
          if (ti >= 0) acc += lconv_glu[static_cast<std::size_t>(ti) * AH + d] * dw_w[ki];
        }
        dr[d] = clamp_f32(acc, -grad_clip, grad_clip);
      }
    }
    rms_norm_rows(lconv_dw.data(), lw.lconv_conv_norm.data(), T_seq, AH, eps, lconv_dw.data());
    for (float& v : lconv_dw) v = silu_f32(v);
    clipped_linear_f32(lconv_dw.data(), lw.lconv_linear_end, use_clip, T_seq, AH, AH, sub_out.data(), clip_scratch);
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += sub_out[i];

    // FFW2
    run_ffw(lw.feed_forward2);

    // norm_out
    for (float& v : hidden) v = clamp_f32(v, -grad_clip, grad_clip);
    rms_norm_rows(hidden.data(), lw.norm_out.data(), T_seq, AH, eps, hidden.data());
  }

  // 3. output_proj [1536, 1024] + bias -> unscaled RMSNorm -> embed_audio [512, 1536] -> compact valid rows
  std::vector<float> aud_proj(static_cast<std::size_t>(T_seq) * OP);
  linear_f32(hidden.data(), weights_.output_proj_weight.data(), weights_.output_proj_bias.data(), T_seq, AH, OP,
             aud_proj.data());
  rms_norm_rows(aud_proj.data(), nullptr, T_seq, OP, eps, aud_proj.data());

  std::vector<float> valid_proj(static_cast<std::size_t>(prep.num_valid_tokens) * OP);
  int dst_idx = 0;
  for (int t = 0; t < T_seq; ++t) {
    if (prep.valid_mask[static_cast<std::size_t>(t)] != 0) {
      std::memcpy(valid_proj.data() + static_cast<std::size_t>(dst_idx) * OP,
                  aud_proj.data() + static_cast<std::size_t>(t) * OP, static_cast<std::size_t>(OP) * sizeof(float));
      ++dst_idx;
    }
  }
  std::vector<float> soft_out(static_cast<std::size_t>(prep.num_valid_tokens) * TH);
  linear_f32(valid_proj.data(), weights_.embed_audio_projection.data(), nullptr, prep.num_valid_tokens, OP, TH,
             soft_out.data());
  return soft_out;
}

}  // namespace ingine::backends::embedding_gemma2
