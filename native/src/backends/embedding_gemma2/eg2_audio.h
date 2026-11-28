// EmbeddingGemma 2 Audio Conformer preprocessing (WAV decode + 128-bin HTK log-mel STFT)
// and pure-CPU FP32 Conformer encoder (#968 Phase 3).
#pragma once

#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "ingine/generator.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ingine::backends::embedding_gemma2 {

struct Eg2PreprocessedAudio {
  int mel_frames = 0;         // number of 10ms log-mel frames
  int num_mels = 128;         // 128 HTK mel bins
  int sscp_frames = 0;        // frames after 2x stride-2 SSCP Conv2d
  int num_valid_tokens = 0;   // sum(valid_mask) = exact <|audio|> placeholder count
  std::vector<float> mel_f32; // [mel_frames, 128] row-major, masked padding zeroed
  std::vector<std::uint8_t> mel_mask;   // [mel_frames] 1 = valid, 0 = pad
  std::vector<std::uint8_t> valid_mask; // [sscp_frames] 1 = valid, 0 = pad
};

// Decodes standard RIFF/WAVE bytes (PCM16/PCM24/PCM32/IEEE float32, mono or multi-channel)
// into 16 kHz mono float32 samples normalized to [-1, 1] (capped at 30.0 s = 480,000 samples).
ingine::model::EmbeddingAudio eg2_decode_wav_bytes(std::span<const std::uint8_t> wav_bytes);

// Decodes either raw base64 WAV or a `data:audio/...;base64,...` URI into 16 kHz mono float32 samples.
ingine::model::EmbeddingAudio eg2_decode_wav_base64_or_data_uri(std::string_view input);

// Exact mirror of Gemma4AudioFeatureExtractor (16 kHz, pad_to_multiple_of=128, pad_left=160,
// frame_length=320, hop_length=160, periodic Hann, 512-pt RFFT magnitude, 128-bin HTK mel, log(x + 0.001)).
Eg2PreprocessedAudio eg2_preprocess_audio(const ingine::model::EmbeddingAudio& audio);

// Computes sinusoidal relative positional embeddings [context_size/2 + 1, hidden_size] (13 x 1024).
std::vector<float> eg2_build_rel_pos_embed_f32(const Eg2AudioConfig& cfg);

// Pure-CPU FP32 Audio Conformer + embed_audio encoder.
// Outputs contiguous [num_valid_tokens, text_hidden_size=512] FP32 soft tokens.
class Eg2CpuAudioEncoder {
 public:
  explicit Eg2CpuAudioEncoder(Eg2AudioWeightsF32 weights);

  const Eg2AudioConfig& config() const noexcept { return weights_.config; }
  std::vector<float> encode_audio(const ingine::model::EmbeddingAudio& audio) const;

 private:
  Eg2AudioWeightsF32 weights_;
  std::vector<float> rel_pos_embed_;
};

}  // namespace ingine::backends::embedding_gemma2
