// Pure-CPU FP32 text encoder backend for google/embeddinggemma-2 (#968).
// Runs identically on CUDA-free builds, desktop/mobile routers, and CPU unit parity tests.
#pragma once

#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "ingine/generator.h"
#include "ingine/model_backend.h"
#include "ingine/platos_key.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ingine::backends::embedding_gemma2 {

// Pure CPU FP32 encoder kernel over loaded Eg2TextWeightsF32.
// Computes per-sequence pooled embeddings (with optional early-exit tap_layer, optional
// embedding_projection 512->768 at full depth, optional Matryoshka truncation, and FP32 L2 norm).
class Eg2CpuEncoder {
 public:
  explicit Eg2CpuEncoder(Eg2TextWeightsF32 weights);

  const Eg2Config& config() const noexcept { return weights_.config; }

  // Forward a single token sequence [T] -> hidden [T, H] (after final norm if tap_layer == nullopt,
  // else after layer `*tap_layer` [0..num_hidden_layers-1]).
  std::vector<float> encode_hidden(std::span<const std::int32_t> token_ids,
                                   std::optional<int> tap_layer = std::nullopt) const;

  // Full batched embedding forward matching Generator::embed() semantics.
  ingine::model::EmbeddingResult embed(const ingine::model::EmbeddingRequest& req,
                                       const PlatosKey* platos_key = nullptr) const;

 private:
  Eg2TextWeightsF32 weights_;
};

// ModelBackend wrapper exposing Eg2CpuEncoder through ingine::model::Generator (embeddings-only).
class Eg2CpuBackend final : public ingine::model::ModelBackend {
 public:
  explicit Eg2CpuBackend(const ingine::model::RuntimeOptions& options);
  explicit Eg2CpuBackend(Eg2TextWeightsF32 weights, int max_ctx = 8192,
                         std::shared_ptr<const PlatosKey> platos_key = nullptr);

  // Generator interface
  using ingine::model::Generator::generate;
  std::string model_id() const override { return model_id_; }
  int max_model_len() const override { return max_ctx_; }
  ingine::model::GenerateStats generate(std::span<const std::int32_t>, int, const ingine::model::Sampling&,
                                        std::span<const std::int32_t>,
                                        const std::function<bool(std::span<const std::int32_t>)>&) override;
  bool supports_images() const override { return false; }
  bool supports_audio() const override { return false; }
  bool is_encoder_only() const override { return true; }
  bool supports_logit_masker() const override { return false; }
  std::string model_family() const override { return "embedding_gemma2"; }
  ingine::model::KvCacheMode kv_cache_mode() const override { return ingine::model::KvCacheMode::PerPass; }
  std::vector<ingine::model::KvCacheMode> supported_kv_cache_modes() const override {
    return {ingine::model::KvCacheMode::PerPass};
  }
  int loop_passes() const override { return 1; }
  ingine::model::LoopCdMode loopcd_mode() const override { return ingine::model::LoopCdMode::Off; }

  bool supports_embeddings() const override { return true; }
  int embedding_dim() const override { return encoder_.config().embedding_dim; }
  int num_tap_layers() const override { return encoder_.config().num_hidden_layers; }
  std::vector<int> matryoshka_dims() const override { return encoder_.config().matryoshka_dims; }
  std::map<std::string, std::string> embedding_prompt_prefixes() const override { return encoder_.config().prompts; }
  bool has_platos_key() const override { return platos_key_ != nullptr; }
  int platos_key_out_dim() const override { return platos_key_ ? platos_key_->out_dim() : 0; }
  std::string platos_key_tap_spec() const override { return platos_key_ ? platos_key_->metadata().tap_spec : ""; }
  ingine::model::EmbeddingResult embed(const ingine::model::EmbeddingRequest& req) override;

  // ModelBackend interface
  const ingine::model::BackendCapabilities& capabilities() const override { return caps_; }
  const ingine::model::WorkerProfile& worker_profile() const override { return profile_; }
  const ingine::model::BackendStateIdentity& state_identity() const override { return identity_; }
  void reset_session() override {}

  const Eg2Config& config() const noexcept { return encoder_.config(); }

 private:
  void init_metadata();

  Eg2CpuEncoder encoder_;
  std::shared_ptr<const PlatosKey> platos_key_;
  std::string model_id_ = "embeddinggemma-2";
  int max_ctx_ = 8192;
  ingine::model::BackendCapabilities caps_;
  ingine::model::WorkerProfile profile_;
  ingine::model::BackendStateIdentity identity_;
};

// Registers EmbeddingGemma2 CPU backend (no-op if already registered by CUDA backend).
void register_eg2_cpu_backend();

}  // namespace ingine::backends::embedding_gemma2
