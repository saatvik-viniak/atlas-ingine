// CUDA text encoder backend for google/embeddinggemma-2 (#968).
//
// Precision policy (strictly enforced — FP16 activations are forbidden for EmbeddingGemma 2):
//   - sm_80+ (Ampere/Ada/Hopper/Blackwell, !INGINE_ACT_FP16): BF16 weights + BF16 GEMMs with FP32
//     cuBLASLt accumulation (`CUDA_R_16BF` / `CUBLAS_COMPUTE_32F`), FP32 norms/RoPE/softmax/pooling/L2.
//   - sm_70 (Volta V100, INGINE_ACT_FP16 == 1): native FP32 weights + FP32 cuBLASLt GEMMs
//     (`CUDA_R_32F` / `CUBLAS_COMPUTE_32F`), zero FP16 casts anywhere.
#pragma once

#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_manifest.h"
#include "ingine/generator.h"
#include "ingine/model_backend.h"
#include "ingine/platos_key.h"

#include <memory>
#include <string>

namespace ingine::backends::embedding_gemma2 {

class Eg2CudaBackend final : public ingine::model::ModelBackend {
 public:
  explicit Eg2CudaBackend(const ingine::model::RuntimeOptions& options);
  ~Eg2CudaBackend() override;

  // Generator interface
  using ingine::model::Generator::generate;
  std::string model_id() const override { return model_id_; }
  int max_model_len() const override { return max_ctx_; }
  ingine::model::GenerateStats generate(std::span<const std::int32_t>, int, const ingine::model::Sampling&,
                                        std::span<const std::int32_t>,
                                        const std::function<bool(std::span<const std::int32_t>)>&) override;
  bool supports_images() const override { return has_vision_; }
  bool supports_audio() const override { return has_audio_; }
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
  int embedding_dim() const override { return config_.embedding_dim; }
  int num_tap_layers() const override { return config_.num_hidden_layers; }
  std::vector<int> matryoshka_dims() const override { return config_.matryoshka_dims; }
  std::map<std::string, std::string> embedding_prompt_prefixes() const override { return config_.prompts; }
  bool has_platos_key() const override { return platos_key_ != nullptr; }
  int platos_key_out_dim() const override { return platos_key_ ? platos_key_->out_dim() : 0; }
  std::string platos_key_tap_spec() const override { return platos_key_ ? platos_key_->metadata().tap_spec : ""; }
  std::shared_ptr<const PlatosKey> platos_key() const override { return platos_key_; }
  ingine::model::EmbeddingResult embed(const ingine::model::EmbeddingRequest& req) override;

  // ModelBackend interface
  const ingine::model::BackendCapabilities& capabilities() const override { return caps_; }
  const ingine::model::WorkerProfile& worker_profile() const override { return profile_; }
  const ingine::model::BackendStateIdentity& state_identity() const override { return identity_; }
  void reset_session() override {}
  ingine::model::ResidencyReport residency_report() const override;

  const Eg2Config& config() const noexcept { return config_; }
  bool uses_bf16() const noexcept { return use_bf16_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Eg2Config config_;
  std::shared_ptr<const PlatosKey> platos_key_;
  std::string model_id_ = "embeddinggemma-2";
  int max_ctx_ = 8192;
  bool use_bf16_ = false;
  bool has_vision_ = false;
  bool has_audio_ = false;
  ingine::model::BackendCapabilities caps_;
  ingine::model::WorkerProfile profile_;
  ingine::model::BackendStateIdentity identity_;
};

// Registers EmbeddingGemma2 CUDA backend into BackendRegistry.
void register_eg2_cuda_backend();

}  // namespace ingine::backends::embedding_gemma2
