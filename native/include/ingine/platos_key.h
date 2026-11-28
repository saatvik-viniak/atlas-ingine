// Modular Plato's Key loader and projector (CPU + optional CUDA GPU).
//
// Supports both `platos_key_v1` safetensors (`R [d_in, d_out]`, optional `donor_mean [d_in]`,
// `target_mean [d_out]`, `scale [1]`) and affine safetensors (`W [d_in, d_out]`, optional `b [d_out]`),
// plus optional external JSON metadata (path or JSON text) so hot-swappable cartridge racks can
// attach and validate cartridge manifests without duplicating loader logic.
//
// Formula (row-major x [B, d_in] -> y [B, d_out]):
//   y = target_mean + scale * ((x - donor_mean) @ R)
//
// This header is CUDA-header-free so CPU-only targets (`ingine_engine`, `ingine_serve`, host tests)
// can share the exact same types and `.safetensors` validation as CUDA backends.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ingine {

struct PlatosKeyMetadata {
  std::string format;          // e.g. "platos_key_v1", "platos_key_v2", "platos_key_svd_v1", "affine_wb", "omni946-inkling-yue2-key/v1"
  std::string tensor_layout;   // "ridge_R", "svd_Ur_Vr", "affine_W_yue2", or "affine_W"
  std::string schema_version;  // optional schema_version string
  std::string cartridge_id;    // optional #964 cartridge_id
  std::string file_sha256;     // lowercase 64-hex SHA-256 computed over exact loaded safetensors bytes
  std::string donor_model;     // optional donor_model_id / donor_model / source_model
  std::string target_model;    // optional target_model
  std::string tap_spec;        // optional tap_spec (e.g. "layer_31", "visit_23", "final_normed")
  std::string method;          // optional solver/method (e.g. "ridge", "ridge_norm", "procrustes_var_match")
  std::string weight_dtype;    // safetensors dtype of primary projection tensor ("F16", "BF16", "F32")
  std::optional<int> expected_tap_layer;  // bound single-tap index from sidecar tap_layer or layer_N/visit_N tap_spec
  bool requires_final_layer = false;      // true when tap_spec binds "final_normed" / "final"
  bool is_multi_tap = false;              // true when tap_spec declares multi-tap concat (rejected on single-tap servers)
  bool forbidden_coord_d2027_zero_masked = false;  // Inkling-2-Mini: zero x[2027] before centering/projection
  int svd_rank = 0;                       // > 0 for factorized svd_Ur_Vr keys
  int d_in = 0;
  int d_out = 0;
  float scale = 1.0f;
  std::map<std::string, std::string> raw_metadata;  // merged safetensors __metadata__ + JSON metadata fields
  std::string json_metadata;                        // optional raw sidecar JSON metadata text
};

// Immutable CPU-resident Plato's Key weights and metadata. Safe to share across threads and
// hot-swap inside a cartridge rack.
class PlatosKey {
 public:
  // Loads a Plato's Key `.safetensors` file. If `expected_in_dim > 0`, fails closed on mismatch.
  static std::shared_ptr<const PlatosKey> load(const std::filesystem::path& safetensors_path,
                                               int expected_in_dim = 0);

  // Loads a Plato's Key `.safetensors` file together with an optional sidecar JSON metadata file.
  // If `metadata_json_path` is empty, behaves identically to the 2-argument overload.
  static std::shared_ptr<const PlatosKey> load(const std::filesystem::path& safetensors_path,
                                               const std::filesystem::path& metadata_json_path,
                                               int expected_in_dim = 0);

  // Loads a Plato's Key `.safetensors` file together with optional inline JSON metadata text.
  static std::shared_ptr<const PlatosKey> load_with_json(const std::filesystem::path& safetensors_path,
                                                         std::string_view metadata_json,
                                                         int expected_in_dim = 0);

  [[nodiscard]] int in_dim() const noexcept { return meta_.d_in; }
  [[nodiscard]] int out_dim() const noexcept { return meta_.d_out; }
  [[nodiscard]] float scale() const noexcept { return meta_.scale; }
  [[nodiscard]] bool has_donor_mean() const noexcept { return has_donor_mean_; }
  [[nodiscard]] bool has_target_mean() const noexcept { return has_target_mean_; }
  [[nodiscard]] const PlatosKeyMetadata& metadata() const noexcept { return meta_; }
  [[nodiscard]] const std::filesystem::path& source_path() const noexcept { return source_path_; }

  // Row-major R [d_in, d_out] in FP32.
  [[nodiscard]] const std::vector<float>& weights_f32() const noexcept { return r_f32_; }
  // Transposed R^T [d_out, d_in] in IEEE-754 binary16 bits (for coalesced GPU upload).
  [[nodiscard]] const std::vector<std::uint16_t>& weights_f16_transposed() const noexcept { return rt_f16_; }
  // FP32 donor_mean [d_in] (zeros when absent in file).
  [[nodiscard]] const std::vector<float>& donor_mean_f32() const noexcept { return donor_mean_; }
  // FP32 target_mean [d_out] (zeros when absent in file).
  [[nodiscard]] const std::vector<float>& target_mean_f32() const noexcept { return target_mean_; }

  // Projects `in_x` [batch, d_in] -> `out_y` [batch, d_out] on CPU with FP64 dot accumulation.
  // `in_x` and `out_y` may alias only if d_in == d_out (handled via internal staging when aliased).
  void project_cpu(const float* in_x, int batch, float* out_y, bool l2_normalize = false) const;

  // Fail-closed runtime binding checks (startup donor/dim/tap-range/layout check + per-request tap/cartridge checks).
  void validate_binding(std::string_view active_model_id, std::string_view active_model_family,
                        int active_embedding_dim, int active_num_tap_layers) const;
  void validate_request_tap(std::optional<int> requested_tap_layer) const;
  void validate_request_identity(std::string_view requested_cartridge_id,
                                 std::string_view requested_cartridge_sha256) const;

 private:
  PlatosKey() = default;

  std::filesystem::path source_path_;
  PlatosKeyMetadata meta_;
  bool has_donor_mean_ = false;
  bool has_target_mean_ = false;
  std::vector<float> r_f32_;             // [d_in, d_out]
  std::vector<std::uint16_t> rt_f16_;    // [d_out, d_in] IEEE FP16
  std::vector<float> donor_mean_;        // [d_in]
  std::vector<float> target_mean_;       // [d_out]
};

// In-place row-wise L2 normalization on CPU (zero rows remain zero).
void l2_normalize_rows_cpu(float* data, int batch, int dim) noexcept;

#if defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
// Immutable GPU-resident Plato's Key projector. `cuda_stream` is a `cudaStream_t` passed as `void*`
// so this public header remains CUDA-header-free.
class PlatosKeyGpu {
 public:
  ~PlatosKeyGpu();
  PlatosKeyGpu(const PlatosKeyGpu&) = delete;
  PlatosKeyGpu& operator=(const PlatosKeyGpu&) = delete;

  static std::shared_ptr<const PlatosKeyGpu> create(std::shared_ptr<const PlatosKey> key,
                                                    void* cuda_stream = nullptr);
  static std::shared_ptr<const PlatosKeyGpu> load(const std::filesystem::path& safetensors_path,
                                                  int expected_in_dim = 0, void* cuda_stream = nullptr);
  static std::shared_ptr<const PlatosKeyGpu> load(const std::filesystem::path& safetensors_path,
                                                  const std::filesystem::path& metadata_json_path,
                                                  int expected_in_dim = 0, void* cuda_stream = nullptr);

  [[nodiscard]] const std::shared_ptr<const PlatosKey>& key() const noexcept { return key_; }
  [[nodiscard]] const PlatosKeyMetadata& metadata() const noexcept { return key_->metadata(); }
  [[nodiscard]] int in_dim() const noexcept { return key_->in_dim(); }
  [[nodiscard]] int out_dim() const noexcept { return key_->out_dim(); }
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  // Projects device `d_in_x` [batch, d_in] (FP32) -> `d_out_y` [batch, d_out] (FP32) on `cuda_stream`.
  // `d_in_x` and `d_out_y` must not overlap.
  void project(const float* d_in_x, int batch, float* d_out_y, bool l2_normalize,
               void* cuda_stream = nullptr) const;

 private:
  struct Impl;
  PlatosKeyGpu(std::shared_ptr<const PlatosKey> key, std::unique_ptr<Impl> impl);

  std::shared_ptr<const PlatosKey> key_;
  std::unique_ptr<Impl> impl_;
};

// Shared GPU pooling + optional L2-normalization kernels used by all embedding backends.
// `pooling_mode`: 0 = Mean, 1 = LastToken, 2 = Cls.
// `d_seq_offsets`: int32 [batch + 1] device prefix sums (token offsets into packed `[T, dim]`).
void embed_pool_f32_cuda(const float* d_hidden, const std::int32_t* d_seq_offsets, int batch, int dim,
                         int pooling_mode, bool l2_normalize, float* d_out, void* cuda_stream = nullptr);
void embed_pool_act_cuda(const std::uint16_t* d_hidden_act, bool is_fp16, const std::int32_t* d_seq_offsets,
                         int batch, int dim, int pooling_mode, bool l2_normalize, float* d_out,
                         void* cuda_stream = nullptr);
void embed_l2_normalize_f32_cuda(float* d_data, int batch, int dim, void* cuda_stream = nullptr);
#endif

}  // namespace ingine
