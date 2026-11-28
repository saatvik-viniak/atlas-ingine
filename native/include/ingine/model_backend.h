// Shared host/model-family contract for atlas-ingine (#821, #822, #833, #834).
//
// Strictly CUDA-free public header extending ingine::model::Generator without altering existing Inkling callers
// or continuous-batching slot semantics (native/include/ingine/generator.h remains unchanged).
//
// Ownership split:
//   * Host (ingine_engine / ingine_serve): selects pinned ModelBackend / WorkerProfile, tokenizer & chat template,
//     advertised capabilities, request admission, scheduling, and cancellation.
//   * ModelBackend (native/src/backends/<family>/): owns weights, prefill, decode, opaque per-session state,
//     exact checkpoint/state cache identities, and exact token-prefix checks.
//   * HAL (core/hal.h, core/ops.h, #835): owns device placement, buffers, streams, events, and device ops.
#pragma once

#include "ingine/generator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::model {

enum class ModelFamily : std::uint8_t {
  Unknown = 0,
  Inkling2 = 1,
  Qwen3_5 = 2,
  DeepSeekV4 = 3,
  GlmFlash = 4,
  Qwen4Exp = 5,
  EmbeddingGemma2 = 6,
};

enum class CheckpointFormat : std::uint8_t {
  Unknown = 0,
  Safetensors = 1,
  Gguf = 2,
};

enum class ActivationDtype : std::uint8_t {
  Unknown = 0,
  Bf16 = 1,
  Fp16 = 2,
  Fp32 = 3,
};

enum class VerificationLevel : std::uint8_t {
  Unverified = 0,
  HeadersOnly = 1,     // Safetensors/GGUF headers inspected without payload SHA-256 consumption
  SelfHashed = 2,      // Payload SHA-256 computed locally without external expected pin verification
  PinnedVerified = 3,  // Full shard payloads verified byte-for-byte against an expected CheckpointPinSpec
};

const char* to_string(ModelFamily family) noexcept;
const char* to_string(CheckpointFormat fmt) noexcept;
const char* to_string(ActivationDtype dt) noexcept;
const char* to_string(VerificationLevel level) noexcept;
ModelFamily parse_model_family(std::string_view s);
// Parses a 64-char hex SHA-256 digest string; throws std::invalid_argument fail-closed on malformed input.
std::array<std::uint8_t, 32> parse_sha256_hex(std::string_view hex);

// Detects model family from a checkpoint directory (config.json, GGUF general.architecture, or Inkling
// tokenizer markers) or direct backbone .gguf file using exact supported family tags.
// Throws fail-closed on unknown/legacy (e.g. V3/GLM-4) or auxiliary GGUF (dflash/clip).
ModelFamily detect_model_family(const std::filesystem::path& model_path);
// Dispatches container format ONLY (Safetensors vs GGUF) from a checkpoint directory or direct shard path;
// throws std::invalid_argument fail-closed on missing, empty, or ambiguous paths. Multi-shard stem/contiguous
// split-metadata discovery and expected-vs-consumed per-file SHA-256 verification belong in family loaders.
CheckpointFormat detect_checkpoint_format(const std::filesystem::path& model_path);

struct ExpectedShardDigest {
  std::string filename;
  std::uint64_t byte_size = 0;
  std::array<std::uint8_t, 32> sha256{};
};

// External expected cryptographic pin specification used by inspect_pinned() loaders.
struct CheckpointPinSpec {
  std::vector<ExpectedShardDigest> shards;
  std::optional<std::array<std::uint8_t, 32>> config_sha256;
  std::optional<std::array<std::uint8_t, 32>> tokenizer_sha256;
  std::optional<std::array<std::uint8_t, 32>> chat_template_sha256;
};

// Cryptographic identity of the checkpoint artifacts and verified loaded weights.
// weight_manifest_verified MUST be true only when headers/manifests were explicitly verified at load time.
// shard_digests_verified MUST ONLY be true when verification_level == VerificationLevel::PinnedVerified
// (i.e., expected-vs-consumed match against an external CheckpointPinSpec; self-hashing never sets it).
struct CheckpointIdentity {
  CheckpointFormat format = CheckpointFormat::Unknown;
  VerificationLevel verification_level = VerificationLevel::Unverified;
  std::string schema_variant;
  std::string model_id;
  std::string revision;
  std::uint32_t shard_count = 0;
  std::array<std::uint8_t, 32> shard_set_sha256{};
  std::array<std::uint8_t, 32> config_sha256{};
  std::array<std::uint8_t, 32> tokenizer_sha256{};
  std::array<std::uint8_t, 32> chat_template_sha256{};
  std::array<std::uint8_t, 32> weight_manifest_sha256{};
  bool weight_manifest_verified = false;
  bool shard_digests_verified = false;

  bool operator==(const CheckpointIdentity&) const = default;
};

// Advertised backend capabilities checked fail-closed before mutating live model state.
struct BackendCapabilities {
  ModelFamily family = ModelFamily::Unknown;
  std::string backend_name;
  std::string architecture;
  bool supports_text = true;
  bool supports_images = false;
  bool supports_audio = false;
  bool supports_speculation = false;
  bool supports_continuous_batching = false;
  int max_supported_batch = 1;
  bool supports_prefix_cache = false;
  bool supports_host_park = false;
  bool supports_weight_parking = false;
  bool supports_tiered_residency = false;
  bool supports_disk_checkpoint = false;
  bool supports_persona_hooks = false;
  bool supports_ablation = false;
  bool supports_logit_capture = false;
  bool supports_mtp = false;
  std::vector<std::string> supported_gears;
  std::vector<std::string> supported_kv_cache_modes;
  std::vector<std::string> supported_quant_layouts;
  std::vector<std::int32_t> media_token_ids;  // positive/special media placeholder IDs rejected when !supports_images
  int vocab_size = 0;
  int max_position_embeddings = 0;
};

// Complete runtime/model/profile identity required for prefix-cache or session-state reuse across all families.
// Any mismatch MUST fail closed to a cache miss.
struct BackendStateIdentity {
  ModelFamily family = ModelFamily::Unknown;
  std::string backend_name;
  CheckpointIdentity checkpoint;
  std::string state_layout_id;  // per-family KV/recurrent/conv state layout version
  ActivationDtype act_dtype = ActivationDtype::Unknown;  // effective compute numeric ABI (distinct Bf16/Fp16)
  std::string numeric_abi;      // canonical execution numeric options string (GEMM/dequant/accum/KV)
  std::string weight_precision;
  std::string kv_format;
  std::string kv_cache_mode;
  std::string gear;
  int max_ctx = 0;
  int prefill_chunk = 0;
  bool spec_enabled = false;
  bool mtp_enabled = false;
  bool media_enabled = false;
  std::array<std::uint8_t, 32> media_sha256{};
  std::uint64_t persona_key = 0;
  std::string persona_fingerprint;
  std::string cache_namespace;
  std::string rendering_fingerprint;
  std::string family_state_fingerprint;

  bool operator==(const BackendStateIdentity&) const = default;

  // True iff both identities carry non-empty required metadata, VerificationLevel::PinnedVerified,
  // non-zero manifest & shard_set digests, and exact equality across every execution/profile field.
  [[nodiscard]] bool is_reusable_with(const BackendStateIdentity& other) const noexcept;
};

// Index hash ONLY (never used for cache equality without BackendStateIdentity::is_reusable_with() and
// exact prefix verification).
std::uint64_t hash_backend_state_identity(const BackendStateIdentity& id) noexcept;

// Cryptographic digest of request PromptImage payloads (all-zero iff images is empty).
std::array<std::uint8_t, 32> hash_prompt_media(std::span<const PromptImage> images) noexcept;

// Exact token-ID prefix check required alongside BackendStateIdentity::is_reusable_with() on any cache lookup.
bool prefix_matches_exact(std::span<const std::int32_t> cached_ids, std::span<const std::int32_t> prompt_ids,
                          std::size_t prefix_len) noexcept;

// Exact token-ID + per-token boundary/Engram mask prefix check required for DeepSeek/token-mask families.
bool prefix_matches_exact_with_mask(std::span<const std::int32_t> cached_ids,
                                    std::span<const std::uint8_t> cached_mask,
                                    std::span<const std::int32_t> prompt_ids,
                                    std::span<const std::uint8_t> prompt_mask, std::size_t prefix_len) noexcept;

// Tiered residency telemetry (#836, GLM/DeepSeek/Qwen unified memory accounting behind #835 HAL).
struct ResidencyReport {
  std::uint64_t device_bytes = 0;
  std::uint64_t host_ram_bytes = 0;
  std::uint64_t mmap_bytes = 0;
  std::uint64_t nvme_swap_bytes = 0;
  std::uint64_t process_rss_bytes = 0;
  std::uint64_t weights_device_bytes = 0;
  std::uint64_t state_device_bytes = 0;
  std::uint64_t cache_device_bytes = 0;
};

// Admission footprint estimate (#836) before allocating buffers for a workload profile.
struct AdmissionEstimate {
  std::uint64_t required_device_bytes = 0;
  std::uint64_t required_host_ram_bytes = 0;
  std::uint64_t required_mmap_bytes = 0;
  std::uint64_t required_nvme_swap_bytes = 0;
  bool fits_current_budget = false;
  std::string reason;
};

// Host-visible worker capability & profile descriptor (aligned with #822 / #829 / #838 dispatch).
struct WorkerProfile {
  ModelFamily family = ModelFamily::Unknown;
  std::string model_id;
  std::string gear;
  std::string kv_cache_mode;
  std::string tokenizer_format;
  std::string chat_template_family;
  int max_ctx = 0;
  int max_batch = 1;
  BackendCapabilities capabilities;
  BackendStateIdentity state_identity;
};

// Opaque per-session handle owned by a ModelBackend (opaque KV, recurrent, conv, or parked state).
class BackendSession {
 public:
  virtual ~BackendSession() = default;
  virtual void reset() = 0;
  virtual std::size_t state_bytes() const noexcept = 0;
  virtual std::span<const std::int32_t> committed_tokens() const noexcept = 0;
  virtual std::span<const std::uint8_t> committed_token_mask() const noexcept { return {}; }
  virtual const BackendStateIdentity& identity() const noexcept = 0;
  virtual void truncate(std::size_t /*new_token_count*/) {
    throw std::logic_error("session does not support truncate");
  }

  // Returns the largest prefix length <= max_prefix_len for which this session holds live
  // or saved restorable state. Default is tip-only: only committed_tokens().size() is
  // restorable (returning 0 for any strictly shorter fork prefix). Backends that save sparse
  // checkpoints or support arbitrary truncation override this alongside truncate().
  virtual std::size_t restorable_prefix_len(std::size_t max_prefix_len) const noexcept {
    const std::size_t tip = committed_tokens().size();
    return (tip > 0 && max_prefix_len >= tip) ? tip : 0;
  }

  // Restores session state to `prefix_len` (no-op at live tip; invokes truncate() for an
  // earlier saved checkpoint). Returns true iff the session now sits at `prefix_len`;
  // fails closed (false) if restoration is unsupported or throws std::logic_error.
  virtual bool try_restore_prefix(std::size_t prefix_len) {
    const std::size_t tip = committed_tokens().size();
    if (prefix_len == 0 || prefix_len > tip) return false;
    if (prefix_len == tip) return true;
    if (restorable_prefix_len(prefix_len) != prefix_len) return false;
    try {
      truncate(prefix_len);
    } catch (const std::logic_error&) {
      return false;
    }
    return committed_tokens().size() == prefix_len;
  }
};

struct LogitStep {
  int position = 0;
  std::span<const float> logits;
};

using LogitCallback = std::function<void(const LogitStep&)>;

// Shared host request payload carrying optional per-token mask and reuse toggle alongside Prompt.
struct BackendRequest {
  Prompt prompt;
  std::span<const std::uint8_t> token_mask;
  std::string cache_namespace;
  bool use_prefix_cache = true;
};

// Shared host query for mask/media-aware session prefix lookup.
struct SessionLookupQuery {
  BackendStateIdentity identity;
  std::span<const std::int32_t> prompt_ids;
  std::span<const std::uint8_t> prompt_mask;
  std::span<const PromptImage> images;
  bool use_prefix_cache = true;
};

struct SessionLookupResult {
  bool hit = false;
  std::size_t reused_tokens = 0;
  Reuse reuse = Reuse::None;
  std::string reason;
};

// Shared fail-closed prefix reuse gate: enforces identity reusability, media digest match,
// token + per-token mask match, image-boundary integrity, and session.restorable_prefix_len()
// availability (bypassing cleanly if !use_prefix_cache).
SessionLookupResult lookup_session_prefix(const BackendSession& session, const SessionLookupQuery& query) noexcept;
// Runs lookup_session_prefix() and confirms restoration via session.try_restore_prefix()
// before reporting a reuse hit. Fails closed to Reuse::None if restoration cannot succeed.
SessionLookupResult restore_session_prefix(BackendSession& session, const SessionLookupQuery& query);

// Fail-closed admission validators; MUST be invoked before allocating or mutating live model state.
void validate_runtime_options_against_capabilities(const BackendCapabilities& caps, const RuntimeOptions& options);
void validate_request_against_capabilities(const BackendCapabilities& caps, int max_model_len, const Prompt& prompt,
                                           int max_new_tokens, const Sampling& sampling);

// Model-family backend interface extending Generator without changing existing Generator callers or slot APIs.
class ModelBackend : public Generator {
 public:
  virtual const BackendCapabilities& capabilities() const = 0;
  virtual const WorkerProfile& worker_profile() const = 0;
  virtual const BackendStateIdentity& state_identity() const = 0;
  virtual void reset_session() = 0;

  // Explicit per-session / slot ownership lifecycle for host-managed state, mask/media reuse, and truncate.
  virtual std::unique_ptr<BackendSession> create_session() {
    throw std::logic_error("backend does not support standalone BackendSession creation");
  }
  virtual BackendSession* acquire_session(int /*slot*/ = 0) { return nullptr; }
  virtual void release_session(int /*slot*/ = 0) {}

  // Shared mask/media/reuse-aware host entry points preserving Generator callers unchanged.
  virtual GenerateStats generate_request(const BackendRequest& req, int max_new_tokens, const Sampling& sampling,
                                         std::span<const std::int32_t> stop_ids,
                                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                         const PrefillCallback& on_prefill = {});

  virtual void evaluate_teacher_forced(const BackendRequest& /*req*/, std::span<const int> /*capture_positions*/,
                                       const LogitCallback& /*on_logits*/) {
    throw std::logic_error("backend does not support teacher-forced request evaluation");
  }

  // Optional tiered-residency / #836 memory pressure hooks.
  virtual ResidencyReport residency_report() const { return {}; }
  virtual AdmissionEstimate admission_estimate(int /*max_ctx*/, int /*max_batch*/) const { return {}; }
  virtual void park_weights() { throw std::logic_error("backend does not support park_weights"); }
  virtual std::uint64_t evict_caches(std::uint64_t /*target_free_bytes*/) { return 0; }

  // Optional teacher-forced prefill/decode logit capture at requested 0-based positions for parity validation.
  virtual void evaluate_teacher_forced(std::span<const std::int32_t> /*token_ids*/,
                                       std::span<const int> /*capture_positions*/, bool /*use_prefix_cache*/,
                                       const LogitCallback& /*on_logits*/,
                                       std::span<const std::uint8_t> /*token_mask*/ = {}) {
    throw std::logic_error("backend does not support teacher-forced logit capture");
  }
};

struct BackendRegistration {
  ModelFamily family = ModelFamily::Unknown;
  std::string name;
  std::function<std::unique_ptr<ModelBackend>(const RuntimeOptions&)> create;
};

// Thread-safe static registry mapping ModelFamily -> immutable backend factory registration.
class BackendRegistry {
 public:
  static BackendRegistry& instance();

  // Rejects Unknown family, empty name, null factory, or duplicate registration for the same family.
  void register_backend(BackendRegistration reg);
  [[nodiscard]] std::optional<BackendRegistration> find(ModelFamily family) const;
  [[nodiscard]] std::unique_ptr<ModelBackend> create(ModelFamily family, const RuntimeOptions& options) const;
  [[nodiscard]] std::vector<ModelFamily> registered_families() const;

 private:
  BackendRegistry() = default;
};

// Default capability baseline describing the existing Inkling-2-Mini backend contract.
BackendCapabilities inkling_default_capabilities();

}  // namespace ingine::model
