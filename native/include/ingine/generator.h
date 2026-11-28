// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include "ingine/persona.h"
#include "ingine/platos_key.h"
#include "ingine/prefix_cache.h"
#include "ingine/token_masker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::model {

// Full-attention KV cache sharing mode across loop passes (layers 11 and 17).
// PerPass (default): all 8 full-attention visits (L5, L11x3, L17x3, L23) have independent KV slots.
// LastFull (opt-in): L11 passes 0..2 share one slot and L17 passes 0..2 share one slot (4 full slots total);
// prior tokens use the final pass's persisted KV while sliding rings (40) and conv states (48) stay pass-local.
enum class KvCacheMode : std::uint8_t {
  PerPass = 0,
  LastFull = 1,
};

inline const char* to_string(KvCacheMode mode) {
  switch (mode) {
    case KvCacheMode::LastFull: return "last_full";
    default: return "per_pass";
  }
}

inline KvCacheMode parse_kv_cache_mode(std::string_view s) {
  if (s == "per_pass") return KvCacheMode::PerPass;
  if (s == "last_full") return KvCacheMode::LastFull;
  throw std::invalid_argument("invalid kv_cache_mode '" + std::string(s) + "' (expected 'per_pass' or 'last_full')");
}

// Training-free Loop-wise Contrastive Decoding (Apple LoopCD, arXiv:2610.02185) for looped architectures (L >= 2).
// Off (default): standard decoding from final pass h_L (zero overhead, 100% bit-identical to baseline).
// LogitsAdaptive: fused single-pass dual-vector lm_head GEMV over h_L and h_{L-1} with adaptive omega = omega_max * (1 - (p1 - p2)).
// Hidden: pre-coda hidden-state extrapolation h' = h_L + omega * (h_L - h_{L-1}) before coda layers 18..23.
enum class LoopCdMode : std::uint8_t {
  Off = 0,
  LogitsAdaptive = 1,
  Hidden = 2,
};

inline const char* to_string(LoopCdMode mode) {
  switch (mode) {
    case LoopCdMode::LogitsAdaptive: return "logits-adaptive";
    case LoopCdMode::Hidden: return "hidden";
    default: return "off";
  }
}

inline LoopCdMode parse_loopcd_mode(std::string_view s) {
  if (s == "off") return LoopCdMode::Off;
  if (s == "logits-adaptive") return LoopCdMode::LogitsAdaptive;
  if (s == "hidden") return LoopCdMode::Hidden;
  throw std::invalid_argument("invalid loopcd mode '" + std::string(s) +
                              "' (expected 'off', 'logits-adaptive', or 'hidden')");
}

inline constexpr float default_loopcd_omega(LoopCdMode mode) noexcept {
  switch (mode) {
    case LoopCdMode::LogitsAdaptive: return 0.5f;
    case LoopCdMode::Hidden: return 0.3f;
    default: return 0.0f;
  }
}

inline constexpr float effective_loopcd_omega(LoopCdMode mode, float requested) noexcept {
  if (mode == LoopCdMode::Off) return 0.0f;
  return requested >= 0.0f ? requested : default_loopcd_omega(mode);
}

// Opt-in attention projection requantization (#888): applies only to Inkling-2-Mini attention projections
// (wproj + wo). Off (default) preserves baseline BF16/FP8 bit-for-bit.
enum class AttnRequantMode : std::uint8_t {
  Off = 0,
  Q5K = 1,
};

inline const char* to_string(AttnRequantMode mode) {
  switch (mode) {
    case AttnRequantMode::Q5K: return "q5_k";
    default: return "off";
  }
}

inline AttnRequantMode parse_attn_requant_mode(std::string_view s) {
  if (s == "off") return AttnRequantMode::Off;
  if (s == "q5_k") return AttnRequantMode::Q5K;
  throw std::invalid_argument("invalid attn_requant mode '" + std::string(s) +
                              "' (expected 'off' or 'q5_k')");
}

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Token ids masked to -inf before argmax/sampling (e.g. <|content_invoke_tool_json|> when no tools are provided).
  std::vector<int32_t> banned_tokens;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
  // Optional resolved persona/mech-interp profile for this request (nullptr => unhooked fast path).
  std::shared_ptr<const persona::PersonaProfile> persona;
  // Optional token-level constrained decoding masker (e.g. JSON / JSON-Schema FSM; nullptr => zero overhead).
  std::shared_ptr<TokenMasker> logit_masker;
};

// The token whose embedding an image token replaces (<|unused_200054|>, the chat template's image placeholder).
inline constexpr std::int32_t IMAGE_PLACEHOLDER_ID = 200054;

// One image of a prompt. The chat template marks an image with one placeholder token; the prompt holds
// tokens() ids in its place, one per 40x40-pixel tile in raster order, whose embeddings the vision tower
// computes from the pixels. These ids are negative (never a vocabulary id) and a function of the image
// content and the tile index (image::token_id), so prompts compare equal exactly when their images do.
struct PromptImage {
  int pos = 0;                                           // prompt position of the image's first token
  int height = 0, width = 0;                             // pixels
  std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // [height, width, 3]
  [[nodiscard]] int rows() const { return (height + 39) / 40; }
  [[nodiscard]] int cols() const { return (width + 39) / 40; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// A prompt: token ids, images in position order (without overlap), optional verified root-prefix snapshot
// boundary hint (0 = none; never a trust key), optional cache isolation namespace, and per-request bypass.
struct Prompt {
  std::span<const std::int32_t> ids;
  std::span<const PromptImage> images;
  int root_prefix_tokens = 0;
  std::string cache_namespace;
  bool bypass_prefix_cache = false;
};

// Validates that `root_prefix_tokens` lies inside [0, ids.size()] and does not straddle any PromptImage.
// Returns the validated boundary (or 0 if invalid/out-of-range).
inline int validate_root_prefix_tokens(std::span<const std::int32_t> ids, std::span<const PromptImage> images,
                                       int root_prefix_tokens) noexcept {
  if (root_prefix_tokens <= 0 || std::size_t(root_prefix_tokens) > ids.size()) return 0;
  for (const PromptImage& img : images) {
    if (img.pos < root_prefix_tokens && img.pos + img.tokens() > root_prefix_tokens) return 0;
  }
  return root_prefix_tokens;
}

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,            // empty state, the whole prompt prefilled
  Extend,          // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,      // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,          // restored a parked conversation's checkpoint in VRAM (a prefix of this prompt; RuntimeOptions::park_mb)
  RootCheckpoint,  // restored a verified root-prefix (system+tools) checkpoint
  HostParked,      // restored a parked conversation's checkpoint from pinned host RAM
  PagedPrefix,     // restored a shared paged full-KV + recurrent prefix checkpoint (batched B>1)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    case Reuse::RootCheckpoint: return "root_checkpoint";
    case Reuse::HostParked: return "host_parked";
    case Reuse::PagedPrefix: return "paged_prefix";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  // Actual bytes copied to restore reused model state, restore time, and cache evictions during this request.
  // std::nullopt means the backend/path did not measure the metric (never reported as a synthetic zero).
  std::optional<std::uint64_t> restored_bytes;
  std::optional<double> restore_seconds;
  std::optional<int> cache_evictions;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
  // Images: tokens of the prompt's images, how many images the vision tower encoded for this request (those
  // not inside the reused prefix) and its time (part of prefill_seconds).
  int image_tokens = 0, images_encoded = 0;
  double vision_seconds = 0;
};

using PrefillCallback = std::function<void(const GenerateStats&)>;

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  // Tokens per prefill chunk. 3072 (the largest the sliding attention takes) reads a long prompt 6-10% faster
  // than 2048 for 0.38 GB more scratch; results differ between chunk sizes only by rounding.
  int prefill_chunk = 3072;
  bool dense_fp8 = true, kv_fp8 = false, kv_q4 = false, spec = false;
  // FP4 (E2M1) full-attention values with FP8 keys in the TP batch pool (needs kv_fp8): 0.75 of the FP8 pool.
  bool kv_fp4v = false;
  std::filesystem::path dflash2_dir;  // optional DFlash2DraftModel dir (implies native DFlash2 drafter when spec=true)
  KvCacheMode kv_cache_mode = KvCacheMode::PerPass;
  float spec_threshold = 0.8f;
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = -1;
  // Bounded pinned host-RAM parking tier (MiB) for checkpoints evicted from VRAM or parked when park_mb=0
  // (LRU eviction, graceful fallback under host memory pressure; default 0 = opt-in).
  int park_host_mb = 0;
  // Symmetric prefix-cache toggle for both B=1 and B>1 (false = cache-off parity baseline, zero reuse).
  bool prefix_cache = true;
  // Image input: load the vision tower (636 MB of the checkpoint, resident) and accept prompts with images.
  bool vision = false;
  // Audio input: load the audio tower (for multimodal encoders such as EmbeddingGemma 2).
  bool audio = false;
  // Dynamic 3-tier MoE expert offloading (atlas-6g-homology):
  //   offload_vram_mb == 0: disabled (all experts resident in VRAM).
  //   offload_vram_mb > 0:  target VRAM cap in MiB (e.g. 6144 for 6 GB GPUs); non-expert weights, KV cache,
  //                         and static homology-pinned HOT experts + dynamic LRU pool stay inside this cap.
  //   offload_ram_mb:       pinned host RAM cap in MiB for WARM experts (default 12288 inside 16 GB RAM);
  //                         any remaining COLD experts stream on demand from read-only NVMe mmap.
  //   offload_mode:         "auto" (dma-lru), "dma-lru" (3-tier async PCIe DMA + dynamic VRAM LRU), or
  //                         "uva" (2-tier HOT VRAM + zero-copy UVA pinned RAM, CUDA-graph capturable).
  int offload_vram_mb = 0;
  int offload_ram_mb = 12288;
  std::string offload_mode = "auto";
  std::filesystem::path expert_ranking;     // optional JSON per-layer [24][128] homology/frequency scores
  std::filesystem::path dump_router_stats;  // optional JSON output path for observed [24][128] router hits
  // 3-tier paged Q4_0 full-attention KV cache (for 1M context on <= 12 GB and <= 6 GB GPUs):
  //   kv_paged / kv_vram_mb > 0: keeps hot 4096-token KV pages in VRAM, overflows warm pages to lazy
  //   pinned UVA HostRegion slabs (<= kv_ram_mb), and spills cold pages to a crash-safe NVMe SwapFile.
  bool kv_paged = false;
  int kv_vram_mb = 0;               // 0 = auto when kv_paged (256 MiB with offload, 1536 MiB otherwise)
  int kv_ram_mb = 12288;            // pinned UVA host RAM cap in MiB before spilling to NVMe SwapFile
  std::filesystem::path kv_swap_dir;  // optional directory for Tier-2 NVMe KV swap file
  // Atomic streaming disk checkpoint & resume for long-context (1M+) prefill runs:
  std::filesystem::path kv_ckpt_path;    // path for periodic & SIGTERM atomic KV checkpoints (auto-resumes if prefix matches)
  int kv_ckpt_every = 0;                 // save atomic checkpoint every N tokens during prefill (e.g. 131072; 0 = SIGTERM/manual only)
  std::filesystem::path kv_ckpt_resume;  // explicit checkpoint path to resume from (fails loudly if missing or mismatched)
  // Continuous batching: max concurrent sequences (1 = single-request EngineGenerator, zero B=1 regression).
  int max_batch = 1;
  // Shared paged full-attention KV pool capacity in tokens (0 = auto-size from free VRAM).
  int max_kv_tokens = 0;
  // Number of recurrent passes over layers 6..17 (default 3 = 48 visits; 1..8 supported for varloop models).
  int loop_passes = 3;
  // Loop-wise Contrastive Decoding (arXiv:2610.02185): opt-in off|logits-adaptive|hidden (requires loop_passes >= 2).
  LoopCdMode loopcd_mode = LoopCdMode::Off;
  // Contrastive strength omega_max (logits-adaptive) or omega (hidden); < 0 uses mode default (0.5 / 0.3), 0 = no-op control.
  float loopcd_omega = -1.0f;
  // Opt-in small-card knobs (#888, all default off / 0 so unmodified behavior is 0-ULP preserved):
  //   loop_bf16:     elide unused FP32 controller & router duplicates (-81.0 MiB persistent VRAM, 0-ULP parity).
  //   attn_requant:  opt-in GGML Q5_K (5.50 b/w) requantization of attention wproj + wo (-193.5 MiB vs FP8).
  //   expert_swaps:  bounded async GPU<->RAM HOT-promotion swap queue capacity (0 = disabled/default, >0 = opt-in;
  //                  strictly NO expert dropping / tail-skip).
  bool loop_bf16 = false;
  AttnRequantMode attn_requant = AttnRequantMode::Off;
  int expert_swaps = 0;
  // Optional VRAM lease cap in MiB for batched pool/drafter/prefix-cache auto-sizing (0 = unbounded).
  int vram_limit_mb = 0;
  // True when caller explicitly passed --loop-passes (non-looped backends reject explicit loop overrides fail-closed).
  bool loop_passes_explicit = false;
  // Mech-interp / persona hook options: pre-capture hooked CUDA graphs when enabled, and enforce refusal map.
  bool enable_persona_hooks = false;
  std::filesystem::path refusal_map;
  // Optional Plato's Key projection (.safetensors + optional sidecar JSON metadata) for /v1/embeddings.
  std::filesystem::path platos_key;
  std::filesystem::path platos_key_metadata;
};

// Pooling strategy for POST /v1/embeddings (zero-KV, no-lm_head prefill path).
enum class EmbeddingPooling : std::uint8_t {
  Mean = 0,
  LastToken = 1,
  Cls = 2,
};

inline const char* to_string(EmbeddingPooling p) noexcept {
  switch (p) {
    case EmbeddingPooling::LastToken: return "last_token";
    case EmbeddingPooling::Cls: return "cls";
    default: return "mean";
  }
}

inline EmbeddingPooling parse_embedding_pooling(std::string_view s) {
  if (s == "mean") return EmbeddingPooling::Mean;
  if (s == "last_token") return EmbeddingPooling::LastToken;
  if (s == "cls") return EmbeddingPooling::Cls;
  throw std::invalid_argument("invalid pooling '" + std::string(s) +
                              "' (expected 'mean', 'last_token', or 'cls')");
}

struct EmbeddingRequest {
  std::vector<std::vector<std::int32_t>> inputs;  // non-empty batch of non-empty token-id sequences
  EmbeddingPooling pooling = EmbeddingPooling::Mean;
  bool normalize = true;
  // Optional early-exit layer/visit index (0-based). std::nullopt = full forward pass + final norm.
  std::optional<int> tap_layer;
  // Opt-in in-GPU Plato's Key projection (requires server started with --platos-key).
  bool use_platos_key = false;
  // Optional #964 cartridge identity pins verified fail-closed against the loaded Plato's Key.
  std::optional<std::string> cartridge_id;
  std::optional<std::string> cartridge_sha256;
  // Optional Matryoshka output dimension (OpenAI "dimensions"); truncation happens before L2 re-normalization.
  std::optional<int> dimensions;
  // Optional named task prefix applied before tokenization (e.g. "SearchQuery", "Document" on EmbeddingGemma 2).
  std::optional<std::string> prompt_name;
};

struct EmbeddingStats {
  int total_prompt_tokens = 0;
  double prefill_seconds = 0.0;
  double platos_key_seconds = 0.0;
  int effective_tap_layer = -1;  // -1 = full depth (final normed hidden), else 0-based early-exit layer/visit
  int output_dim = 0;
  int kv_slots_allocated = 0;
  bool lm_head_skipped = true;
};

struct EmbeddingResult {
  std::vector<std::vector<float>> embeddings;  // [inputs.size()][output_dim]
  EmbeddingStats stats;
};

// One active sequence entering a batched decode step (continuous batching).
struct BatchSlotStep {
  int slot = -1;
  int32_t token = 0;
  // Optional per-step emission cap (0 = unconstrained; >0 clamps this slot's verify width w_cap <= max_tokens).
  int max_tokens = 0;
};

// Output of one batched speculative (or plain 1-token) decode step for one slot.
struct BatchSlotResult {
  static constexpr int kMaxTokens = 8;
  int32_t tokens[kMaxTokens] = {};
  int count = 0;          // number of emitted tokens in tokens[0..count-1] (1..kMaxTokens)
  int spec_rounds = 0;    // 1 if speculative verification ran on this slot, else 0
  int spec_accepted = 0;  // accepted draft count (0..count-1)
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  // The same for a prompt with images. Generators without image input take prompts without images only.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    if (!prompt.images.empty()) throw std::invalid_argument("this generator does not take images");
    return generate(prompt.ids, max_new_tokens, sampling, stop_ids, on_tokens);
  }
  // Variant that delivers completed prefill stats through `on_prefill` strictly before the first token is
  // emitted via `on_tokens`, so request threads that stop early (e.g. stop strings) still observe exact
  // prefill/cache accounting. Subclasses that do not override this fall back to firing `on_prefill` on return.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                 const PrefillCallback& on_prefill) {
    GenerateStats stats = generate(prompt, max_new_tokens, sampling, stop_ids, on_tokens);
    if (on_prefill) on_prefill(stats);
    return stats;
  }
  // Pre-prefills `prompt` into the backend prefix cache at startup without emitting output tokens (#887).
  // Defaults fail-closed to unsupported so generators without a verified prefix cache cannot advertise warm_start=true.
  virtual GenerateStats warm_start(const Prompt& /*prompt*/, const Sampling& /*sampling*/) {
    throw std::invalid_argument("warm_start is not supported by this generator backend (no prefix cache)");
  }
  virtual bool supports_images() const { return false; }
  virtual bool supports_audio() const { return false; }
  virtual bool is_encoder_only() const { return false; }
  virtual bool supports_logit_masker() const { return true; }
  virtual std::string model_family() const { return "inkling2"; }
  virtual KvCacheMode kv_cache_mode() const { return KvCacheMode::PerPass; }
  // Modes supported by this worker's execution profile (e.g. {"per_pass", "last_full"} for B=1 non-spec TP=1 L=3,
  // or {"per_pass"} for speculative, continuous-batch B>1, TP>1, or --loop-passes != 3 workers). Active mode is fixed
  // at worker init.
  virtual std::vector<KvCacheMode> supported_kv_cache_modes() const { return {kv_cache_mode()}; }
  virtual int loop_passes() const { return 3; }
  virtual LoopCdMode loopcd_mode() const { return LoopCdMode::Off; }
  virtual AttnRequantMode attn_requant_mode() const { return AttnRequantMode::Off; }
  virtual std::string offload_summary() const { return {}; }
  virtual PrefixCacheStats cache_stats() const { return {}; }

  // ---- Zero-KV, no-lm_head batched embedding interface (POST /v1/embeddings)
  virtual bool supports_embeddings() const { return false; }
  virtual int embedding_dim() const { return 0; }
  virtual int num_tap_layers() const { return 0; }
  virtual std::vector<int> matryoshka_dims() const { return {}; }
  virtual std::map<std::string, std::string> embedding_prompt_prefixes() const { return {}; }
  virtual bool has_platos_key() const { return false; }
  virtual int platos_key_out_dim() const { return 0; }
  virtual std::string platos_key_tap_spec() const { return {}; }
  virtual std::shared_ptr<const PlatosKey> platos_key() const { return nullptr; }
  virtual EmbeddingResult embed(const EmbeddingRequest& /*req*/) {
    throw std::runtime_error("embeddings are not supported by this generator backend");
  }

  // ---- Optional continuous-batching interface (used by Scheduler and ingine-bench when max_batch_size() > 1)
  virtual int max_batch_size() const { return 1; }
  virtual bool can_admit_slot(int /*prompt_tokens*/, int /*max_new_tokens*/) const { return false; }
  // Prompt-aware admission check accounting for reusable shared committed KV pages (defaults to cold estimate).
  virtual bool can_admit_slot(const Prompt& prompt, int max_new_tokens, const Sampling& /*sampling*/) const {
    return can_admit_slot(int(prompt.ids.size()), max_new_tokens);
  }
  // Allocates a free slot, prefills `prompt`, samples `first_token` and fills `stats`. Returns slot id >= 0,
  // or -1 if no slot / KV pages are currently available. Throws std::invalid_argument on invalid prompt.
  virtual int start_slot(const Prompt& /*prompt*/, int /*max_new_tokens*/, const Sampling& /*sampling*/,
                         int32_t& /*first_token*/, GenerateStats& /*stats*/) {
    throw std::logic_error("this generator does not support batched decoding");
  }
  // Runs one batched decode step across `inputs` (1..max_batch_size() active slots), sampling each slot's
  // next token into `out_tokens[i]` and advancing each slot's position.
  virtual void step_slots(std::span<const BatchSlotStep> /*inputs*/, std::span<int32_t> /*out_tokens*/) {
    throw std::logic_error("this generator does not support batched decoding");
  }
  // True when batched multi-token speculative decoding (step_slots_multi) is active.
  virtual bool supports_batched_spec() const { return false; }
  // Runs one batched speculative (or plain 1-token) step across `inputs`, emitting 1..8 tokens per slot.
  // Default wraps step_slots() so non-spec generators and CPU fakes require zero changes.
  virtual void step_slots_multi(std::span<const BatchSlotStep> inputs, std::span<BatchSlotResult> out) {
    if (out.size() < inputs.size()) throw std::invalid_argument("step_slots_multi: out too small");
    std::vector<int32_t> toks(inputs.size(), 0);
    step_slots(inputs, toks);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      out[i] = BatchSlotResult{};
      out[i].tokens[0] = toks[i];
      out[i].count = 1;
    }
  }
  // Releases `slot` and returns its KV pages immediately. Safe to call on any active slot.
  virtual void release_slot(int /*slot*/) {}

  // Shared single-request lifecycle over the continuous-batching slot API (used by BatchGenerator::generate()
  // and unit tests so slot/KV-page exception cleanup is tested directly through the production path).
  GenerateStats generate_from_slots(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                                    std::span<const int32_t> stop_ids,
                                    const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                    const PrefillCallback& on_prefill) {
    GenerateStats stats;
    std::int32_t token = 0;
    const int n = int(full_prompt.ids.size());
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    const int max_len = max_model_len();
    if (n + max_new_tokens > max_len) {
      max_new_tokens = max_len - n;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(max_len));
    }
    const int s = start_slot(full_prompt, max_new_tokens, sampling, token, stats);
    if (s < 0) throw std::runtime_error("BatchGenerator::generate: no free batch slot");
    auto is_stop = [&](std::int32_t tok) {
      return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
    };
    const auto t0 = std::chrono::steady_clock::now();
    try {
      if (on_prefill) on_prefill(stats);
      int emitted = 1;
      stats.generated_tokens = 1;
      const std::int32_t first_out[1] = {token};
      bool go_on = on_tokens(first_out);
      const bool use_multi = supports_batched_spec();
      while (go_on && emitted < max_new_tokens && !is_stop(token)) {
        const BatchSlotStep step[1] = {{s, token}};
        if (use_multi) {
          BatchSlotResult res[1] = {};
          step_slots_multi(step, res);
          stats.spec_rounds += res[0].spec_rounds;
          stats.spec_accepted += res[0].spec_accepted;
          int take = 0;
          bool hit_stop = false;
          for (int k = 0; k < res[0].count && emitted < max_new_tokens; ++k) {
            ++emitted;
            ++take;
            if (is_stop(res[0].tokens[k])) {
              hit_stop = true;
              break;
            }
          }
          stats.generated_tokens = emitted;
          if (take > 0) {
            go_on = on_tokens(std::span<const std::int32_t>(res[0].tokens, std::size_t(take)));
            token = res[0].tokens[take - 1];
          }
          if (hit_stop || take == 0) break;
        } else {
          std::int32_t next[1] = {0};
          step_slots(step, next);
          token = next[0];
          ++emitted;
          stats.generated_tokens = emitted;
          const std::int32_t out[1] = {token};
          go_on = on_tokens(out);
        }
      }
    } catch (...) {
      release_slot(s);
      throw;
    }
    stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    release_slot(s);
    return stats;
  }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
