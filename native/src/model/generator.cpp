// The Generator (engine.Engine): one resident model, one request at a time. Prefill runs in chunks through
// the Prefiller; plain decoding replays the width-1 graph per token, pipelined (each step's input token
// stays on the device, the host launches the next step before it reads the current token); with
// options.spec the speculative loop (SpecLoop) drafts with DSpark and verifies with the width-2/4/8 graphs.
//
// Prefix reuse hierarchy (all fail-closed on exact StateIdentity + exact token-ID match; disabled wholly
// when RuntimeOptions::prefix_cache == false):
//   1. Extend: live state is a strict prefix of the incoming prompt.
//   2. Checkpoint: last prefill's in-place device checkpoint is a prefix of the incoming prompt.
//   3. Parked (2-tier bounded LRU: Tier 0 VRAM `park_mb` + Tier 1 pinned host RAM `park_host_mb`):
//      longest exact prefix match across parked conversations and verified root-prefix checkpoints.
//      Root checkpoints (`is_root = true`) are captured at a verified token boundary `root_prefix_tokens`
//      during cold prefill and are never erased as "superseded" by longer child extensions.
//   4. Optional disk KV checkpoint resume (`kv_ckpt_resume` / `kv_ckpt_path`).
#include "ingine/generator.h"
#include "ingine/platos_key.h"

#include "backends/embedding_gemma2/eg2_cuda.h"
#include "backends/qwen38/qwen38_dense_gpu.h"
#include "backends/qwen4exp/qwen4exp_gpu.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/ops.h"
#include "ingine/model_backend.h"
#include "kernels/head/head.h"
#include "model/batch_engine.h"
#include "model/tp_engine.h"
#include "model/drafter.h"
#include "model/kv_checkpoint.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/spec.h"
#include "model/vision.h"
#include "model/weights_gguf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

namespace ingine::model {
namespace {

namespace head = kernels::head;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const RuntimeOptions& without_spec(const RuntimeOptions& options) {
  if (options.spec) throw std::invalid_argument("make_generator over given weights: spec needs the model directory");
  return options;
}

std::array<std::uint8_t, 32> compute_model_identity_sha(const RuntimeOptions& opt, const Weights& weights) {
  std::array<std::uint8_t, 128> buf{};
  if (weights.content_sha256.has_value()) {
    std::memcpy(buf.data(), weights.content_sha256->data(), 32);
  }
  const auto abl_sha = opt.ablation.empty() ? std::array<std::uint8_t, 32>{} : sha256_file_or_zero(opt.ablation);
  std::memcpy(buf.data() + 32, abl_sha.data(), 32);
  if (opt.dflash2_dir.empty()) {
    return sha256_bytes(buf.data(), 64);
  }
  const auto df2_cfg = sha256_file_or_zero(opt.dflash2_dir / "config.json");
  const auto df2_wt = sha256_file_or_zero(opt.dflash2_dir / "model.safetensors");
  std::memcpy(buf.data() + 64, df2_cfg.data(), 32);
  std::memcpy(buf.data() + 96, df2_wt.data(), 32);
  return sha256_bytes(buf.data(), buf.size());
}

// Verified media identity bound to exact position, dimensions, and full SHA-256 over RGB pixels.
struct VerifiedImageSpan {
  int pos = 0;
  int height = 0;
  int width = 0;
  int tokens = 0;
  std::array<std::uint8_t, 32> rgb_sha256{};
  bool operator==(const VerifiedImageSpan&) const = default;
};

std::vector<VerifiedImageSpan> build_verified_images(std::span<const PromptImage> images) {
  std::vector<VerifiedImageSpan> out;
  out.reserve(images.size());
  for (const PromptImage& img : images) {
    VerifiedImageSpan span;
    span.pos = img.pos;
    span.height = img.height;
    span.width = img.width;
    span.tokens = img.tokens();
    if (img.rgb && !img.rgb->empty()) {
      span.rgb_sha256 = sha256_bytes(img.rgb->data(), img.rgb->size());
    }
    out.push_back(span);
  }
  return out;
}

std::vector<VerifiedImageSpan> filter_images_in_prefix(const std::vector<VerifiedImageSpan>& imgs,
                                                       std::size_t prefix_len) {
  std::vector<VerifiedImageSpan> out;
  for (const VerifiedImageSpan& img : imgs) {
    if (std::size_t(img.pos) + std::size_t(img.tokens) <= prefix_len) out.push_back(img);
  }
  return out;
}

// True iff every image of `incoming` intersecting [0, prefix_len) lies wholly inside [0, prefix_len) and matches
// `cached` (exact position, geometry, and full RGB SHA-256).
bool images_match_prefix(const std::vector<VerifiedImageSpan>& cached, const std::vector<VerifiedImageSpan>& incoming,
                         std::size_t prefix_len) {
  std::size_t ci = 0;
  for (const VerifiedImageSpan& img : incoming) {
    if (std::size_t(img.pos) >= prefix_len) break;
    if (std::size_t(img.pos) + std::size_t(img.tokens) > prefix_len) return false;  // straddles boundary
    if (ci >= cached.size() || !(cached[ci] == img)) return false;
    ++ci;
  }
  return ci == cached.size();
}

class EngineGenerator final : public Generator {
 public:
  explicit EngineGenerator(const RuntimeOptions& options)
      : EngineGenerator(options, std::make_unique<Runtime>(options), /*file_backed_runtime=*/true) {}
  EngineGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : EngineGenerator(without_spec(options),
                        std::make_unique<Runtime>(options, std::move(weights), std::move(ablation)),
                        /*file_backed_runtime=*/false) {}

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }

  bool supports_images() const override { return vision_ != nullptr; }
  bool supports_embeddings() const override { return true; }
  int embedding_dim() const override { return HIDDEN; }
  int num_tap_layers() const override { return prefiller_.num_visits(); }
  bool has_platos_key() const override { return platos_key_gpu_ != nullptr; }
  int platos_key_out_dim() const override { return platos_key_gpu_ ? platos_key_gpu_->out_dim() : 0; }
  std::string platos_key_tap_spec() const override {
    return platos_key_gpu_ ? platos_key_gpu_->metadata().tap_spec : "";
  }
  std::shared_ptr<const PlatosKey> platos_key() const override {
    return platos_key_gpu_ ? platos_key_gpu_->key() : nullptr;
  }
  EmbeddingResult embed(const EmbeddingRequest& req) override {
    std::lock_guard<std::mutex> lock(mutex_);
    // Stateless zero-KV embedding prefill: never reads or writes rt_.state(), checkpoint_, or parked_.
    return prefiller_.embed(req, options_.max_ctx, platos_key_gpu_.get());
  }

  PrefixCacheStats cache_stats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_stats_locked();
  }

  KvCacheMode kv_cache_mode() const override { return options_.kv_cache_mode; }

  std::vector<KvCacheMode> supported_kv_cache_modes() const override {
    if (options_.spec || options_.loop_passes != PASSES || options_.loopcd_mode != LoopCdMode::Off) {
      return {KvCacheMode::PerPass};
    }
    return {KvCacheMode::PerPass, KvCacheMode::LastFull};
  }

  int loop_passes() const override { return options_.loop_passes; }
  LoopCdMode loopcd_mode() const override { return options_.loopcd_mode; }
  AttnRequantMode attn_requant_mode() const override { return options_.attn_requant; }

  std::string offload_summary() const override {
    std::string out;
    if (rt_.weights().expert_cache) out = rt_.weights().expert_cache->summary();
    if (rt_.state().paged_kv) {
      if (!out.empty()) out += "\n";
      out += rt_.state().paged_kv->summary();
    }
    const PrefixCacheStats cs = cache_stats();
    if (cs.vram_budget_bytes > 0 || cs.host_budget_bytes > 0) {
      char buf[512];
      std::snprintf(
          buf, sizeof(buf),
          "[prefix-cache] staging=%.1f MiB | vram=%.1f/%.1f MiB (peak=%.1f MiB, entries=%llu, evict=%llu) | "
          "host_pinned=%.1f/%.1f MiB (peak=%.1f MiB, entries=%llu, evict=%llu, demote=%llu, oom_fallback=%llu) | "
          "roots=%llu | xfer(h2d=%.2f MB d2h=%.2f MB d2d_r=%.2f MB d2d_p=%.2f MB)",
          double(cs.device_staging_bytes) / (1024.0 * 1024.0), double(cs.vram_used_bytes) / (1024.0 * 1024.0),
          double(cs.vram_budget_bytes) / (1024.0 * 1024.0), double(cs.vram_peak_bytes) / (1024.0 * 1024.0),
          static_cast<unsigned long long>(cs.vram_entries), static_cast<unsigned long long>(cs.vram_evictions),
          double(cs.host_used_bytes) / (1024.0 * 1024.0), double(cs.host_budget_bytes) / (1024.0 * 1024.0),
          double(cs.host_peak_bytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(cs.host_entries),
          static_cast<unsigned long long>(cs.host_evictions), static_cast<unsigned long long>(cs.host_demotions),
          static_cast<unsigned long long>(cs.host_alloc_failures), static_cast<unsigned long long>(cs.root_entries),
          double(cs.h2d_restore_bytes) / 1e6, double(cs.d2h_park_bytes) / 1e6, double(cs.d2d_restore_bytes) / 1e6,
          double(cs.d2d_park_bytes) / 1e6);
      if (!out.empty()) out += "\n";
      out += buf;
    }
    return out;
  }

 private:
  EngineGenerator(const RuntimeOptions& options, std::unique_ptr<Runtime> runtime, bool file_backed_runtime)
      : options_(runtime->options()),
        runtime_(std::move(runtime)),
        rt_(*runtime_),
        prefiller_(rt_.weights(), rt_.ablation(), rt_.options().prefill_chunk, rt_.stream(), rt_.options().loop_passes,
                   rt_.options().loopcd_mode,
                   effective_loopcd_omega(rt_.options().loopcd_mode, rt_.options().loopcd_omega)),
        chunk_tokens_(std::size_t(rt_.options().prefill_chunk) * sizeof(std::int32_t)),
        logits_(std::size_t(UNPADDED_VOCAB) * sizeof(float)),
        token_(sizeof(std::int32_t)),
        host_token_(sizeof(std::int32_t)) {
    if (options_.prefill_chunk < 1) throw std::invalid_argument("prefill_chunk must be positive");
    refusal_map_ = options_.refusal_map.empty() ? persona::RefusalMap::default_map()
                                                : persona::RefusalMap::load(options_.refusal_map);
    if (options_.spec) {
      drafter_ = std::make_unique<Drafter>(options_.model_dir, rt_.weights(), options_.max_ctx, rt_.stream(),
                                           Drafter::SHORTLIST_ROWS, options_.dflash2_dir);
      drafter_->capture();
      taps_ = DeviceBuffer(std::size_t(options_.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
      spec_ = std::make_unique<SpecLoop>(rt_, *drafter_);
    }
    // checkpoint layout: target state | drafter context | last prompt logits
    checkpoint_drafter_ = carved(rt_.checkpoint_bytes());
    checkpoint_logits_ = checkpoint_drafter_ + (drafter_ ? carved(drafter_->checkpoint_bytes()) : 0);
    checkpoint_ = DeviceBuffer(checkpoint_logits_ + logits_.bytes);
    cache_stats_.device_staging_bytes = checkpoint_.bytes;

    CUDA_CHECK(cudaEventCreate(&restore_start_));
    CUDA_CHECK(cudaEventCreate(&restore_end_));

    // Build fail-closed runtime/model identity pinned for all prefix-cache lookups.
    // Verified disk producer identity is permitted ONLY for untouched file-backed models (make_generator(options))
    // with consumed-byte weight SHA-256, non-tiered resident experts, and no active ablation (file OR supplied).
    const bool ablation_active = rt_.ablation().active || !options_.ablation.empty();
    has_verified_producer_ = file_backed_runtime && rt_.weights().content_sha256.has_value() && !ablation_active &&
                             !rt_.weights().expert_cache;
    base_identity_.model_sha256 = compute_model_identity_sha(options_, rt_.weights());
    base_identity_.act_dtype = (kActDType == DType::FP16) ? 1u : 0u;
    base_identity_.dense_fp8 = options_.dense_fp8;
    base_identity_.kv_format = options_.kv_q4 ? 2u : (options_.kv_fp8 ? 1u : 0u);
    base_identity_.kv_cache_mode = static_cast<std::uint32_t>(options_.kv_cache_mode);
    base_identity_.loop_passes = options_.loop_passes;
    base_identity_.loopcd_mode = static_cast<std::uint32_t>(options_.loopcd_mode);
    if (options_.loopcd_mode != LoopCdMode::Off) {
      const float eff_omega = effective_loopcd_omega(options_.loopcd_mode, options_.loopcd_omega);
      std::uint32_t omega_bits = 0u;
      std::memcpy(&omega_bits, &eff_omega, sizeof(omega_bits));
      base_identity_.loopcd_omega_bits = omega_bits;
    } else {
      base_identity_.loopcd_omega_bits = 0u;
    }
    base_identity_.attn_requant = static_cast<std::uint8_t>(options_.attn_requant);
    base_identity_.max_ctx = options_.max_ctx;
    base_identity_.prefill_chunk = options_.prefill_chunk;
    base_identity_.spec_enabled = (drafter_ != nullptr);
    if (has_verified_producer_) {
      disk_producer_sha_ = compute_disk_producer_sha256(base_identity_);
    }

    // the vision tower's weights are resident: loaded before the park budget below is measured
    if (options_.vision) vision_ = std::make_unique<VisionEncoder>(options_.model_dir, rt_.stream());
    if (!options_.platos_key.empty()) {
      platos_key_gpu_ = PlatosKeyGpu::load(options_.platos_key, options_.platos_key_metadata, HIDDEN, rt_.stream());
      platos_key_gpu_->key()->validate_binding(model_id(), model_family(), HIDDEN, prefiller_.num_visits());
    }
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));

    if (!options_.prefix_cache) {
      park_budget_ = 0;
      host_park_budget_ = 0;
    } else if (rt_.state().paged_kv) {
      // Full-row parking requires contiguous full-attention buffers; fail closed under 3-tier paged KV.
      if (options_.park_mb > 0 || options_.park_host_mb > 0) {
        std::fprintf(stderr, "prefix-cache: parking disabled with --kv-paged (contiguous full-KV required)\n");
      }
      park_budget_ = 0;
      host_park_budget_ = 0;
    } else {
      if (options_.offload_vram_mb > 0 && options_.park_mb < 0) {
        park_budget_ = 0;  // keep strict VRAM cap in offload mode unless user explicitly requests VRAM parking
      } else if (options_.park_mb < 0) {  // leave room for the workspaces a long prefill adds later
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        park_budget_ = std::min<std::size_t>(3072, std::max<std::size_t>(free >> 20, 4096) - 4096) << 20;
      } else {
        park_budget_ = std::size_t(options_.park_mb) << 20;
      }
      host_park_budget_ = options_.park_host_mb > 0 ? (std::size_t(options_.park_host_mb) << 20) : 0;
    }
    cache_stats_.vram_budget_bytes = park_budget_;
    cache_stats_.host_budget_bytes = host_park_budget_;

    // parked VRAM conversations are best-effort: a device allocation that fails releases them and retries.
    // HookControl prevents a copied ReleaseHook invoked outside arena.cu's registry lock from touching a
    // destroyed EngineGenerator.
    hook_ctrl_ = std::make_shared<HookControl>();
    hook_ctrl_->gen = this;
    auto ctrl = hook_ctrl_;
    release_hook_ = add_release_hook([ctrl] {
      std::lock_guard<std::mutex> hlk(ctrl->m);
      return ctrl->gen ? ctrl->gen->release_parked() : false;
    });
  }

 public:
  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }

  GenerateStats warm_start(const Prompt& full_prompt, const Sampling& sampling) override {
    if (!options_.prefix_cache) {
      throw std::invalid_argument("warm_start requires prefix caching to be enabled");
    }
    if (full_prompt.ids.empty()) {
      throw std::invalid_argument("warm_start: prompt is empty");
    }
    if (full_prompt.bypass_prefix_cache) {
      throw std::invalid_argument("warm_start: prefix cache bypass requested");
    }
    if (!full_prompt.images.empty()) {
      throw std::invalid_argument("warm_start: images are not supported in startup warm-start prompts");
    }
    if (sampling.persona && sampling.persona->bypasses_prefix_reuse()) {
      throw std::invalid_argument("warm_start: default persona bypasses prefix cache reuse");
    }
    const int reserve = drafter_ ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (int(full_prompt.ids.size()) + reserve >= options_.max_ctx) {
      throw std::invalid_argument("warm_start: prompt of " + std::to_string(full_prompt.ids.size()) +
                                  " tokens leaves no generation room within max_ctx " +
                                  std::to_string(options_.max_ctx));
    }
    const GenerateStats stats = generate(full_prompt, /*max_new_tokens=*/0, sampling, {},
                                         [](std::span<const std::int32_t>) { return false; }, PrefillCallback{});
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const bool active_persona = (sampling.persona && !sampling.persona->is_unhooked());
      StateIdentity expected_id = base_identity_;
      if (active_persona) {
        expected_id.persona_key = sampling.persona->cache_key;
        expected_id.persona_fingerprint = sampling.persona->fingerprint_sha256;
      }
      expected_id.cache_namespace = full_prompt.cache_namespace;
      const bool live_retained = !live_bypass_reuse_ && live_identity_ == expected_id && live_images_.empty() &&
                                 tokens_in_state_.size() == full_prompt.ids.size() &&
                                 std::equal(full_prompt.ids.begin(), full_prompt.ids.end(), tokens_in_state_.begin());
      if (!live_retained) {
        throw std::runtime_error(
            "warm_start: requested prefix/identity could not be retained as reusable in B=1 prefix cache");
      }
      const bool ckpt_retained = checkpoint_identity_ == expected_id && checkpoint_images_.empty() &&
                                 checkpoint_ids_.size() == full_prompt.ids.size() &&
                                 std::equal(full_prompt.ids.begin(), full_prompt.ids.end(), checkpoint_ids_.begin());
      if (!ckpt_retained) {
        save_checkpoint();
        CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
      }
    }
    return stats;
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    const std::span<const std::int32_t> prompt = full_prompt.ids;
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    const int n = int(prompt.size());
    const int image_tokens = check_images(full_prompt);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      if (prompt[i] >= VOCAB) {
        throw std::out_of_range("prompt token id " + std::to_string(prompt[i]) +
                                " out of vocabulary range [0, " + std::to_string(VOCAB) + ")");
      }
    }
    if (!sampling.forced.empty()) {
      ops::validate_token_ids_i32(sampling.forced.data(), int(sampling.forced.size()), VOCAB);
      max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    }
    // a verify call writes up to max(verify widths) positions
    const int reserve = drafter_ ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    GenerateStats stats;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    if (sampling.persona) persona::validate_profile_safety(*sampling.persona, refusal_map_);
    const bool active_persona = (sampling.persona && !sampling.persona->is_unhooked());
    StateIdentity req_identity = base_identity_;
    if (active_persona) {
      req_identity.persona_key = sampling.persona->cache_key;
      req_identity.persona_fingerprint = sampling.persona->fingerprint_sha256;
    }
    req_identity.cache_namespace = full_prompt.cache_namespace;
    const bool bypass_reuse = !options_.prefix_cache || full_prompt.bypass_prefix_cache ||
                              (sampling.persona && sampling.persona->bypasses_prefix_reuse());
    const int root_prefix_tokens =
        validate_root_prefix_tokens(prompt, full_prompt.images, full_prompt.root_prefix_tokens);
    std::lock_guard<std::mutex> lock(mutex_);
    struct OwnerGuard {
      std::atomic<std::thread::id>& tid;
      explicit OwnerGuard(std::atomic<std::thread::id>& t) : tid(t) {
        tid.store(std::this_thread::get_id(), std::memory_order_release);
      }
      ~OwnerGuard() { tid.store(std::thread::id{}, std::memory_order_release); }
    } owner_guard(active_owner_);

    auto t = std::chrono::steady_clock::now();
    bool restore_timed = false;
    try {
      rt_.set_persona(sampling.persona.get());
      prefill(prompt, full_prompt.images, root_prefix_tokens, req_identity, bypass_reuse, stats, restore_timed);
      stats.prompt_tokens = n;
      stats.image_tokens = image_tokens;
      auto finalize_restore_timing = [&] {
        if (restore_timed) {
          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, restore_start_, restore_end_));
          stats.restore_seconds = double(ms) * 1e-3;
          restore_timed = false;
        }
      };
      if (spec_ && !sampling.logit_masker) {
        std::int32_t token = sample(static_cast<const float*>(logits_.p), sampling, rng);
        if (!sampling.forced.empty()) token = sampling.forced[0];
        finalize_restore_timing();
        stats.prefill_seconds = seconds_since(t);
        if (on_prefill) on_prefill(stats);
        t = std::chrono::steady_clock::now();
        if (max_new_tokens > 0) {
          drafter_->set_seed(rng.seed);
          spec_->generate(token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, tokens_in_state_);
        }
      } else {
        decode(n, max_new_tokens, sampling, rng, stop_ids, on_tokens, on_prefill, finalize_restore_timing, stats, t);
        // Constrained decode skips speculative drafter updates; reset drafter so the next unconstrained
        // --spec request re-syncs cleanly without stale draft rows.
        if (drafter_ && sampling.logit_masker) drafter_->reset();
      }
      CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
      finalize_restore_timing();
    } catch (...) {
      forget_state();
      throw;
    }
    stats.decode_seconds = seconds_since(t);
    return stats;
  }

  ~EngineGenerator() override {
    if (hook_ctrl_) {
      std::lock_guard<std::mutex> hlk(hook_ctrl_->m);
      hook_ctrl_->gen = nullptr;
    }
    remove_release_hook(release_hook_);
    // nothing queued may outlive the graphs and buffers of the members destroyed below
    if (rt_.stream()) cudaStreamSynchronize(rt_.stream());
    if (restore_start_) cudaEventDestroy(restore_start_);
    if (restore_end_) cudaEventDestroy(restore_end_);
  }

 private:
  // The images of a prompt lie inside it, in order and without overlap, and cover exactly its negative ids
  // (an image token's id; anything else negative would index outside the embedding table). Returns their tokens.
  int check_images(const Prompt& prompt) const {
    if (!prompt.images.empty() && !vision_) {
      throw std::invalid_argument("the prompt has images but the vision tower is not loaded (RuntimeOptions::vision)");
    }
    const std::int64_t n = std::int64_t(prompt.ids.size());
    std::int64_t end = 0, tokens = 0;
    for (const PromptImage& image : prompt.images) {
      if (!image.rgb || image.height < 1 || image.width < 1 ||
          image.rgb->size() != std::size_t(image.height) * std::size_t(image.width) * 3) {
        throw std::invalid_argument("a prompt image has no pixels or pixels of another size");
      }
      if (image.pos < end || std::int64_t(image.pos) + image.tokens() > n) {
        throw std::invalid_argument("the prompt's images overlap, are out of order or lie outside the prompt");
      }
      end = std::int64_t(image.pos) + image.tokens();
      tokens += image.tokens();
    }
    const std::int64_t negative = std::count_if(prompt.ids.begin(), prompt.ids.end(), [](std::int32_t id) { return id < 0; });
    bool covered = negative == tokens;
    for (const PromptImage& image : prompt.images) {
      for (int k = 0; covered && k < image.tokens(); ++k) covered = prompt.ids[std::size_t(image.pos + k)] < 0;
    }
    if (!covered) throw std::invalid_argument("negative token ids of the prompt are not exactly its image tokens");
    return int(tokens);
  }

  // After a failure anywhere in a request the device state may be ahead of or behind tokens_in_state_ (a step
  // in flight, a partial prefill, a round between launch and bookkeeping): wait for the stream and forget
  // everything reusable, so the next request starts from an empty state.
  void forget_state() noexcept {
    cudaStreamSynchronize(rt_.stream());
    tokens_in_state_.clear();
    checkpoint_ids_.clear();
    live_images_.clear();
    checkpoint_images_.clear();
    live_identity_ = StateIdentity{};
    checkpoint_identity_ = StateIdentity{};
    live_bypass_reuse_ = false;
    active_keep_ = nullptr;
    unparking_ = false;
    if (drafter_) drafter_->reset();  // stream-ordered; CUDA failures abort (CUDA_CHECK)
  }

  head::SamplingParams to_head_params(const Sampling& sampling) const {
    head::SamplingParams p;
    p.temperature = sampling.temperature;
    p.top_k = sampling.top_k;
    p.top_p = sampling.top_p;
    for (std::int32_t id : sampling.banned_tokens) {
      if (id >= 0 && p.banned.count < head::BannedTokens::kMax) p.banned.ids[p.banned.count++] = id;
    }
    return p;
  }

  void decode(int n, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
              std::span<const std::int32_t> stop_ids,
              const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
              const PrefillCallback& on_prefill, const std::function<void()>& finalize_restore_timing,
              GenerateStats& stats, std::chrono::steady_clock::time_point& t) {
    const cudaStream_t stream = rt_.stream();
    if (max_new_tokens <= 0) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      finalize_restore_timing();
      stats.prefill_seconds = seconds_since(t);
      if (on_prefill) on_prefill(stats);
      t = std::chrono::steady_clock::now();
      return;
    }
    const head::SamplingParams params = to_head_params(sampling);
    auto* sampled = static_cast<std::int32_t*>(token_.p);
    // Constrained decoding path (response_format json_object / json_schema):
    // Keeps the captured CUDA decode graph (rt_.run_step(1)) untouched and applies a 25 KiB
    // bitmask kernel only on steps where logit_masker->is_active() is true (zero masking in thinking).
    if (sampling.logit_masker) {
      TokenMasker& masker = *sampling.logit_masker;
      const std::size_t words = (std::size_t(UNPADDED_VOCAB) + 31u) / 32u;
      if (mask_host_.size() < words) mask_host_.resize(words);
      if (!mask_dev_.p || mask_dev_.bytes < words * sizeof(std::uint32_t)) {
        mask_dev_ = DeviceBuffer(words * sizeof(std::uint32_t));
      }
      if (!masked_logits_.p || masked_logits_.bytes < logits_.bytes) {
        masked_logits_ = DeviceBuffer(logits_.bytes);
      }
      auto sample_masked = [&](const float* row_logits, int step_idx) -> std::int32_t {
        const float* use_logits = row_logits;
        if (masker.is_active()) {
          std::fill(mask_host_.begin(), mask_host_.begin() + std::ptrdiff_t(words), 0u);
          if (masker.fill_mask(mask_host_.data(), words)) {
            CUDA_CHECK(cudaMemcpyAsync(mask_dev_.p, mask_host_.data(), words * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
            head::apply_bitmask(row_logits, static_cast<const std::uint32_t*>(mask_dev_.p), UNPADDED_VOCAB,
                                static_cast<float*>(masked_logits_.p), stream);
            use_logits = static_cast<const float*>(masked_logits_.p);
          }
        }
        head::sample(use_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
        CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), sampled, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int32_t tok = *static_cast<const std::int32_t*>(host_token_.data());
        if (std::size_t(step_idx) < sampling.forced.size()) tok = sampling.forced[std::size_t(step_idx)];
        return tok;
      };
      auto is_stop = [&](std::int32_t tok) {
        return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
      };

      const float* cur_logits = static_cast<const float*>(logits_.p);
      for (int i = 0; i < max_new_tokens; ++i) {
        const std::int32_t token = sample_masked(cur_logits, i);
        if (i == 0) {
          finalize_restore_timing();
          stats.prefill_seconds = seconds_since(t);
          if (on_prefill) on_prefill(stats);
          t = std::chrono::steady_clock::now();
        }
        masker.advance(token);
        const bool last = i + 1 == max_new_tokens;
        stats.generated_tokens = i + 1;
        const std::int32_t out[1] = {token};
        const bool go_on = on_tokens(out);
        if (last || is_stop(token) || !go_on) break;
        tokens_in_state_.push_back(token);
        CUDA_CHECK(cudaMemcpyAsync(sampled, &token, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        rt_.chain_advance(sampled, n + i);
        cur_logits = rt_.run_step(1);
      }
      return;
    }

    // forced replay: the forced tokens on the device; each sampled token is replaced by its forced one
    DeviceBuffer forced;
    if (!sampling.forced.empty()) {
      forced = DeviceBuffer(sampling.forced.size_bytes());
      CUDA_CHECK(cudaMemcpyAsync(forced.p, sampling.forced.data(), sampling.forced.size_bytes(),
                                 cudaMemcpyHostToDevice, stream));
    }
    // engine._sample of the logits into `sampled`, then the inputs of the step at `position`
    auto sample_for = [&](const float* logits, int position) {
      head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
      if (forced.p)
        CUDA_CHECK(cudaMemcpyAsync(sampled, static_cast<const std::int32_t*>(forced.p) + (position - n),
                                   sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
      rt_.chain_advance(sampled, position);
    };
    // the step at `position` (its input token is on the device), then the next token
    auto launch = [&](int position) { sample_for(rt_.run_step(1), position + 1); };
    auto is_stop = [&](std::int32_t tok) { return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end(); };

    sample_for(static_cast<const float*>(logits_.p), n);
    if (max_new_tokens > 1) launch(n);
    std::int32_t token = rt_.chain_token(n);
    finalize_restore_timing();
    stats.prefill_seconds = seconds_since(t);
    if (on_prefill) on_prefill(stats);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
      // `token` is at position n + i; its step is in flight unless it is the budget's last token
      const bool last = i + 1 == max_new_tokens;
      stats.generated_tokens = i + 1;
      const std::int32_t out[1] = {token};
      const bool go_on = on_tokens(out);
      if (last || is_stop(token) || !go_on) {
        if (!last) rt_.chain_undo(n + i);
        break;
      }
      tokens_in_state_.push_back(token);
      if (i + 2 < max_new_tokens) launch(n + i + 1);
      token = rt_.chain_token(n + i + 1);
    }
  }

  void prefill(std::span<const std::int32_t> ids, std::span<const PromptImage> images, int root_prefix_tokens,
               const StateIdentity& req_identity, bool bypass_reuse, GenerateStats& stats, bool& restore_timed) {
    const std::vector<std::int32_t>& prev = tokens_in_state_;
    const std::size_t n = ids.size();
    const std::vector<VerifiedImageSpan> req_images = build_verified_images(images);
    const bool drafter_in_step = !drafter_ || drafter_->length() == int(prev.size());  // else a row is missing
    State& st = rt_.state();
    const cudaStream_t stream = rt_.stream();
    const bool extend = !bypass_reuse && !live_bypass_reuse_ && live_identity_ == req_identity && drafter_in_step &&
                        !prev.empty() && prev.size() < n && std::equal(prev.begin(), prev.end(), ids.begin()) &&
                        images_match_prefix(live_images_, req_images, prev.size());
    const bool checkpoint = !bypass_reuse && !extend && checkpoint_identity_ == req_identity &&
                            !checkpoint_ids_.empty() && checkpoint_ids_.size() <= n &&
                            std::equal(checkpoint_ids_.begin(), checkpoint_ids_.end(), ids.begin()) &&
                            images_match_prefix(checkpoint_images_, req_images, checkpoint_ids_.size());
    const std::size_t live_match = extend ? prev.size() : (checkpoint ? checkpoint_ids_.size() : 0u);
    Parked* parked = bypass_reuse ? nullptr : find_parked(ids, req_images, live_match, req_identity);
    std::size_t start = 0;
    bool resumed_has_full_state = true;
    bool resumed_unverified_legacy = false;
    request_evictions_ = 0;
    if (parked) {
      // Pin the selected parked node across park_live() and unpark() so neither LRU eviction nor a re-entrant
      // release_parked() hook can drop the entry we are about to restore (std::list keeps node pointers stable).
      struct RestorePinGuard {
        const Parked*& keep_ref;
        bool& unpark_ref;
        RestorePinGuard(const Parked*& k, bool& u, const Parked* target) : keep_ref(k), unpark_ref(u) {
          keep_ref = target;
        }
        ~RestorePinGuard() {
          unpark_ref = false;
          keep_ref = nullptr;
        }
      } pin_guard(active_keep_, unparking_, parked);

      park_live(parked, request_evictions_);
      const ParkTier restored_tier = parked->tier;
      const bool restored_root = parked->is_root;
      unparking_ = true;
      CUDA_CHECK(cudaEventRecord(restore_start_, stream));
      const std::size_t unpark_bytes = unpark(*parked);
      const std::size_t ckpt_bytes = restore_checkpoint();
      CUDA_CHECK(cudaEventRecord(restore_end_, stream));
      unparking_ = false;
      restore_timed = true;
      start = checkpoint_ids_.size();
      stats.restored_bytes = std::uint64_t(unpark_bytes + ckpt_bytes);
      stats.reuse = restored_tier == ParkTier::Host
                        ? Reuse::HostParked
                        : (restored_root ? Reuse::RootCheckpoint : Reuse::Parked);
    } else if (extend) {
      start = prev.size();
      stats.reuse = Reuse::Extend;
      stats.restored_bytes = 0u;
      stats.restore_seconds = 0.0;
    } else if (checkpoint) {
      CUDA_CHECK(cudaEventRecord(restore_start_, stream));
      const std::size_t ckpt_bytes = restore_checkpoint();
      CUDA_CHECK(cudaEventRecord(restore_end_, stream));
      restore_timed = true;
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Checkpoint;
      stats.restored_bytes = std::uint64_t(ckpt_bytes);
    } else {
      if (!bypass_reuse) park_live(nullptr, request_evictions_);
      reset_state(st, stream);
      if (drafter_) drafter_->reset();
      tokens_in_state_.clear();
      checkpoint_ids_.clear();
      live_images_.clear();
      checkpoint_images_.clear();
      live_identity_ = req_identity;
      checkpoint_identity_ = StateIdentity{};
      live_bypass_reuse_ = bypass_reuse;
      stats.reuse = Reuse::None;
      stats.restored_bytes = 0u;
      stats.restore_seconds = 0.0;
      // Optional disk KV checkpoint resume (single-fd atomic validation + load):
      //   * Explicit --kv-ckpt-resume fails loudly on runtime/producer/media/missing-logits mismatches — requiring a
      //     verified matching reader whenever the file carries a v1 producer stamp — while preserving unstamped legacy
      //     checkpoints that lack the v1 producer trailer (legacy resumes remain isolated and are never promoted to
      //     reusable root entries).
      //   * Auto --kv-ckpt-path requires verified producer identity (weights package + ablation + act_dtype +
      //     dense_fp8 + kv_format + prefill_chunk), text-only prefix, non-spec mode, and boundary logits when
      //     pos == n; deliberate compatible case: max_ctx_at_save may differ as long as pos <= runtime.max_ctx.
      auto has_media_before = [&](int pos_limit) {
        for (const PromptImage& img : images) {
          if (img.pos < pos_limit) return true;
        }
        return false;
      };
      const bool ablation_active = rt_.ablation().active || !options_.ablation.empty();
      const bool baseline_disk_ok =
          req_identity.persona_key == 0 && req_identity.cache_namespace.empty() && !bypass_reuse && !ablation_active;
      if (!options_.kv_ckpt_resume.empty()) {
        if (!baseline_disk_ok) {
          throw std::invalid_argument(
              "--kv-ckpt-resume cannot be combined with an ablation, hooked persona, namespace, or --no-prefix-cache");
        }
        if (drafter_) {
          throw std::invalid_argument(
              "--kv-ckpt-resume is incompatible with --spec (disk checkpoint stores no DSpark drafter state)");
        }
        const auto tr = std::chrono::steady_clock::now();
        const KvCheckpointLoadResult res = load_kv_checkpoint_checked(
            st, options_.model_dir, ids, static_cast<float*>(logits_.p), options_.kv_ckpt_resume, stream,
            [&](const KvCheckpointInfo& info) -> bool {
              if (!kv_checkpoint_runtime_matches(info, req_identity)) {
                throw std::runtime_error("--kv-ckpt-resume runtime dtype/kv_format mismatch");
              }
              if (info.has_producer_identity) {
                if (!has_verified_producer_) {
                  throw std::runtime_error(
                      "--kv-ckpt-resume cannot verify stamped checkpoint producer against an unverified runtime "
                      "(tiered offload or caller-supplied weights)");
                }
                if (info.producer_sha256 != disk_producer_sha_) {
                  throw std::runtime_error(
                      "--kv-ckpt-resume producer identity mismatch (weights package, dense_fp8, or prefill_chunk)");
                }
              }
              if (info.pos > 0 && has_media_before(info.pos)) {
                throw std::invalid_argument(
                    "--kv-ckpt-resume cannot resume a prefix containing images (disk checkpoint hashes token IDs only)");
              }
              if (info.pos == int(n) && !info.has_logits) {
                throw std::runtime_error(
                    "--kv-ckpt-resume checkpoint omits boundary logits (has_logits=false) at full prompt length");
              }
              return true;
            });
        start = std::size_t(res.info.pos);
        const bool verified_producer =
            has_verified_producer_ && res.info.has_producer_identity && res.info.producer_sha256 == disk_producer_sha_;
        resumed_has_full_state = res.info.has_logits && !drafter_ && verified_producer;
        resumed_unverified_legacy = !verified_producer;
        stats.reuse = Reuse::Checkpoint;
        stats.restored_bytes = std::nullopt;  // disk payload transfer bytes not counted inside load_kv_checkpoint
        stats.restore_seconds = seconds_since(tr);
        std::fprintf(stderr, "[kv-ckpt] resumed %d / %zu tokens from %s (has_logits=%s, verified_producer=%s)\n",
                     res.info.pos, n, options_.kv_ckpt_resume.c_str(), res.info.has_logits ? "true" : "false",
                     verified_producer ? "true" : "false");
      } else if (has_verified_producer_ && baseline_disk_ok && !drafter_ && !options_.kv_ckpt_path.empty() &&
                 std::filesystem::exists(options_.kv_ckpt_path)) {
        try {
          const auto tr = std::chrono::steady_clock::now();
          const KvCheckpointLoadResult res = load_kv_checkpoint_checked(
              st, options_.model_dir, ids, static_cast<float*>(logits_.p), options_.kv_ckpt_path, stream,
              [&](const KvCheckpointInfo& info) -> bool {
                if (!kv_checkpoint_auto_resume_eligible(info, req_identity, disk_producer_sha_, n, options_.max_ctx,
                                                        has_media_before(info.pos))) {
                  if (info.kv_cache_mode != options_.kv_cache_mode) {
                    std::fprintf(stderr, "[kv-ckpt] skipping auto-resume %s: kv_cache_mode mismatch (ckpt=%s, engine=%s)\n",
                                 options_.kv_ckpt_path.c_str(), to_string(info.kv_cache_mode),
                                 to_string(options_.kv_cache_mode));
                  }
                  return false;
                }
                const auto psha = sha256_bytes(ids.data(), std::size_t(info.pos) * sizeof(std::int32_t));
                return psha == info.prompt_sha256;
              });
          if (res.loaded) {
            start = std::size_t(res.info.pos);
            const bool verified_producer =
                has_verified_producer_ && res.info.has_producer_identity &&
                res.info.producer_sha256 == disk_producer_sha_;
            resumed_has_full_state = res.info.has_logits && !drafter_ && verified_producer;
            resumed_unverified_legacy = !verified_producer;
            stats.reuse = Reuse::Checkpoint;
            stats.restored_bytes = std::nullopt;
            stats.restore_seconds = seconds_since(tr);
            std::fprintf(stderr, "[kv-ckpt] auto-resumed %d / %zu tokens from %s (has_logits=%s)\n", res.info.pos, n,
                         options_.kv_ckpt_path.c_str(), res.info.has_logits ? "true" : "false");
          }
        } catch (const std::exception& e) {
          // Always reset live/paged/recurrent state before cold fallback in case copying started before failure.
          reset_state(st, stream);
          if (drafter_) drafter_->reset();
          start = 0;
          resumed_has_full_state = false;
          resumed_unverified_legacy = false;
          stats.reuse = Reuse::None;
          stats.restored_bytes = 0u;
          stats.restore_seconds = 0.0;
          std::fprintf(stderr, "[kv-ckpt] ignoring incompatible existing checkpoint %s: %s\n",
                       options_.kv_ckpt_path.c_str(), e.what());
        }
      }
    }

    // Verified root-prefix snapshot boundary: only capture if we have parking budget, a valid verified root
    // boundary >= kMinRootTokens, no unverified legacy disk provenance, and no exact parked copy already exists.
    const std::size_t root_len =
        (!bypass_reuse && !resumed_unverified_legacy && can_park_any() && root_prefix_tokens >= int(kMinRootTokens))
            ? std::size_t(root_prefix_tokens)
            : 0u;
    const std::vector<VerifiedImageSpan> root_images =
        root_len > 0 ? filter_images_in_prefix(req_images, root_len) : std::vector<VerifiedImageSpan>{};
    bool need_root_capture = !resumed_unverified_legacy && root_len > 0 &&
                             !has_exact_parked(ids.subspan(0, root_len), root_images, req_identity, /*mark_root=*/true);
    // Only promote an exact-length restored checkpoint to a reusable root when full runtime/drafter state AND
    // boundary logits at root_len - 1 are proved (never promote a logit-less or drafter-less disk resume).
    if (need_root_capture && start == root_len && stats.reuse == Reuse::Checkpoint && resumed_has_full_state) {
      capture_root_checkpoint(ids.subspan(0, root_len), root_images, req_identity, request_evictions_);
      need_root_capture = false;
    }

    st.length = int(start);  // decode steps advance the state without its host-side length
    std::size_t next_ckpt =
        options_.kv_ckpt_every > 0
            ? ((start / std::size_t(options_.kv_ckpt_every)) + 1u) * std::size_t(options_.kv_ckpt_every)
            : 0u;
    const std::size_t chunk = std::size_t(options_.prefill_chunk);
    struct Encoded {
      const PromptImage* image;
      DeviceBuffer rows;  // bf16 [>= tokens, HIDDEN]
    };
    std::deque<Encoded> encoded;  // images whose rows the chunks still need (or were queued to read)
    std::size_t next_image = 0;
    while (next_image < images.size() &&
           std::size_t(images[next_image].pos) + std::size_t(images[next_image].tokens()) <= start) {
      ++next_image;  // inside the reused prefix
    }
    std::vector<EmbedRows> rows;
    for (std::size_t s = start; s < n;) {
      std::size_t step = std::min(chunk, n - s);
      // Split once at the verified root boundary so sliding rings, conv states, drafter context, and exact
      // logits at root_len - 1 can be snapshotted in one pass with zero recomputation.
      if (need_root_capture && s < root_len && s + step > root_len) {
        step = root_len - s;
      }
      const int T = int(step);
      for (; next_image < images.size() && std::size_t(images[next_image].pos) < s + step; ++next_image) {
        const PromptImage& image = images[next_image];
        const auto t = std::chrono::steady_clock::now();
        DeviceBuffer out = vision_->encode(image.rgb->data(), image.height, image.width);
        // encode() returns with the stream idle: the rows of images that ended before this chunk have been read
        while (!encoded.empty() && std::size_t(encoded.front().image->pos + encoded.front().image->tokens()) <= s) {
          encoded.pop_front();
        }
        encoded.push_back(Encoded{&image, std::move(out)});
        stats.vision_seconds += seconds_since(t);
        ++stats.images_encoded;
      }
      rows.clear();
      for (const Encoded& e : encoded) {
        const std::int64_t first = std::max<std::int64_t>(e.image->pos, std::int64_t(s));
        const std::int64_t last = std::min<std::int64_t>(std::int64_t(e.image->pos) + e.image->tokens(), std::int64_t(s) + T);
        if (first < last) {
          rows.push_back(EmbedRows{int(first - std::int64_t(s)), int(last - first),
                                   static_cast<const ActT*>(e.rows.p) + std::size_t(first - e.image->pos) * HIDDEN});
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, ids.data() + s, std::size_t(T) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream));
      Tensor taps;
      if (drafter_) taps = rm(taps_.p, kActDType, {T, NUM_TAPS, HIDDEN});
      prefiller_.run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, int(s), static_cast<float*>(logits_.p),
                     drafter_ ? &taps : nullptr, rows, rt_.active_hooks());
      if (drafter_) {
        drafter_->note_tokens(ids.subspan(s, step));  // the prompt's tokens join the drafter's shortlist
        drafter_->commit(taps, int(s));
      }
      const std::size_t done = s + step;
      if (need_root_capture && done == root_len) {
        capture_root_checkpoint(ids.subspan(0, root_len), root_images, req_identity, request_evictions_);
        need_root_capture = false;
      }
      if (!drafter_ && options_.ablation.empty() && !rt_.ablation().active && images.empty() &&
          req_identity.persona_key == 0 && req_identity.cache_namespace.empty() && !bypass_reuse &&
          !options_.kv_ckpt_path.empty()) {
        const bool sigterm = kv_checkpoint_sigterm_requested();
        const bool periodic = (options_.kv_ckpt_every > 0 && done >= next_ckpt && done < n);
        if (periodic || sigterm) {
          // Never stamp verified producer identity on checkpoints from unverified given weights or legacy descendants.
          const bool stamp_verified_producer = has_verified_producer_ && !resumed_unverified_legacy;
          const KvCheckpointProducer producer{stamp_verified_producer, options_.dense_fp8, options_.prefill_chunk,
                                              disk_producer_sha_};
          save_kv_checkpoint(st, options_.model_dir, ids.subspan(0, done), static_cast<const float*>(logits_.p),
                             options_.kv_ckpt_path, stream, stamp_verified_producer ? &producer : nullptr);
          std::fprintf(stderr, "[kv-ckpt] %s saved atomic checkpoint at pos=%zu / %zu -> %s (verified_producer=%s)\n",
                       sigterm ? "SIGTERM" : "periodic", done, n, options_.kv_ckpt_path.c_str(),
                       stamp_verified_producer ? "true" : "false");
          if (options_.kv_ckpt_every > 0) {
            while (next_ckpt <= done) next_ckpt += std::size_t(options_.kv_ckpt_every);
          }
          if (sigterm) {
            throw std::runtime_error("prefill interrupted by SIGTERM after saving atomic KV checkpoint");
          }
        }
      }
      s = done;
    }
    if (!encoded.empty()) CUDA_CHECK(cudaStreamSynchronize(stream));  // queued chunks read the rows freed here
    tokens_in_state_.assign(ids.begin(), ids.end());
    live_images_ = req_images;
    live_identity_ = req_identity;
    // Isolate explicit legacy-resumed descendants from all reusable state (live Extend, last-prefill Checkpoint,
    // and parked Tier-0/Tier-1 entries) until a subsequent verified cold recompute replaces the live state.
    const bool isolate_reuse = bypass_reuse || resumed_unverified_legacy;
    live_bypass_reuse_ = isolate_reuse;
    if (isolate_reuse) {
      checkpoint_ids_.clear();
      checkpoint_images_.clear();
      checkpoint_identity_ = StateIdentity{};
    } else if (start < n) {
      save_checkpoint();
    }
    stats.reused_tokens = int(start);
    stats.cache_evictions = request_evictions_;
  }

  // Two-tier bounded conversation & root-prefix parking (stored in std::list for stable node addresses/iterators
  // across concurrent-tier LRU evictions and demotions).
  // Tier 0 (VRAM): up to park_budget_ bytes, fast D2D copies, dropped on VRAM pressure.
  // Tier 1 (Pinned Host RAM): up to host_park_budget_ bytes, async PCIe D2H/H2D DMA, strict LRU eviction,
  //   graceful fallback under host memory pressure, survives GPU VRAM pressure (0 bytes of VRAM).
  // Every entry is keyed by full StateIdentity + exact token IDs + verified image SHA-256 spans.
  // Eviction semantics:
  //   - vram_evictions: Tier-0 VRAM entries dropped (capacity drop without demotion, VRAM OOM release, or disable).
  //   - host_evictions: Tier-1 Host entries dropped (capacity LRU or host pinned OOM retry).
  //   - host_demotions: Tier-0 -> Tier-1 moves (preserved in Tier 1, NOT counted as evictions).
  //   - Superseded same-conversation prefix replacements are NOT counted as capacity evictions.
  enum class ParkTier : std::uint8_t { Vram = 0, Host = 1 };

  struct Parked {
    std::vector<std::int32_t> ids, context;
    std::vector<VerifiedImageSpan> images;
    StateIdentity identity;
    std::uint64_t id_hash = 0;
    std::uint64_t prefix_hash = 0;
    bool is_root = false;  // root checkpoints are never erased as superseded by longer child extensions
    ParkTier tier = ParkTier::Vram;
    DeviceBuffer vram_data;
    PinnedHostBuffer host_data;
    std::size_t bytes = 0;
    std::uint64_t used = 0;

    void* ptr() noexcept { return tier == ParkTier::Vram ? vram_data.p : host_data.data(); }
    const void* ptr() const noexcept { return tier == ParkTier::Vram ? vram_data.p : host_data.data(); }
  };
  static constexpr std::size_t kMinParked = 256;       // shorter non-root prompts are cheaper to prefill than to keep
  static constexpr std::size_t kMinRootTokens = 64;    // skip bare ~12-tok effort-only prefixes with no system/tools
  static constexpr std::uint64_t kHostSafetyReserveBytes = std::uint64_t(256) << 20;  // keep >=256 MiB host RAM free

  bool can_park_any() const noexcept { return park_budget_ > 0 || host_park_budget_ > 0; }

  // the layout of a parked copy of n tokens: {offset, bytes} of each drafter region, logits, full rows
  struct ParkLayout {
    std::array<Drafter::Region, 3> drafter{};
    std::size_t drafter_off[3] = {}, logits = 0, rows = 0, bytes = 0;
  };
  ParkLayout park_layout(int n) const {
    ParkLayout l;
    std::size_t off = checkpoint_drafter_;
    if (drafter_) {
      l.drafter = drafter_->checkpoint_regions(n);
      for (int i = 0; i < 3; ++i) l.drafter_off[i] = off, off += carved(l.drafter[i].bytes);
    }
    l.logits = off;
    l.rows = off + carved(logits_.bytes);
    l.bytes = l.rows + rt_.full_rows_bytes(n);
    return l;
  }

  void refresh_cache_usage_locked() noexcept {
    std::size_t v_used = 0, h_used = 0;
    std::uint64_t v_cnt = 0, h_cnt = 0, r_cnt = 0;
    for (const Parked& p : parked_) {
      if (p.tier == ParkTier::Vram) {
        v_used += p.bytes;
        ++v_cnt;
      } else {
        h_used += p.bytes;
        ++h_cnt;
      }
      if (p.is_root) ++r_cnt;
    }
    cache_stats_.vram_budget_bytes = park_budget_;
    cache_stats_.host_budget_bytes = host_park_budget_;
    cache_stats_.vram_used_bytes = v_used;
    cache_stats_.host_used_bytes = h_used;
    cache_stats_.vram_peak_bytes = std::max(cache_stats_.vram_peak_bytes, v_used);
    cache_stats_.host_peak_bytes = std::max(cache_stats_.host_peak_bytes, h_used);
    cache_stats_.vram_entries = v_cnt;
    cache_stats_.host_entries = h_cnt;
    cache_stats_.root_entries = r_cnt;
  }

  PrefixCacheStats cache_stats_locked() const noexcept { return cache_stats_; }

  bool has_exact_parked(std::span<const std::int32_t> prefix, const std::vector<VerifiedImageSpan>& imgs,
                        const StateIdentity& identity, bool mark_root) {
    const std::uint64_t ih = hash_state_identity(identity);
    const std::uint64_t ph = hash_token_prefix(prefix);
    for (Parked& p : parked_) {
      if (p.id_hash == ih && p.prefix_hash == ph && p.ids.size() == prefix.size() && p.identity == identity &&
          p.images == imgs && std::equal(p.ids.begin(), p.ids.end(), prefix.begin())) {
        if (mark_root) p.is_root = true;
        p.used = ++clock_;
        refresh_cache_usage_locked();
        return true;
      }
    }
    return false;
  }

  // Longest exact parked prefix of `ids` matching `identity` and verified `req_images` with length > `live`.
  // Prefers Tier 0 (VRAM) over Tier 1 (Host) on equal length.
  Parked* find_parked(std::span<const std::int32_t> ids, const std::vector<VerifiedImageSpan>& req_images,
                      std::size_t live, const StateIdentity& identity) {
    const std::uint64_t ih = hash_state_identity(identity);
    Parked* best = nullptr;
    for (Parked& p : parked_) {
      const std::size_t min_len = p.is_root ? kMinRootTokens : kMinParked;
      if (p.id_hash != ih || p.ids.size() < min_len || p.ids.size() <= live || p.ids.size() > ids.size()) continue;
      if (best && p.ids.size() < best->ids.size()) continue;
      if (best && p.ids.size() == best->ids.size() && best->tier == ParkTier::Vram) continue;
      if (p.identity == identity && std::equal(p.ids.begin(), p.ids.end(), ids.begin()) &&
          images_match_prefix(p.images, req_images, p.ids.size())) {
        best = &p;
      }
    }
    return best;
  }

  // The release hook (core/arena.h): invoked outside arena.cu's registry lock by THIS generator (same thread,
  // holding mutex_) or by ANOTHER generator on another thread. Never deadlocks or races with an active request,
  // never drops a pinned pending restore (`active_keep_`), and preserves Tier-1 pinned host entries (0 VRAM).
  bool release_parked() {
    const bool same_thread = (active_owner_.load(std::memory_order_acquire) == std::this_thread::get_id());
    std::unique_lock<std::mutex> lk;
    if (!same_thread) {
      lk = std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
      if (!lk.owns_lock()) return false;  // another thread is mid-request on this generator; skip safely
    }
    if (parking_ || unparking_) return false;
    bool any_vram = false;
    for (const Parked& p : parked_) {
      if (p.tier == ParkTier::Vram && &p != active_keep_) {
        any_vram = true;
        break;
      }
    }
    if (!any_vram) return false;
    cudaStreamSynchronize(rt_.stream());  // copies from or into parked buffers may be queued
    std::size_t dropped = 0;
    for (auto it = parked_.begin(); it != parked_.end();) {
      if (it->tier == ParkTier::Vram && &(*it) != active_keep_) {
        it = parked_.erase(it);
        ++dropped;
      } else {
        ++it;
      }
    }
    if (same_thread) request_evictions_ += int(dropped);
    cache_stats_.vram_evictions += dropped;
    ++cache_stats_.vram_pressure_releases;
    refresh_cache_usage_locked();
    std::fprintf(stderr, "device memory low: released %zu VRAM parked conversation(s)\n", dropped);
    return dropped > 0;
  }

  std::size_t tier_used_bytes(ParkTier tier, const Parked* keep_entry, std::size_t* kept_out = nullptr) const noexcept {
    std::size_t total = 0, kept_bytes = 0;
    for (const Parked& p : parked_) {
      if (p.tier != tier) continue;
      total += p.bytes;
      if (keep_entry && &p == keep_entry) kept_bytes += p.bytes;
    }
    if (kept_out) *kept_out = kept_bytes;
    return total;
  }

  // Attempt to allocate a PinnedHostBuffer of `bytes` within `host_park_budget_`, evicting Tier-1 LRU entries
  // as needed and degrading gracefully under host memory pressure (never throws). Because `parked_` is a
  // std::list, erasing any Tier-1 entry NEVER invalidates caller iterators or `keep_entry`.
  bool alloc_host_park_slot(std::size_t bytes, const Parked* keep_entry, PinnedHostBuffer& out, int& req_evictions) {
    if (host_park_budget_ == 0 || bytes == 0) return false;
    auto kept = [&](const Parked& p) { return keep_entry && &p == keep_entry; };
    std::size_t kept_host = 0;
    std::size_t host_used = tier_used_bytes(ParkTier::Host, keep_entry, &kept_host);
    if (kept_host + bytes > host_park_budget_) return false;

    auto evict_one_host_lru = [&]() -> bool {
      auto lru = parked_.end();
      for (auto it = parked_.begin(); it != parked_.end(); ++it) {
        if (it->tier == ParkTier::Host && !kept(*it) && (lru == parked_.end() || it->used < lru->used)) lru = it;
      }
      if (lru == parked_.end()) return false;
      CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));  // ensure no in-flight H2D/D2H touches the buffer being freed
      host_used -= lru->bytes;
      parked_.erase(lru);
      ++cache_stats_.host_evictions;
      ++req_evictions;
      return true;
    };

    while (host_used + bytes > host_park_budget_) {
      if (!evict_one_host_lru()) return false;
    }

    // Respect OS MemAvailable safety margin before pinning new host RAM.
    const std::uint64_t avail = mem_available_bytes();
    if (avail > 0 && avail < kHostSafetyReserveBytes + bytes) {
      std::uint64_t cur_avail = avail;
      while (cur_avail < kHostSafetyReserveBytes + bytes && evict_one_host_lru()) {
        cur_avail = mem_available_bytes();
      }
      if (cur_avail > 0 && cur_avail < kHostSafetyReserveBytes + bytes) {
        ++cache_stats_.host_alloc_failures;
        refresh_cache_usage_locked();
        std::fprintf(stderr, "prefix-cache: skipped host park of %.1f MiB (MemAvailable %.1f MiB below safety reserve)\n",
                     double(bytes) / (1024.0 * 1024.0), double(cur_avail) / (1024.0 * 1024.0));
        return false;
      }
    }

    try {
      out = PinnedHostBuffer(bytes);
      return true;
    } catch (const std::exception& e) {
      cudaGetLastError();  // clear sticky allocation error
      bool freed_any = false;
      while (evict_one_host_lru()) freed_any = true;
      if (freed_any) {
        try {
          out = PinnedHostBuffer(bytes);
          return true;
        } catch (const std::exception&) {
          cudaGetLastError();
        }
      }
      ++cache_stats_.host_alloc_failures;
      refresh_cache_usage_locked();
      std::fprintf(stderr, "prefix-cache: host pinned allocation failed (%.1f MiB): %s (continuing without parking)\n",
                   double(bytes) / (1024.0 * 1024.0), e.what());
      return false;
    }
  }

  // Demote a Tier-0 VRAM list node into Tier-1 pinned host RAM instead of dropping it, when budget allows.
  // `victim` is a stable std::list iterator and remains valid even if alloc_host_park_slot evicts earlier nodes
  // or fails allocation.
  bool try_demote_to_host(std::list<Parked>::iterator victim, const Parked* keep_entry, int& req_evictions) {
    if (host_park_budget_ == 0 || victim == parked_.end() || victim->tier != ParkTier::Vram ||
        &(*victim) == keep_entry) {
      return false;
    }
    PinnedHostBuffer hbuf;
    if (!alloc_host_park_slot(victim->bytes, keep_entry, hbuf, req_evictions)) return false;
    const cudaStream_t s = rt_.stream();
    CUDA_CHECK(cudaMemcpyAsync(hbuf.data(), victim->vram_data.p, victim->bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));  // D2H must finish before releasing victim->vram_data
    cache_stats_.d2h_park_bytes += victim->bytes;  // exact cudaMemcpyAsync transfer size
    victim->vram_data = DeviceBuffer();
    victim->host_data = std::move(hbuf);
    victim->tier = ParkTier::Host;
    ++cache_stats_.host_demotions;
    refresh_cache_usage_locked();
    return true;
  }

  // Copies the current packed checkpoint in `checkpoint_` plus full rows [0, n) from `rt_.state()` into a new
  // Tier-0 VRAM or Tier-1 pinned host RAM entry. Never fails the request and never evicts `keep_entry`.
  void store_parked_from_checkpoint(std::span<const std::int32_t> ids, const std::vector<VerifiedImageSpan>& imgs,
                                    const StateIdentity& identity, bool is_root, const Parked* keep_entry,
                                    int& req_evictions) {
    const std::size_t n = ids.size();
    const std::size_t min_len = is_root ? kMinRootTokens : kMinParked;
    if (!can_park_any() || n < min_len) return;
    if (has_exact_parked(ids, imgs, identity, is_root)) return;

    auto kept = [&](const Parked& p) { return keep_entry && &p == keep_entry; };

    // A non-root parked prefix of this conversation (same StateIdentity & verified images) is superseded by it.
    // Root checkpoints (`p.is_root == true`) are NEVER erased as superseded so forks immediately after the
    // root boundary still hit the root checkpoint even after a sibling conversation grows longer.
    if (!is_root) {
      bool any_superseded = false;
      for (const Parked& p : parked_) {
        if (!kept(p) && !p.is_root && p.identity == identity && p.ids.size() < n &&
            std::equal(p.ids.begin(), p.ids.end(), ids.begin()) && images_match_prefix(p.images, imgs, p.ids.size())) {
          any_superseded = true;
          break;
        }
      }
      if (any_superseded) {
        CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
        for (auto it = parked_.begin(); it != parked_.end();) {
          if (!kept(*it) && !it->is_root && it->identity == identity && it->ids.size() < n &&
              std::equal(it->ids.begin(), it->ids.end(), ids.begin()) &&
              images_match_prefix(it->images, imgs, it->ids.size())) {
            it = parked_.erase(it);
          } else {
            ++it;
          }
        }
        refresh_cache_usage_locked();
      }
    }

    const ParkLayout l = park_layout(int(n));
    Parked entry;
    entry.ids.assign(ids.begin(), ids.end());
    entry.images = imgs;
    entry.identity = identity;
    entry.id_hash = hash_state_identity(identity);
    entry.prefix_hash = hash_token_prefix(ids);
    entry.is_root = is_root;
    entry.bytes = l.bytes;

    bool allocated = false;
    // Try Tier 0 (VRAM) first if budget fits.
    if (park_budget_ > 0) {
      std::size_t kept_vram = 0;
      std::size_t vram_used = tier_used_bytes(ParkTier::Vram, keep_entry, &kept_vram);
      if (kept_vram + l.bytes <= park_budget_) {
        while (vram_used + l.bytes > park_budget_) {
          auto lru = parked_.end();
          for (auto it = parked_.begin(); it != parked_.end(); ++it) {
            if (it->tier == ParkTier::Vram && !kept(*it) && (lru == parked_.end() || it->used < lru->used)) lru = it;
          }
          if (lru == parked_.end()) break;
          const std::size_t victim_bytes = lru->bytes;
          if (try_demote_to_host(lru, keep_entry, req_evictions)) {
            vram_used -= victim_bytes;
          } else {
            // `lru` remains valid in std::list even if alloc_host_park_slot evicted earlier Tier-1 nodes.
            CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
            vram_used -= victim_bytes;
            parked_.erase(lru);
            ++cache_stats_.vram_evictions;
            ++req_evictions;
          }
        }
        if (vram_used + l.bytes <= park_budget_) {
          try {
            parking_ = true;  // a failure here must not trigger release_parked()
            entry.vram_data = DeviceBuffer(l.bytes);
            parking_ = false;
            entry.tier = ParkTier::Vram;
            allocated = true;
          } catch (const std::exception& e) {
            parking_ = false;
            std::fprintf(stderr, "VRAM conversation parking disabled: %s\n", e.what());
            cudaGetLastError();  // clear sticky error
            CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
            park_budget_ = 0;
            std::size_t dropped = 0;
            for (auto it = parked_.begin(); it != parked_.end();) {
              if (it->tier == ParkTier::Vram && !kept(*it)) {
                it = parked_.erase(it);
                ++dropped;
              } else {
                ++it;
              }
            }
            req_evictions += int(dropped);
            cache_stats_.vram_evictions += dropped;
            refresh_cache_usage_locked();
          }
        }
      }
    }

    // Fall back to Tier 1 (Pinned Host RAM) if Tier 0 was disabled, too small, or failed allocation.
    if (!allocated && host_park_budget_ > 0) {
      PinnedHostBuffer hbuf;
      if (alloc_host_park_slot(l.bytes, keep_entry, hbuf, req_evictions)) {
        entry.host_data = std::move(hbuf);
        entry.tier = ParkTier::Host;
        allocated = true;
      }
    }
    if (!allocated) {
      refresh_cache_usage_locked();
      return;
    }

    char* dst = static_cast<char*>(entry.ptr());
    const char* c = static_cast<const char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    const cudaMemcpyKind kind = (entry.tier == ParkTier::Vram) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    std::size_t copied = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst, c, checkpoint_drafter_, kind, s));
    copied += checkpoint_drafter_;
    for (int i = 0; i < 3 && drafter_; ++i) {
      if (l.drafter[i].bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst + l.drafter_off[i], c + checkpoint_drafter_ + l.drafter[i].offset,
                                   l.drafter[i].bytes, kind, s));
        copied += l.drafter[i].bytes;
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(dst + l.logits, c + checkpoint_logits_, logits_.bytes, kind, s));
    copied += logits_.bytes;
    copied += rt_.save_full_rows(dst + l.rows, int(n));
    if (drafter_) entry.context = drafter_->checkpoint_context();
    entry.used = ++clock_;
    if (entry.tier == ParkTier::Vram) cache_stats_.d2d_park_bytes += copied;
    else cache_stats_.d2h_park_bytes += copied;
    parked_.push_back(std::move(entry));
    refresh_cache_usage_locked();
  }

  // Parks the live conversation's checkpoint (from checkpoint_ + live full rows [0, n)).
  void park_live(const Parked* keep_entry, int& req_evictions) {
    if (checkpoint_ids_.empty()) return;
    store_parked_from_checkpoint(checkpoint_ids_, checkpoint_images_, checkpoint_identity_, /*is_root=*/false,
                                 keep_entry, req_evictions);
  }

  // Captures the live state at an exact verified root-prefix boundary `root_ids` right after prefilling to it.
  // Uses the pre-allocated fixed `checkpoint_` device staging buffer (zero dynamic device staging allocation) and
  // books the exact live-to-staging D2D bytes copied into `cache_stats_.d2d_park_bytes`.
  void capture_root_checkpoint(std::span<const std::int32_t> root_ids, const std::vector<VerifiedImageSpan>& root_imgs,
                               const StateIdentity& identity, int& req_evictions) {
    char* p = static_cast<char*>(checkpoint_.p);
    std::size_t staging_d2d = rt_.save_checkpoint(p);
    if (drafter_) staging_d2d += drafter_->save_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    staging_d2d += logits_.bytes;
    cache_stats_.d2d_park_bytes += staging_d2d;
    store_parked_from_checkpoint(root_ids, root_imgs, identity, /*is_root=*/true, nullptr, req_evictions);
  }

  // Puts a parked conversation (VRAM or pinned host RAM) back into the fixed `checkpoint_` staging buffer and
  // restores its full rows into the target caches. Entry stays parked. Returns exact bytes copied via cudaMemcpyAsync.
  std::size_t unpark(Parked& p) {
    const int n = int(p.ids.size());
    const ParkLayout l = park_layout(n);
    const char* src = static_cast<const char*>(p.ptr());
    char* c = static_cast<char*>(checkpoint_.p);
    const cudaStream_t s = rt_.stream();
    const cudaMemcpyKind kind = (p.tier == ParkTier::Vram) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    std::size_t copied = 0;
    CUDA_CHECK(cudaMemcpyAsync(c, src, checkpoint_drafter_, kind, s));
    copied += checkpoint_drafter_;
    for (int i = 0; i < 3 && drafter_; ++i) {
      if (l.drafter[i].bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_drafter_ + l.drafter[i].offset, src + l.drafter_off[i],
                                   l.drafter[i].bytes, kind, s));
        copied += l.drafter[i].bytes;
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(c + checkpoint_logits_, src + l.logits, logits_.bytes, kind, s));
    copied += logits_.bytes;
    copied += rt_.restore_full_rows(src + l.rows, n);
    if (drafter_) drafter_->adopt_checkpoint(c + checkpoint_drafter_, n, p.context);
    checkpoint_ids_ = p.ids;
    checkpoint_images_ = p.images;
    checkpoint_identity_ = p.identity;
    p.used = ++clock_;
    if (p.tier == ParkTier::Vram) cache_stats_.d2d_restore_bytes += copied;
    else cache_stats_.h2d_restore_bytes += copied;
    return copied;
  }

  // The checkpoint (on the device): target state (Runtime::save_checkpoint) | drafter context
  // (Drafter::save_checkpoint) | logits of the last prompt token.
  void save_checkpoint() {
    char* p = static_cast<char*>(checkpoint_.p);
    rt_.save_checkpoint(p);
    if (drafter_) drafter_->save_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    checkpoint_ids_ = tokens_in_state_;
    checkpoint_images_ = live_images_;
    checkpoint_identity_ = live_identity_;
  }

  // Restores the saved checkpoint into live device state and returns exact bytes copied via cudaMemcpyAsync.
  std::size_t restore_checkpoint() {
    const char* p = static_cast<const char*>(checkpoint_.p);
    std::size_t copied = rt_.restore_checkpoint(p);
    if (drafter_) copied += drafter_->restore_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(logits_.p, p + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    copied += logits_.bytes;
    cache_stats_.d2d_restore_bytes += copied;
    tokens_in_state_ = checkpoint_ids_;
    live_images_ = checkpoint_images_;
    live_identity_ = checkpoint_identity_;
    live_bypass_reuse_ = false;
    return copied;
  }

  static std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

  // engine._sample with a host synchronisation (the speculative loop's first anchor): argmax at
  // temperature <= 0, else one draw from engine._probs.
  std::int32_t sample(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    const head::SamplingParams p = to_head_params(sampling);
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_.p),
                 rt_.stream());
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               rt_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  RuntimeOptions options_;
  std::unique_ptr<Runtime> runtime_;
  Runtime& rt_;
  Prefiller prefiller_;
  std::unique_ptr<Drafter> drafter_;
  std::unique_ptr<SpecLoop> spec_;
  std::unique_ptr<VisionEncoder> vision_;  // image input (RuntimeOptions::vision)
  std::shared_ptr<const PlatosKeyGpu> platos_key_gpu_;  // optional --platos-key projection for /v1/embeddings
  DeviceBuffer chunk_tokens_, logits_, token_, taps_;
  DeviceBuffer masked_logits_, mask_dev_;
  std::vector<std::uint32_t> mask_host_;
  PinnedHostBuffer host_token_;
  // prompt checkpoint (save_checkpoint) — also serves as the fixed device staging buffer for parking
  DeviceBuffer checkpoint_;
  std::size_t checkpoint_drafter_ = 0, checkpoint_logits_ = 0;  // byte offsets in checkpoint_
  cudaEvent_t restore_start_ = nullptr, restore_end_ = nullptr;
  struct HookControl {
    std::mutex m;
    EngineGenerator* gen = nullptr;
  };

  StateIdentity base_identity_;
  bool has_verified_producer_ = false;
  std::array<std::uint8_t, 32> disk_producer_sha_{};
  StateIdentity live_identity_;
  StateIdentity checkpoint_identity_;
  std::vector<std::int32_t> checkpoint_ids_;
  std::vector<std::int32_t> tokens_in_state_;
  std::vector<VerifiedImageSpan> checkpoint_images_;
  std::vector<VerifiedImageSpan> live_images_;
  bool live_bypass_reuse_ = false;
  persona::RefusalMap refusal_map_;
  std::list<Parked> parked_;
  std::size_t park_budget_ = 0;       // Tier-0 VRAM bytes
  std::size_t host_park_budget_ = 0;  // Tier-1 pinned host RAM bytes
  PrefixCacheStats cache_stats_;
  std::uint64_t clock_ = 0;           // monotonic LRU clock
  bool parking_ = false;              // park allocation guard against re-entrant release hook
  bool unparking_ = false;            // restore guard against re-entrant release hook
  const Parked* active_keep_ = nullptr;  // pinned entry currently being restored
  int request_evictions_ = 0;         // parked entries evicted/released during the active request
  std::shared_ptr<HookControl> hook_ctrl_;
  int release_hook_ = -1;
  std::atomic<std::thread::id> active_owner_{};
  mutable std::mutex mutex_;
};

}  // namespace

namespace {

bool has_detectable_model_metadata(const std::filesystem::path& model_dir) {
  if (model_dir.empty()) return false;
  std::error_code ec;
  if (std::filesystem::is_regular_file(model_dir, ec)) {
    return model_dir.extension() == ".gguf";
  }
  if (!std::filesystem::is_directory(model_dir, ec)) return false;
  if (std::filesystem::is_regular_file(model_dir / "config.json", ec) ||
      std::filesystem::is_regular_file(model_dir / "tokenizer.json", ec)) {
    return true;
  }
  for (const auto& entry : std::filesystem::directory_iterator(model_dir, ec)) {
    if (ec) break;
    if (entry.is_regular_file(ec) && entry.path().extension() == ".gguf") return true;
  }
  return false;
}

void validate_loopcd_options(const RuntimeOptions& options) {
  if (!std::isfinite(options.loopcd_omega) ||
      (options.loopcd_omega < 0.0f && options.loopcd_omega != -1.0f) || options.loopcd_omega > 4.0f) {
    throw std::invalid_argument("--loopcd-omega must be finite and in [0, 4]");
  }
  if (options.loopcd_mode == LoopCdMode::Off) return;
  if (options.loop_passes < 2) {
    throw std::invalid_argument("--loopcd requires --loop-passes >= 2 (needs earlier pass L-1 for contrast)");
  }
  if (options.kv_cache_mode != KvCacheMode::PerPass) {
    throw std::invalid_argument("--loopcd requires --kv-cache-mode per_pass");
  }
#if INGINE_TP_SIZE > 1
  throw std::invalid_argument("--loopcd currently requires TP=1");
#endif
  if (options.max_batch > 1) {
    throw std::invalid_argument("--loopcd currently requires --max-batch 1");
  }
}

void validate_small_card_options(const RuntimeOptions& options) {
  if (options.expert_swaps < 0) {
    throw std::invalid_argument("--expert-swaps must be >= 0");
  }
  if (options.expert_swaps > 0) {
    if (options.offload_vram_mb <= 0) {
      throw std::invalid_argument("--expert-swaps > 0 requires --offload-vram-mb > 0 or --atlas-6g");
    }
    if (options.offload_mode == "uva") {
      throw std::invalid_argument(
          "--expert-swaps > 0 is not supported with --offload-mode uva (use auto or dma-lru)");
    }
    if (options.max_batch > 1) {
      throw std::invalid_argument("--expert-swaps > 0 currently requires --max-batch 1");
    }
  }
  if (options.attn_requant != AttnRequantMode::Off) {
    if (options.max_batch > 1) {
      throw std::invalid_argument("--attn-requant q5_k currently requires --max-batch 1");
    }
#if INGINE_TP_SIZE > 1
    throw std::invalid_argument("--attn-requant q5_k currently requires TP=1");
#endif
  }
}

}  // namespace

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options) {
  if (has_detectable_model_metadata(options.model_dir)) {
    const ModelFamily family = detect_model_family(options.model_dir);
    if (family == ModelFamily::Qwen3_5) {
#if INGINE_TP_SIZE > 1
      throw std::invalid_argument("qwen38-dense does not support INGINE_TP_SIZE > 1");
#else
      backends::qwen38::register_qwen38_cuda_backend();
      return BackendRegistry::instance().create(ModelFamily::Qwen3_5, options);
#endif
    }
    if (family == ModelFamily::Qwen4Exp) {
#if INGINE_TP_SIZE > 1
      throw std::invalid_argument("qwen4exp-cuda does not support INGINE_TP_SIZE > 1 yet");
#else
      backends::qwen4exp::register_qwen4exp_cuda_backend();
      return BackendRegistry::instance().create(ModelFamily::Qwen4Exp, options);
#endif
    }
    if (family == ModelFamily::EmbeddingGemma2) {
#if INGINE_TP_SIZE > 1
      throw std::invalid_argument("embeddinggemma2-cuda does not support INGINE_TP_SIZE > 1");
#else
      backends::embedding_gemma2::register_eg2_cuda_backend();
      return BackendRegistry::instance().create(ModelFamily::EmbeddingGemma2, options);
#endif
    }
    if (family != ModelFamily::Inkling2) {
      throw std::invalid_argument(std::string("make_generator: model family '") + to_string(family) +
                                  "' has no enabled CUDA generator backend");
    }
  }
  if (options.loop_passes < 1 || options.loop_passes > MAX_PASSES) {
    throw std::invalid_argument("loop_passes must be in 1.." + std::to_string(MAX_PASSES));
  }
  validate_loopcd_options(options);
  validate_small_card_options(options);
  if (!options.dflash2_dir.empty()) {
    if (!options.spec) {
      throw std::invalid_argument("--dflash2-dir requires --spec");
    }
    if (options.loop_passes != PASSES) {
      throw std::invalid_argument("--dflash2-dir requires --loop-passes 3");
    }
    if (options.kv_cache_mode != KvCacheMode::PerPass) {
      throw std::invalid_argument("--dflash2-dir requires --kv-cache-mode per_pass");
    }
#if INGINE_TP_SIZE > 1
    throw std::invalid_argument("--dflash2-dir currently requires TP=1");
#endif
    if (options.max_batch > 1) {
      throw std::invalid_argument("--dflash2-dir currently requires --max-batch 1 (batched speculation uses DSpark BatchDrafter)");
    }
  }
#if INGINE_TP_SIZE > 1
  if (options.loop_passes != PASSES) {
    throw std::invalid_argument("--loop-passes != 3 currently requires TP=1");
  }
  return make_tp_generator(options);
#endif
  if (options.max_batch > 1) {
    if (options.kv_cache_mode != KvCacheMode::PerPass) {
      throw std::invalid_argument("--kv-cache-mode last_full requires --max-batch 1 (B=1 qualification control)");
    }
    if (options.loop_passes != PASSES) {
      throw std::invalid_argument("--loop-passes != 3 currently requires --max-batch 1");
    }
    if (options.enable_persona_hooks) {
      throw std::invalid_argument("persona hooks require --max-batch 1 (continuous batching with hooked persona tables is not yet enabled)");
    }
    return make_batch_generator(options);
  }
  if (is_gguf_model(options.model_dir)) {
    throw std::invalid_argument("TP1 GGUF models require --max-batch > 1 (batched wgemm path) or TP > 1");
  }
  return std::make_unique<EngineGenerator>(options);
}

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  if (options.loop_passes < 1 || options.loop_passes > MAX_PASSES) {
    throw std::invalid_argument("loop_passes must be in 1.." + std::to_string(MAX_PASSES));
  }
  validate_loopcd_options(options);
  validate_small_card_options(options);
  if (!options.dflash2_dir.empty()) {
    if (!options.spec) {
      throw std::invalid_argument("--dflash2-dir requires --spec");
    }
    if (options.loop_passes != PASSES) {
      throw std::invalid_argument("--dflash2-dir requires --loop-passes 3");
    }
    if (options.kv_cache_mode != KvCacheMode::PerPass) {
      throw std::invalid_argument("--dflash2-dir requires --kv-cache-mode per_pass");
    }
#if INGINE_TP_SIZE > 1
    throw std::invalid_argument("--dflash2-dir currently requires TP=1");
#endif
    if (options.max_batch > 1) {
      throw std::invalid_argument("--dflash2-dir currently requires --max-batch 1 (batched speculation uses DSpark BatchDrafter)");
    }
  }
  if (options.max_batch > 1) {
    if (options.kv_cache_mode != KvCacheMode::PerPass) {
      throw std::invalid_argument("--kv-cache-mode last_full requires --max-batch 1 (B=1 qualification control)");
    }
    if (options.loop_passes != PASSES) {
      throw std::invalid_argument("--loop-passes != 3 currently requires --max-batch 1");
    }
    if (options.enable_persona_hooks) {
      throw std::invalid_argument("persona hooks require --max-batch 1 (continuous batching with hooked persona tables is not yet enabled)");
    }
    return make_batch_generator(options, std::move(weights), std::move(ablation));
  }
  for (const Layer& layer : weights.layers) {
    if (static_cast<int>(layer.experts.w13_fmt) > 5 || static_cast<int>(layer.experts.w2_fmt) > 5) {
      throw std::invalid_argument("TP1 GGUF models require --max-batch > 1 (batched wgemm path) or TP > 1");
    }
  }
  return std::make_unique<EngineGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model
