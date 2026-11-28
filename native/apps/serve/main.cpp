// ingine-serve: the OpenAI-compatible chat completions server (flags and defaults of `python -m ingine.server`).
#include "backends/embedding_gemma2/eg2_config.h"
#include "backends/embedding_gemma2/eg2_cpu.h"
#include "engine/fake_generator.h"
#include "engine/log.h"
#include "image/image.h"
#include "ingine/generator.h"
#include "ingine/persona.h"
#include "serve/chat_server.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#if defined(__linux__) && defined(INGINE_WITH_MODEL)
#include <cuda_runtime.h>
#endif

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>

namespace {

int g_stop_pipe[2] = {-1, -1};

void handle_signal(int) {
  const char c = 's';
  (void)!::write(g_stop_pipe[1], &c, 1);
}

// tokenizer_config.json "model_max_length" (the Python server's fallback context limit), else 1 << 20.
int tokenizer_max_length(const std::filesystem::path& model_dir) {
  std::ifstream in(model_dir / "tokenizer_config.json");
  if (!in) return 1 << 20;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("model_max_length"); it != cfg.end() && it->is_number()) {
      return int(std::min<double>(it->get<double>(), 2147483647.0));
    }
  } catch (const nlohmann::json::exception&) {
  }
  return 1 << 20;
}

// Sampling of requests that omit temperature / top_p / top_k: generation_config.json's values where present,
// else ingine's defaults (temperature 1, no top-p, no top-k).
ingine::model::Sampling generation_defaults(const std::filesystem::path& model_dir) {
  ingine::model::Sampling s;
  std::ifstream in(model_dir / "generation_config.json");
  if (!in) return s;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("temperature"); it != cfg.end() && it->is_number()) s.temperature = it->get<float>();
    if (auto it = cfg.find("top_p"); it != cfg.end() && it->is_number()) s.top_p = it->get<float>();
    if (auto it = cfg.find("top_k"); it != cfg.end() && it->is_number_integer()) s.top_k = std::max(0, it->get<int>());
  } catch (const nlohmann::json::exception&) {
  }
  return s;
}

#if defined(__linux__) && defined(INGINE_WITH_MODEL)
bool is_geforce_rtx_20_30_40(std::string_view name) {
  return name.find("GeForce RTX 20") != std::string_view::npos ||
         name.find("GeForce RTX 30") != std::string_view::npos ||
         name.find("GeForce RTX 40") != std::string_view::npos;
}

void maybe_log_gpu_unlock_notice(spdlog::logger& log) {
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return;
  const std::string_view name(prop.name);
  if (!is_geforce_rtx_20_30_40(name)) return;
  std::error_code ec;
  if (std::filesystem::exists("/sys/module/gpu_unlock", ec)) return;
  log.info("Linux GeForce RTX 20/30/40 ({}) without /sys/module/gpu_unlock: for optional multi-GPU P2P/compute tuning see external https://github.com/Zagitta/gpu-unlock (GPL-2.0, not bundled)",
           name);
}
#endif

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, host = "127.0.0.1", ablation, dflash2_dir, backend = "fast", model_id, log_level = "INFO";
  std::string offload_mode = "auto", expert_ranking, dump_router_stats, kv_swap_dir, kv_ckpt_path, kv_ckpt_resume;
  std::string kv_cache_mode = "per_pass";
  std::string loopcd_mode = "off";
  std::string attn_requant = "off";
  std::string persona_spec, refusal_map, prompt_renderer = "ingine_chat_format_v1", cache_namespace_header;
  std::string warm_start_prompt, warm_start_file, warm_start_namespace;
  std::string platos_key, platos_key_metadata;
  int port = 8003, max_ctx = 65536, prefill_chunk = 3072, max_queue = 4, max_model_len = 0, park_mb = -1,
      park_host_mb = 0;
  int max_batch = 1, max_kv_tokens = 0, loop_passes = 3;
  int offload_vram_mb = 0, offload_ram_mb = 12288, expert_swaps = 0;
  int kv_vram_mb = 0, kv_ram_mb = 12288, kv_ckpt_every = 0;
  bool engine = false, fake = false, dense_fp8 = false, kv_fp8 = false, kv_fp4v = false, kv_q4 = false, kv_paged = false, spec = false,
       no_vision = false, no_audio = false, atlas_6g = false, loop_bf16 = false, persona_probe_mode = false, no_prefix_cache = false;
  int image_max_tokens = 4096, max_images = 16, max_image_tokens = 32768;
  double spec_threshold = 0.8, fake_delay = 0.02, loopcd_omega = -1.0;
  // sampling defaults; negative: from generation_config.json
  double temperature = -1, top_p = -1;
  int top_k = -1;

  ingine::serve::ArgParser ap("ingine-serve", "OpenAI-compatible chat completions server for ingine.");
  ap.option("--model-dir", "", "checkpoint dir (tokenizer + chat template)", &model_dir, {}, true);
  ap.option("--host", "", "", &host);
  ap.option("--port", "", "", &port);
  ap.flag("--engine", "serve the model with ingine", &engine);
  ap.flag("--fake", "serve canned outputs (FakeGenerator)", &fake);
  ap.exclusive({"--engine", "--fake"}, true);
  ap.option("--ablation", "", "ablation file (safetensors) applied at every visit", &ablation);
  ap.option("--persona-spec", "", "persona/mech-interp JSON spec (atlas-persona-spec/v1)", &persona_spec);
  ap.flag("--persona-probe-mode", "allow probe/unvalidated persona profiles (research & diagnostics only)",
          &persona_probe_mode);
  ap.option("--refusal-map", "", "refusal-linked unit deny-list JSON (unconditionally blocks listed units)",
            &refusal_map);
  ap.option("--prompt-renderer", "",
            "canonical prompt renderer contract checked against persona model_binding.prompt_renderer",
            &prompt_renderer);
  ap.option("--max-ctx", "", "positions held by the full-attention caches", &max_ctx);
  ap.option("--prefill-chunk", "", "tokens per prefill chunk (default 3072; use 512 on 12 GB GPUs for 262K ctx)",
            &prefill_chunk);
  ap.option("--backend", "{fast}", "decode step implementation", &backend, {"fast"});
  ap.flag("--dense-fp8", "FP8 E4M3 dense projections and head (row scales)", &dense_fp8);
  ap.flag("--spec", "speculative decoding (DSpark mtp.safetensors, or DFlash2 with --dflash2-dir)", &spec);
  ap.option("--dflash2-dir", "", "native Apache-2.0 DFlash2DraftModel directory (config.json + model.safetensors; implies --spec)",
            &dflash2_dir);
  ap.flag("--kv-fp8", "FP8 E4M3 full-attention KV caches (half the memory; needed for 262144 positions on 32 GB)",
          &kv_fp8);
  ap.flag("--kv-fp4v", "FP8 keys + FP4 values in the TP2 batch full-attention pool (implies --kv-fp8)", &kv_fp4v);
  ap.flag("--kv-q4", "Q4_0 4-bit full-attention KV caches (9 KiB/tok per_pass, 4.5 KiB/tok last_full; SWA stays BF16)",
          &kv_q4);
  ap.option("--kv-cache-mode", "{per_pass,last_full}",
            "full-attention KV mode: per_pass (default, 8 visit slots) or last_full (opt-in L11/L17 shared across passes, 4 slots; B=1 non-spec)",
            &kv_cache_mode, {"per_pass", "last_full"});
  ap.flag("--kv-paged", "3-tier paged Q4_0 full-attention KV cache (VRAM LRU + pinned UVA RAM + NVMe SwapFile)", &kv_paged);
  ap.option("--kv-vram-mb", "", "VRAM cap in MiB for paged Q4_0 KV cache (implies --kv-paged --kv-q4)", &kv_vram_mb);
  ap.option("--kv-ram-mb", "", "pinned UVA host RAM cap in MiB for paged Q4_0 KV before NVMe SwapFile spill (default 12288)",
            &kv_ram_mb);
  ap.option("--kv-swap-dir", "", "optional directory for Tier-2 NVMe KV swap file", &kv_swap_dir);
  ap.option("--kv-ckpt-path", "", "atomic disk KV checkpoint file path (periodic & auto-resume on prefix match)",
            &kv_ckpt_path);
  ap.option("--kv-ckpt-every", "", "save atomic disk KV checkpoint every N tokens during long prefill (0 = off)",
            &kv_ckpt_every);
  ap.option("--kv-ckpt-resume", "", "explicit disk KV checkpoint file to resume from (fails loudly on mismatch)",
            &kv_ckpt_resume);
  ap.exclusive({"--kv-fp8", "--kv-q4"}, false);
  ap.flag("--atlas-6g", "6 GB VRAM tiered offload preset (offload-vram-mb=5888, dense-fp8, kv-fp8 unless kv-q4, prefill-chunk=1024, no-vision)",
          &atlas_6g);
  ap.option("--offload-vram-mb", "", "tiered MoE offloading VRAM cap in MiB (0 = off, e.g. 5888 for 6 GB GPU)",
            &offload_vram_mb);
  ap.option("--offload-ram-mb", "", "pinned host RAM cap in MiB for WARM experts (default 12288)", &offload_ram_mb);
  ap.option("--offload-mode", "{auto,dma-lru,uva}", "tiered offload transport mode", &offload_mode,
            {"auto", "dma-lru", "uva"});
  ap.option("--expert-ranking", "", "optional JSON with per-layer [24][128] homology/frequency scores", &expert_ranking);
  ap.option("--dump-router-stats", "", "write observed [24][128] router hits JSON on exit", &dump_router_stats);
  ap.flag("--loop-bf16",
          "opt-in small-card knob: elide unused FP32 controller & router duplicates (-81.0 MiB VRAM, 0-ULP parity)",
          &loop_bf16);
  ap.option("--attn-requant", "{off,q5_k}",
            "opt-in small-card attention requantization for wproj+wo: off (default) or q5_k (GGML Q5_K 5.5 b/w)",
            &attn_requant, {"off", "q5_k"});
  ap.option("--expert-swaps", "",
            "opt-in bounded async GPU<->RAM HOT-promotion swap queue capacity (default 0 = static HOT + LRU; >0 requires offload and non-uva; NO expert dropping)",
            &expert_swaps);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--park-mb", "",
            "device memory (MiB) for conversations displaced by another one, released when memory runs short, "
            "0 = off (default: min(3072, free memory after start-up - 4096))",
            &park_mb);
  ap.option("--park-host-mb", "",
            "pinned host RAM cap (MiB) for Tier-1 parked checkpoints (LRU eviction, graceful fallback; default 0 = off)",
            &park_host_mb);
  ap.flag("--no-prefix-cache", "disable all prefix/root/parked checkpoint reuse (cache-off parity baseline)",
          &no_prefix_cache);
  ap.option("--cache-namespace-header", "",
            "trusted proxy HTTP header isolating prefix-cache namespaces (absent/empty fails closed to bypass)",
            &cache_namespace_header);
  ap.option("--warm-start-prompt", "",
            "system prompt text pre-prefilled into the prefix cache at startup (sets warm_start=true in /health)",
            &warm_start_prompt);
  ap.option("--warm-start-file", "",
            "UTF-8 file containing a system prompt pre-prefilled into the prefix cache at startup",
            &warm_start_file);
  ap.option("--warm-start-namespace", "",
            "prefix-cache namespace for startup warm-start (required when combining warm-start with --cache-namespace-header)",
            &warm_start_namespace);
  ap.exclusive({"--warm-start-prompt", "--warm-start-file"}, false);
  ap.option("--platos-key", "",
            "optional Plato's Key .safetensors projection for POST /v1/embeddings (enabled per-request via platos_key=true)",
            &platos_key);
  ap.option("--platos-key-metadata", "",
            "optional sidecar JSON metadata file validated against --platos-key tensor dimensions",
            &platos_key_metadata);
  ap.flag("--no-vision", "text only: do not load the vision tower (saves 0.6 GB) and reject requests with images",
          &no_vision);
  ap.flag("--no-audio", "do not load the audio tower (saves 583 MiB on EmbeddingGemma 2) and reject audio inputs",
          &no_audio);
  ap.option("--image-max-tokens", "",
            "tokens of one image (one per 40x40 pixels); a larger image is reduced to fit (at most 10609)",
            &image_max_tokens);
  ap.option("--max-images", "", "images accepted in one request", &max_images);
  ap.option("--max-image-tokens", "", "image tokens accepted in one request", &max_image_tokens);
  ap.option("--fake-delay", "", "seconds per fake token", &fake_delay);
  ap.option("--model-id", "", "model name reported by the API (default: generator.model_id)", &model_id);
  ap.option("--max-queue", "", "requests allowed to wait behind the running one", &max_queue);
  ap.option("--max-batch", "", "max concurrent sequences for continuous batching (1 = single-request mode)", &max_batch);
  ap.option("--max-kv-tokens", "", "shared paged full-KV token pool capacity for --max-batch > 1 (0 = auto)", &max_kv_tokens);
  ap.flag("--no-prefix-cache", "disable prefix checkpoint caching (cache-off parity baseline for B=1 and B>1)",
          &no_prefix_cache);
  ap.option("--cache-namespace-header", "",
            "trusted proxy header scoping prefix cache per tenant (default: single trusted server scope; requests "
            "with Authorization/user/prompt_cache_key fail-closed bypass reuse when unset)",
            &cache_namespace_header);
  ap.option("--loop-passes", "", "recurrent loop passes over layers 6..17 (1..8, default 3 = 48 visits)", &loop_passes);
  ap.option("--loopcd", "{off,logits-adaptive,hidden}",
            "LoopCD loop-contrastive decoding (arXiv:2610.02185): off (default), logits-adaptive (shared-coda approximation + fused single-pass lm_head), or hidden (pre-coda Eq. 2)",
            &loopcd_mode, {"off", "logits-adaptive", "hidden"});
  ap.option("--loopcd-omega", "",
            "LoopCD contrast strength in [0, 4] (default: 0.5 for logits-adaptive, 0.3 for hidden; 0 = exact no-op control)",
            &loopcd_omega);
  ap.option("--max-model-len", "",
            "context limit (default: generator.max_model_len or tokenizer model_max_length)", &max_model_len);
  ap.option("--log-level", "", "", &log_level);
  ap.option("--temperature", "", "temperature of requests that omit it (default: generation_config.json, else 1)",
            &temperature);
  ap.option("--top-p", "", "top_p of requests that omit it (default: generation_config.json, else 1)", &top_p);
  ap.option("--top-k", "", "top_k of requests that omit it, 0 = off (default: generation_config.json, else 0)",
            &top_k);
  ap.parse(argc, argv);
  if (!dflash2_dir.empty()) spec = true;
  if (!dflash2_dir.empty() && max_batch > 1) {
    std::cerr << "ingine-serve: error: --dflash2-dir currently requires --max-batch 1 (batched speculation uses DSpark BatchDrafter)\n";
    return 2;
  }
  if (image_max_tokens < 1 || image_max_tokens > ingine::image::MAX_TOKENS || max_images < 0 || max_image_tokens < 0) {
    std::cerr << "ingine-serve: error: --image-max-tokens must be in [1, " << ingine::image::MAX_TOKENS
              << "] and --max-images, --max-image-tokens at least 0\n";
    return 2;
  }
  if (loop_passes < 1 || loop_passes > 8) {
    std::cerr << "ingine-serve: error: --loop-passes must be in [1, 8]\n";
    return 2;
  }
  const auto parsed_loopcd_mode = ingine::model::parse_loopcd_mode(loopcd_mode);
  const auto parsed_attn_requant = ingine::model::parse_attn_requant_mode(attn_requant);
  if (expert_swaps < 0) {
    std::cerr << "ingine-serve: error: --expert-swaps must be >= 0\n";
    return 2;
  }
  if (ap.given("--loopcd-omega") && (!std::isfinite(loopcd_omega) || loopcd_omega < 0.0 || loopcd_omega > 4.0)) {
    std::cerr << "ingine-serve: error: --loopcd-omega must be finite and in [0, 4]\n";
    return 2;
  }
  if (temperature > 2 || top_p == 0 || top_p > 1) {
    std::cerr << "ingine-serve: error: --temperature must be in [0, 2] and --top-p in (0, 1]\n";
    return 2;
  }
  const bool has_warm_start = ap.given("--warm-start-prompt") || ap.given("--warm-start-file");
  if (has_warm_start) {
    if (no_prefix_cache) {
      std::cerr << "ingine-serve: error: --warm-start-prompt/--warm-start-file cannot be combined with --no-prefix-cache\n";
      return 2;
    }
    if (!cache_namespace_header.empty() && warm_start_namespace.empty()) {
      std::cerr << "ingine-serve: error: --warm-start-prompt/--warm-start-file with --cache-namespace-header requires --warm-start-namespace\n";
      return 2;
    }
    if (cache_namespace_header.empty() && !warm_start_namespace.empty()) {
      std::cerr << "ingine-serve: error: --warm-start-namespace requires --cache-namespace-header\n";
      return 2;
    }
    if (ap.given("--warm-start-file")) {
      std::error_code ec;
      if (!std::filesystem::is_regular_file(warm_start_file, ec)) {
        std::cerr << "ingine-serve: error: --warm-start-file is not a readable regular file: '" << warm_start_file
                  << "'\n";
        return 2;
      }
      std::ifstream in(warm_start_file, std::ios::binary);
      if (!in) {
        std::cerr << "ingine-serve: error: cannot open --warm-start-file '" << warm_start_file << "'\n";
        return 2;
      }
      warm_start_prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (warm_start_prompt.find_first_not_of(" \t\r\n") == std::string::npos) {
      std::cerr << "ingine-serve: error: warm-start system prompt must contain non-whitespace text\n";
      return 2;
    }
  } else if (!warm_start_namespace.empty()) {
    std::cerr << "ingine-serve: error: --warm-start-namespace requires --warm-start-prompt or --warm-start-file\n";
    return 2;
  }
  if (!platos_key_metadata.empty() && platos_key.empty()) {
    std::cerr << "ingine-serve: error: --platos-key-metadata requires --platos-key\n";
    return 2;
  }

  auto& log = ingine::engine::server_log();
  if (!ingine::engine::set_log_level(log_level)) {
    std::cerr << "ingine-serve: error: unknown level: '" << log_level << "'\n";
    return 2;
  }
  std::signal(SIGPIPE, SIG_IGN);

  try {
    const ingine::persona::RefusalMap rmap = refusal_map.empty() ? ingine::persona::RefusalMap::default_map()
                                                                 : ingine::persona::RefusalMap::load(refusal_map);
    std::shared_ptr<const ingine::persona::PersonaCatalog> personas;
    if (!persona_spec.empty()) {
      ingine::persona::ModelRuntimeIdentity runtime_id;
      runtime_id.prompt_renderer = prompt_renderer;
      const std::filesystem::path tmpl_path = std::filesystem::path(model_dir) / "chat_template.jinja";
      if (std::filesystem::is_regular_file(tmpl_path)) {
        runtime_id.chat_template_sha256 = ingine::persona::sha256_file_hex(tmpl_path);
      }
      personas = std::make_shared<ingine::persona::PersonaCatalog>(
          ingine::persona::PersonaCatalog::load(persona_spec, persona_probe_mode, rmap, runtime_id));
      log.info("loaded persona spec {} ({} profile(s), default='{}', probe_mode={})",
               personas->spec_sha256.substr(0, 12), personas->profiles.size(), personas->default_profile,
               personas->probe_mode ? "on" : "off");
      for (const auto& [pname, prof] : personas->profiles) {
        if (!prof->downgrade_reason.empty()) {
          log.warn("persona '{}': downgraded to {} ({})", pname, ingine::persona::to_string(prof->effective_status),
                   prof->downgrade_reason);
        }
      }
    }

    const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
    const bool is_qwen35 = (fmt.kind() == ingine::text::ChatFormatKind::Qwen3_5);
    const bool is_eg2 = (fmt.kind() == ingine::text::ChatFormatKind::EmbeddingGemma2);
    const bool loop_passes_explicit = ap.given("--loop-passes");
    if (is_eg2) {
      if (loop_passes_explicit) {
        throw std::invalid_argument("embedding_gemma2 is an encoder-only model and does not support --loop-passes");
      }
      if (parsed_loopcd_mode != ingine::model::LoopCdMode::Off) {
        throw std::invalid_argument("embedding_gemma2 is an encoder-only model and does not support --loopcd");
      }
      if (loop_bf16) {
        throw std::invalid_argument("embedding_gemma2 does not support --loop-bf16");
      }
      if (parsed_attn_requant != ingine::model::AttnRequantMode::Off) {
        throw std::invalid_argument("embedding_gemma2 does not support --attn-requant");
      }
      if (expert_swaps != 0 || offload_vram_mb != 0) {
        throw std::invalid_argument("embedding_gemma2 has no MoE experts and does not support MoE offload options");
      }
      if (spec) {
        throw std::invalid_argument("embedding_gemma2 is an encoder-only model and does not support --spec");
      }
      if (kv_fp8 || kv_q4 || kv_paged || kv_vram_mb != 0) {
        throw std::invalid_argument("embedding_gemma2 is stateless (zero KV cache) and does not support KV cache flags");
      }
      if (!ablation.empty() || personas != nullptr || persona_probe_mode || !refusal_map.empty()) {
        throw std::invalid_argument("embedding_gemma2 does not support Inkling ablation/persona hooks");
      }
      if (!ap.given("--max-ctx")) max_ctx = 8192;
      // EmbeddingGemma 2 supports text + vision + audio natively; --no-vision / --no-audio opt out cleanly.
    }
    if (is_qwen35) {
      if (loop_passes_explicit) {
        throw std::invalid_argument(
            "qwen38-dense is a non-looped dense architecture and does not support --loop-passes");
      }
      if (parsed_loopcd_mode != ingine::model::LoopCdMode::Off) {
        throw std::invalid_argument(
            "qwen38-dense is a non-looped dense architecture and does not support --loopcd");
      }
      if (loop_bf16) {
        throw std::invalid_argument(
            "qwen38-dense is a non-looped dense architecture and does not support --loop-bf16");
      }
      if (parsed_attn_requant != ingine::model::AttnRequantMode::Off) {
        throw std::invalid_argument("qwen38-dense does not support --attn-requant");
      }
      if (expert_swaps != 0) {
        throw std::invalid_argument("qwen38-dense has no MoE experts and does not support --expert-swaps");
      }
      if (!ap.given("--max-ctx")) max_ctx = 8192;
      if (!ap.given("--prefill-chunk")) prefill_chunk = 256;
      // Phase 1 Qwen3.8 backend is text-only (ingine-serve defaults vision enabled for Inkling).
      no_vision = true;
    }
    std::unique_ptr<ingine::model::Generator> generator;
    if (fake) {
      const auto parsed_kv_mode = ingine::model::parse_kv_cache_mode(kv_cache_mode);
      if (is_eg2) {
        auto eg2_cfg = ingine::backends::embedding_gemma2::Eg2Config::load(model_dir);
        auto efake = std::make_unique<ingine::engine::FakeGenerator>(
            fmt.tokenizer(), "embeddinggemma-2-fake", max_ctx, ingine::engine::FakeGenerator::Responder{}, fake_delay,
            /*loop_passes=*/1, ingine::model::KvCacheMode::PerPass, /*allow_last_full=*/false);
        efake->set_model_family("embedding_gemma2");
        efake->set_is_encoder_only(true);
        efake->set_supports_logit_masker(false);
        efake->set_supports_images(false);
        efake->set_supports_audio(false);
        efake->set_embedding_dim(eg2_cfg.embedding_dim);
        efake->set_tap_embedding_dim(eg2_cfg.hidden_size);
        efake->set_num_tap_layers(eg2_cfg.num_hidden_layers);
        efake->set_matryoshka_dims(eg2_cfg.matryoshka_dims);
        efake->set_embedding_prompt_prefixes(eg2_cfg.prompts);
        if (!platos_key.empty()) efake->load_platos_key(platos_key, platos_key_metadata);
        generator = std::move(efake);
      } else if (is_qwen35) {
        if (parsed_kv_mode != ingine::model::KvCacheMode::PerPass) {
          throw std::invalid_argument("qwen38-dense does not support --kv-cache-mode " + kv_cache_mode +
                                      " (only per_pass is supported)");
        }
        if (spec) throw std::invalid_argument("qwen38-dense does not support --spec");
        if (max_batch > 1) throw std::invalid_argument("qwen38-dense Phase 1 requires --max-batch 1");
        if (kv_fp8 || kv_q4 || kv_paged || kv_vram_mb != 0) {
          throw std::invalid_argument("qwen38-dense does not support Inkling --kv-fp8/--kv-q4/--kv-paged options");
        }
        if (offload_vram_mb != 0 || park_mb > 0 || park_host_mb > 0) {
          throw std::invalid_argument("qwen38-dense does not support Inkling offload/parking options");
        }
        if (!ablation.empty() || personas != nullptr || persona_probe_mode || !refusal_map.empty()) {
          throw std::invalid_argument("qwen38-dense does not support Inkling ablation/persona hooks");
        }
        auto qfake = std::make_unique<ingine::engine::FakeGenerator>(
            fmt.tokenizer(), "Qwen3.8-27B-NVFP4-fake", tokenizer_max_length(model_dir),
            ingine::engine::FakeGenerator::Responder{}, fake_delay, /*loop_passes=*/1,
            ingine::model::KvCacheMode::PerPass, /*allow_last_full=*/false);
        qfake->set_model_family("qwen3_5");
        qfake->set_supports_logit_masker(false);
        qfake->set_supports_images(false);
        qfake->set_embedding_dim(5120);
        qfake->set_num_tap_layers(64);
        if (!platos_key.empty()) qfake->load_platos_key(platos_key, platos_key_metadata);
        generator = std::move(qfake);
      } else {
        if (loop_passes != 3) {
          if (parsed_kv_mode == ingine::model::KvCacheMode::LastFull) {
            throw std::invalid_argument("--loop-passes != 3 is not supported with --kv-cache-mode last_full");
          }
          if (spec) {
            throw std::invalid_argument("--loop-passes != 3 is not supported with --spec (DSpark drafter requires 3 passes)");
          }
          if (!ablation.empty()) {
            throw std::invalid_argument("--loop-passes != 3 is not supported with --ablation (48-visit ablation file)");
          }
          if (personas != nullptr || persona_probe_mode) {
            throw std::invalid_argument("--loop-passes != 3 is not supported with persona hooks (fixed 48-visit hook ABI)");
          }
          if (max_batch > 1) {
            throw std::invalid_argument("--loop-passes != 3 currently requires --max-batch 1");
          }
        }
        if (parsed_kv_mode == ingine::model::KvCacheMode::LastFull) {
          if (spec) {
            throw std::invalid_argument("--kv-cache-mode last_full cannot be combined with --spec (B=1 non-spec control)");
          }
          if (max_batch > 1) {
            throw std::invalid_argument("--kv-cache-mode last_full requires --max-batch 1 (B=1 qualification control)");
          }
        }
        if (parsed_loopcd_mode != ingine::model::LoopCdMode::Off) {
          if (loop_passes < 2) {
            throw std::invalid_argument("--loopcd requires --loop-passes >= 2");
          }
          if (parsed_kv_mode == ingine::model::KvCacheMode::LastFull) {
            throw std::invalid_argument("--loopcd requires --kv-cache-mode per_pass");
          }
          if (max_batch > 1) {
            throw std::invalid_argument("--loopcd currently requires --max-batch 1");
          }
          if (parsed_loopcd_mode == ingine::model::LoopCdMode::Hidden && !ablation.empty()) {
            throw std::invalid_argument("--loopcd hidden cannot be combined with --ablation");
          }
        }
        const int eff_offload_vram = (atlas_6g && offload_vram_mb == 0) ? 5888 : offload_vram_mb;
        if (expert_swaps > 0) {
          if (eff_offload_vram <= 0) {
            throw std::invalid_argument("--expert-swaps > 0 requires --offload-vram-mb > 0 or --atlas-6g");
          }
          if (offload_mode == "uva") {
            throw std::invalid_argument(
                "--expert-swaps > 0 is not supported with --offload-mode uva (use auto or dma-lru)");
          }
          if (max_batch > 1) {
            throw std::invalid_argument("--expert-swaps > 0 currently requires --max-batch 1");
          }
        }
        if (parsed_attn_requant != ingine::model::AttnRequantMode::Off && max_batch > 1) {
          throw std::invalid_argument("--attn-requant q5_k currently requires --max-batch 1");
        }
        const bool fake_allow_last_full =
            (loop_passes == 3) && !spec && (max_batch <= 1) && (parsed_loopcd_mode == ingine::model::LoopCdMode::Off);
        auto ifake = std::make_unique<ingine::engine::FakeGenerator>(
            fmt.tokenizer(), "Inkling-2-Mini-fake", tokenizer_max_length(model_dir),
            ingine::engine::FakeGenerator::Responder{}, fake_delay, loop_passes, parsed_kv_mode, fake_allow_last_full);
        ifake->set_loopcd_mode(parsed_loopcd_mode);
        ifake->set_embedding_dim(2304);
        ifake->set_num_tap_layers(6 + loop_passes * 12 + 6);
        if (!platos_key.empty()) ifake->load_platos_key(platos_key, platos_key_metadata);
        generator = std::move(ifake);
      }
    } else {
#ifdef INGINE_WITH_MODEL
      ingine::model::RuntimeOptions options;
      if (atlas_6g) {
        if (offload_vram_mb == 0) offload_vram_mb = 5888;
        dense_fp8 = true;
        if (kv_paged || kv_vram_mb > 0) kv_q4 = true;
        if (!kv_q4) kv_fp8 = true;
        no_vision = true;
        if (prefill_chunk > 1024) prefill_chunk = 1024;
        if (park_mb < 0) park_mb = 0;
        if (!kv_paged && kv_vram_mb <= 0 && max_ctx > 32768) max_ctx = 32768;
      }
      options.model_dir = model_dir;
      options.ablation = ablation;
      options.max_ctx = max_ctx;
      options.prefill_chunk = prefill_chunk;
      options.dense_fp8 = dense_fp8;
      options.kv_fp8 = kv_fp8 || kv_fp4v;
      options.kv_fp4v = kv_fp4v;
      options.kv_q4 = kv_q4;
      options.kv_cache_mode = ingine::model::parse_kv_cache_mode(kv_cache_mode);
      options.kv_paged = kv_paged || kv_vram_mb > 0;
      options.kv_vram_mb = kv_vram_mb;
      options.kv_ram_mb = kv_ram_mb;
      options.kv_swap_dir = kv_swap_dir;
      options.kv_ckpt_path = kv_ckpt_path;
      options.kv_ckpt_every = kv_ckpt_every;
      options.kv_ckpt_resume = kv_ckpt_resume;
      options.spec = spec;
      options.dflash2_dir = dflash2_dir;
      options.spec_threshold = float(spec_threshold);
      options.park_mb = park_mb;
      options.park_host_mb = park_host_mb;
      options.prefix_cache = !no_prefix_cache;
      options.vision = !no_vision;
      options.audio = !no_audio;
      options.offload_vram_mb = offload_vram_mb;
      options.offload_ram_mb = offload_ram_mb;
      options.offload_mode = offload_mode;
      options.expert_ranking = expert_ranking;
      options.dump_router_stats = dump_router_stats;
      options.max_batch = max_batch;
      options.max_kv_tokens = max_kv_tokens;
      options.loop_passes = loop_passes;
      options.loop_passes_explicit = loop_passes_explicit;
      options.loopcd_mode = parsed_loopcd_mode;
      options.loopcd_omega = ap.given("--loopcd-omega") ? float(loopcd_omega) : -1.0f;
      options.loop_bf16 = loop_bf16;
      options.attn_requant = parsed_attn_requant;
      options.expert_swaps = expert_swaps;
      options.enable_persona_hooks = (personas != nullptr) || persona_probe_mode;
      options.refusal_map = refusal_map;
      options.platos_key = platos_key;
      options.platos_key_metadata = platos_key_metadata;
      generator = ingine::model::make_generator(options);
#if defined(__linux__)
      maybe_log_gpu_unlock_notice(log);
#endif
#else
      if (is_eg2) {
        ingine::model::RuntimeOptions options;
        options.model_dir = model_dir;
        options.max_ctx = max_ctx;
        options.vision = !no_vision;
        options.audio = !no_audio;
        options.platos_key = platos_key;
        options.platos_key_metadata = platos_key_metadata;
        generator = std::make_unique<ingine::backends::embedding_gemma2::Eg2CpuBackend>(options);
      } else {
        log.critical("--engine: this ingine-serve was built without ingine_model (src/model)");
        return 1;
      }
#endif
    }

    ingine::serve::ServerOptions server_options;
    server_options.max_queue = max_queue;
    server_options.max_model_len = max_model_len;
    server_options.model_id = model_id;
    server_options.sampling = generation_defaults(model_dir);
    if (temperature >= 0) server_options.sampling.temperature = float(temperature);
    if (top_p >= 0) server_options.sampling.top_p = float(top_p);
    if (top_k >= 0) server_options.sampling.top_k = top_k;
    server_options.images.enabled = !no_vision;
    server_options.images.limits.max_tokens = image_max_tokens;
    server_options.images.max_images = max_images;
    server_options.images.max_total_tokens = max_image_tokens;
    server_options.personas = personas;
    server_options.cache_namespace_header = cache_namespace_header;
    server_options.warm_start_prompt = warm_start_prompt;
    server_options.warm_start_namespace = warm_start_namespace;
    log.info("default sampling: temperature {} top_p {} top_k {}", server_options.sampling.temperature,
             server_options.sampling.top_p, server_options.sampling.top_k);
    ingine::serve::ChatServer server(*generator, fmt, server_options);
    if (!server.bind(host, port)) {
      log.critical("cannot bind {}:{}", host, port);
      return 1;
    }

    if (::pipe(g_stop_pipe) != 0) throw std::runtime_error("pipe() failed");
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    log.info("image input: {}", !no_vision && generator->supports_images()
                                    ? fmt::format("on (at most {} tokens per image, {} images and {} image tokens per request)",
                                                  image_max_tokens, max_images, max_image_tokens)
                                    : std::string("off"));
    log.info("serving {} on http://{}:{}/v1 (kv_cache_mode={}, loop_passes={}, loopcd={}, embeddings={}, platos_key={})",
             server.model_id(), host, server.port(), ingine::model::to_string(generator->kv_cache_mode()),
             generator->loop_passes(), ingine::model::to_string(generator->loopcd_mode()),
             generator->supports_embeddings() ? fmt::format("on(dim={},taps={})", generator->embedding_dim(),
                                                            generator->num_tap_layers())
                                              : std::string("off"),
             generator->has_platos_key() ? fmt::format("on(out_dim={})", generator->platos_key_out_dim())
                                         : std::string("off"));
    bool listened = true;
    std::thread listener([&] {
      listened = server.listen();
      handle_signal(0);  // the accept loop ended on its own: wake the main thread
    });
    char c;
    while (::read(g_stop_pipe[0], &c, 1) < 0 && errno == EINTR) {
    }
    server.stop();
    listener.join();
    if (!listened) {
      log.critical("listen failed on {}:{}", host, server.port());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    log.critical("{}", e.what());
    return 1;
  }
}
