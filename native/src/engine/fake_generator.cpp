#include "engine/fake_generator.h"

#include "text/chat.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <optional>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace ingine::engine {
namespace {

constexpr std::string_view kMessageModel = text::MESSAGE_MODEL;
constexpr std::string_view kMessageUser = text::MESSAGE_USER;
constexpr std::string_view kContentText = text::CONTENT_TEXT;
constexpr std::string_view kContentThinking = text::CONTENT_THINKING;
constexpr std::string_view kContentInvokeToolJson = text::CONTENT_INVOKE_TOOL_JSON;
constexpr std::string_view kEndMessage = text::END_MESSAGE;
constexpr std::string_view kEndSampling = text::END_SAMPLING;

// Python's str \s (ASCII part, plus the \x1c-\x1f separators).
bool is_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f); }
// Python's str \w: ASCII letters, digits and '_'; every non-ASCII byte counts as part of a word character.
bool is_word(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

// Number of Unicode code points (Python len) of UTF-8 text.
size_t code_points(std::string_view s) {
  return std::count_if(s.begin(), s.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; });
}

// re.fullmatch(r"\s*call\s+(\w+)\s*(\{.*\})?\s*", s, re.S): {name, args-or-empty}.
std::optional<std::pair<std::string, std::string>> match_call(std::string_view s) {
  size_t i = 0;
  while (i < s.size() && is_space(s[i])) ++i;
  if (s.substr(i, 4) != "call") return std::nullopt;
  i += 4;
  const size_t ws = i;
  while (i < s.size() && is_space(s[i])) ++i;
  if (i == ws) return std::nullopt;
  const size_t name0 = i;
  while (i < s.size() && is_word(s[i])) ++i;
  if (i == name0) return std::nullopt;
  std::string name(s.substr(name0, i - name0));
  while (i < s.size() && is_space(s[i])) ++i;
  size_t end = s.size();
  while (end > i && is_space(s[end - 1])) --end;
  std::string_view rest = s.substr(i, end - i);
  if (rest.empty()) return std::pair{std::move(name), std::string()};
  if (rest.size() >= 2 && rest.front() == '{' && rest.back() == '}') return std::pair{std::move(name), std::string(rest)};
  return std::nullopt;
}

}  // namespace

FakeGenerator::FakeGenerator(const text::Tokenizer& tokenizer, std::string model_id, int max_model_len,
                             Responder responder, double token_delay, int loop_passes,
                             model::KvCacheMode kv_cache_mode, bool allow_last_full)
    : tokenizer_(tokenizer),
      model_id_(std::move(model_id)),
      max_model_len_(max_model_len),
      loop_passes_(loop_passes),
      responder_(std::move(responder)),
      token_delay_(token_delay),
      kv_cache_mode_(kv_cache_mode) {
  if (loop_passes_ < 1 || loop_passes_ > 8) {
    throw std::invalid_argument("loop_passes must be in 1..8");
  }
  const bool can_last_full = allow_last_full && (loop_passes_ == 3);
  if (kv_cache_mode_ == model::KvCacheMode::LastFull && !can_last_full) {
    if (loop_passes_ != 3) {
      throw std::invalid_argument("--loop-passes != 3 is not supported with --kv-cache-mode last_full");
    }
    throw std::invalid_argument("--kv-cache-mode last_full is not supported with this worker configuration");
  }
  if (can_last_full) {
    supported_kv_cache_modes_ = {model::KvCacheMode::PerPass, model::KvCacheMode::LastFull};
  } else {
    supported_kv_cache_modes_ = {model::KvCacheMode::PerPass};
  }
}

void FakeGenerator::set_kv_cache_mode(model::KvCacheMode mode) {
  if (mode == model::KvCacheMode::LastFull && loop_passes_ != 3) {
    throw std::invalid_argument("--loop-passes != 3 is not supported with --kv-cache-mode last_full");
  }
  kv_cache_mode_ = mode;
}

std::string FakeGenerator::echo(std::span<const int32_t> prompt_ids) const {
  const std::string text = tokenizer_.decode(prompt_ids);
  if (!tokenizer_.token_id(kMessageModel).has_value() && tokenizer_.token_id("<|im_start|>").has_value() &&
      tokenizer_.token_id("<|im_end|>").has_value() && tokenizer_.token_id("<think>").has_value() &&
      tokenizer_.token_id("</think>").has_value() && tokenizer_.token_id("<tool_call>").has_value() &&
      tokenizer_.token_id("</tool_call>").has_value()) {
    const std::string qwen_open = "<|im_start|>user\n";
    const std::string qwen_end = "<|im_end|>";
    std::string last;
    for (size_t pos = 0;;) {
      const size_t a = text.find(qwen_open, pos);
      if (a == std::string::npos) break;
      const size_t b = text.find(qwen_end, a + qwen_open.size());
      if (b == std::string::npos) break;
      last = text.substr(a + qwen_open.size(), b - a - qwen_open.size());
      pos = b + qwen_end.size();
    }
    const bool no_thinking = text.ends_with("</think>\n\n");
    const bool prefilled_think = text.ends_with("<think>\n");
    std::string out;
    if (auto call = match_call(last)) {
      auto& [name, args_text] = *call;
      const nlohmann::ordered_json args = nlohmann::ordered_json::parse(args_text.empty() ? "{}" : args_text);
      if (!no_thinking) {
        if (!prefilled_think) out += "<think>\n";
        out += "Calling " + name + ".</think>\n";
      }
      out += "<tool_call>\n<function=" + name + ">\n";
      if (args.is_object()) {
        for (const auto& [k, v] : args.items()) {
          const std::string vs = v.is_string()
                                     ? v.get<std::string>()
                                     : v.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
          out += "<parameter=" + k + ">\n" + vs + "\n</parameter>\n";
        }
      }
      out += "</function>\n</tool_call>" + qwen_end;
      return out;
    }
    if (!no_thinking) {
      if (!prefilled_think) out += "<think>\n";
      out += "The user wrote " + std::to_string(code_points(last)) + " characters.</think>\n\n";
    }
    out += "You said: " + last + qwen_end;
    return out;
  }
  // re.findall(r"<\|message_user\|><\|content_text\|>(.*?)<\|end_message\|>", text, re.S)[-1]
  std::string open = std::string(kMessageUser) + std::string(kContentText);
  std::string last;
  for (size_t pos = 0;;) {
    const size_t a = text.find(open, pos);
    if (a == std::string::npos) break;
    const size_t b = text.find(kEndMessage, a + open.size());
    if (b == std::string::npos) break;
    last = text.substr(a + open.size(), b - a - open.size());
    pos = b + kEndMessage.size();
  }
  std::string out;
  if (auto call = match_call(last)) {
    auto& [name, args_text] = *call;
    nlohmann::ordered_json body = {{"name", name},
                                   {"args", nlohmann::ordered_json::parse(args_text.empty() ? "{}" : args_text)}};
    out.append(kMessageModel).append(kContentThinking).append("Calling ").append(name).append(".");
    out.append(kEndMessage).append(kMessageModel).append(name).append(kContentInvokeToolJson);
    out.append(body.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace));
    out.append(kEndMessage).append(kEndSampling);
    return out;
  }
  std::string answer = "You said: " + last;
  answer.append(kEndMessage).append(kEndSampling);
  const std::string no_thinking = std::string(kMessageModel) + std::string(kContentText);
  if (text.ends_with(no_thinking)) return answer;  // thinking disabled by prefill
  out.append(kMessageModel).append(kContentThinking).append("The user wrote ");
  out.append(std::to_string(code_points(last))).append(" characters.").append(kEndMessage);
  out.append(kMessageModel).append(kContentText).append(answer);
  return out;
}

void FakeGenerator::set_prefill_stats_hook(PrefillStatsFn hook) {
  std::lock_guard lock(mutex_);
  prefill_stats_hook_ = std::move(hook);
}

model::GenerateStats FakeGenerator::warm_start(const model::Prompt& full_prompt, const model::Sampling& sampling) {
  if (full_prompt.ids.empty()) {
    throw std::invalid_argument("warm_start: prompt is empty");
  }
  if (full_prompt.bypass_prefix_cache) {
    throw std::invalid_argument("warm_start: prefix cache bypass requested");
  }
  if (!full_prompt.images.empty()) {
    throw std::invalid_argument("warm_start: images are not supported in startup warm-start prompts");
  }
  if (int(full_prompt.ids.size()) >= max_model_len_) {
    throw std::invalid_argument("warm_start: prompt of " + std::to_string(full_prompt.ids.size()) +
                                " tokens leaves no generation room within max_ctx " +
                                std::to_string(max_model_len_));
  }
  if (sampling.persona && sampling.persona->bypasses_prefix_reuse()) {
    throw std::invalid_argument("warm_start: default persona bypasses prefix cache reuse");
  }
  const std::string persona_fp =
      (sampling.persona && !sampling.persona->is_unhooked()) ? sampling.persona->fingerprint_sha256 : std::string{};
  {
    std::lock_guard lock(mutex_);
    warmed_ids_.assign(full_prompt.ids.begin(), full_prompt.ids.end());
    warmed_namespace_ = full_prompt.cache_namespace;
    warmed_persona_fp_ = persona_fp;
    Call c;
    c.prompt_ids = warmed_ids_;
    c.max_new_tokens = 0;
    c.sampling = sampling;
    c.root_prefix_tokens = full_prompt.root_prefix_tokens;
    c.cache_namespace = full_prompt.cache_namespace;
    c.bypass_prefix_cache = full_prompt.bypass_prefix_cache;
    c.done = true;
    warm_start_calls_.push_back(std::move(c));
  }
  model::GenerateStats stats;
  stats.prompt_tokens = int(full_prompt.ids.size());
  stats.reuse = model::Reuse::None;
  stats.reused_tokens = 0;
  stats.restored_bytes = 0ull;
  return stats;
}

model::GenerateStats FakeGenerator::generate(const model::Prompt& full_prompt, int max_new_tokens,
                                             const model::Sampling& sampling, std::span<const int32_t> stop_ids,
                                             const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                             const model::PrefillCallback& on_prefill) {
  if (sampling.logit_masker && !supports_logit_masker_) {
    throw std::invalid_argument("this generator does not support token-level logit_masker constrained decoding");
  }
  const std::span<const int32_t> prompt = full_prompt.ids;
  int image_tokens = 0;
  for (const model::PromptImage& image : full_prompt.images) image_tokens += image.tokens();
  Output reply = responder_ ? responder_(prompt) : Output(echo(prompt));
  std::vector<int32_t> out = std::holds_alternative<std::string>(reply)
                                 ? tokenizer_.encode(std::get<std::string>(reply), /*special=*/true)
                                 : std::get<std::vector<int32_t>>(std::move(reply));
  size_t index;
  PrefillStatsFn hook;
  std::vector<int32_t> warmed_ids;
  std::string warmed_ns;
  std::string warmed_persona_fp;
  {
    std::lock_guard lock(mutex_);
    index = calls_.size();
    calls_.push_back(Call{std::vector<int32_t>(prompt.begin(), prompt.end()), max_new_tokens, sampling,
                          int(full_prompt.images.size()), image_tokens, full_prompt.root_prefix_tokens,
                          full_prompt.cache_namespace, full_prompt.bypass_prefix_cache});
    hook = prefill_stats_hook_;
    warmed_ids = warmed_ids_;
    warmed_ns = warmed_namespace_;
    warmed_persona_fp = warmed_persona_fp_;
  }
  model::GenerateStats stats;
  stats.prompt_tokens = int(prompt.size());
  stats.image_tokens = image_tokens;
  if (hook) {
    hook(prompt, stats);
  } else if (!warmed_ids.empty() && !full_prompt.bypass_prefix_cache &&
             !(sampling.persona && sampling.persona->bypasses_prefix_reuse())) {
    const std::string req_persona_fp =
        (sampling.persona && !sampling.persona->is_unhooked()) ? sampling.persona->fingerprint_sha256 : std::string{};
    if (full_prompt.cache_namespace == warmed_ns && req_persona_fp == warmed_persona_fp &&
        prompt.size() >= warmed_ids.size() &&
        std::equal(warmed_ids.begin(), warmed_ids.end(), prompt.begin())) {
      stats.reuse = (index == 0 && prompt.size() > warmed_ids.size()) ? model::Reuse::Extend
                                                                      : model::Reuse::RootCheckpoint;
      stats.reused_tokens = int(warmed_ids.size());
      stats.restored_bytes = 0ull;
    }
  }
  if (on_prefill) on_prefill(stats);

  const std::unordered_set<int32_t> stops(stop_ids.begin(), stop_ids.end());
  const size_t limit = size_t(std::max(max_new_tokens, 0));
  bool cancelled = false;
  int yielded = 0;

  if (sampling.logit_masker) {
    // Honor token-level constrained decoding mask on FakeGenerator too (enables CPU E2E HTTP tests).
    model::TokenMasker& masker = *sampling.logit_masker;
    const int V = std::min(masker.vocab_size(), int(tokenizer_.vocab_size()));
    const size_t words = (size_t(V) + 31u) / 32u;
    std::vector<std::uint32_t> mask(words, 0u);
    auto bit_ok = [&](int32_t id) {
      return id >= 0 && id < V && ((mask[size_t(id) >> 5u] >> (uint32_t(id) & 31u)) & 1u) != 0u;
    };
    const int32_t end_msg = tokenizer_.token_id(kEndMessage).value_or(-1);
    const int32_t end_samp = tokenizer_.token_id(kEndSampling).value_or(-1);
    size_t src_i = 0;
    for (size_t step = 0; step < limit; ++step) {
      int32_t chosen = -1;
      if (!masker.is_active()) {
        if (src_i >= out.size()) break;
        chosen = out[src_i++];
      } else {
        std::fill(mask.begin(), mask.end(), 0u);
        const bool masked = masker.fill_mask(mask.data(), words);
        if (!masked) {
          if (src_i >= out.size()) break;
          chosen = out[src_i++];
        } else if (src_i < out.size() && bit_ok(out[src_i])) {
          chosen = out[src_i++];
        } else {
          // Deterministically pick a valid minimal token allowed by the FSM bitmask.
          if (end_samp >= 0 && bit_ok(end_samp)) {
            chosen = end_samp;
          } else if (end_msg >= 0 && bit_ok(end_msg)) {
            chosen = end_msg;
          } else {
            // Prefer closing quotes before structural delimiters so open strings terminate immediately.
            static constexpr const char* kPrefs[] = {"\"", "}", "]", ":", ",", "{", "[", "0", "true", "false", "null", "a"};
            for (const char* p : kPrefs) {
              const auto enc = tokenizer_.encode(p, false);
              if (enc.size() == 1 && bit_ok(enc[0])) {
                chosen = enc[0];
                break;
              }
            }
            if (chosen < 0) {
              for (int32_t id = 0; id < V; ++id) {
                if (bit_ok(id)) {
                  const std::string b = tokenizer_.token_bytes(id);
                  if (!b.empty() && b.size() == 1 && static_cast<unsigned char>(b[0]) >= 0x20 &&
                      static_cast<unsigned char>(b[0]) < 0x7f) {
                    chosen = id;
                    break;
                  }
                }
              }
            }
            if (chosen < 0) {
              for (int32_t id = 0; id < V; ++id) {
                if (bit_ok(id)) {
                  chosen = id;
                  break;
                }
              }
            }
          }
        }
      }
      if (chosen < 0) break;
      masker.advance(chosen);
      if (token_delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(token_delay_));
      {
        std::lock_guard lock(mutex_);
        calls_[index].yielded = ++yielded;
      }
      const bool more = on_tokens(std::span<const int32_t>(&chosen, 1));
      if (stops.contains(chosen)) break;
      if (!more) {
        cancelled = int(step) + 1 < max_new_tokens;
        break;
      }
    }
  } else {
    const size_t n_emit = std::min(out.size(), limit);
    for (size_t i = 0; i < n_emit; ++i) {
      if (token_delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(token_delay_));
      {
        std::lock_guard lock(mutex_);
        calls_[index].yielded = ++yielded;
      }
      const bool more = on_tokens(std::span<const int32_t>(&out[i], 1));
      if (stops.contains(out[i])) break;
      if (!more) {
        // Stop ids and max_new_tokens are handled here as well, so a refusal otherwise means cancellation.
        cancelled = int(i) + 1 < max_new_tokens;
        break;
      }
    }
  }
  {
    std::lock_guard lock(mutex_);
    calls_[index].cancelled = cancelled;
    calls_[index].done = true;
  }
  stats.generated_tokens = yielded;
  return stats;
}

std::vector<FakeGenerator::Call> FakeGenerator::calls() const {
  std::lock_guard lock(mutex_);
  return calls_;
}

std::vector<FakeGenerator::Call> FakeGenerator::warm_start_calls() const {
  std::lock_guard lock(mutex_);
  return warm_start_calls_;
}

void FakeGenerator::set_platos_key(std::shared_ptr<const PlatosKey> key) {
  if (key) {
    key->validate_binding(model_id_, model_family_, embedding_dim_, num_tap_layers_);
  }
  std::lock_guard lock(mutex_);
  platos_key_ = std::move(key);
}

void FakeGenerator::load_platos_key(const std::filesystem::path& safetensors_path,
                                    const std::filesystem::path& metadata_json_path) {
  auto key = PlatosKey::load(safetensors_path, metadata_json_path, embedding_dim_);
  set_platos_key(std::move(key));
}

std::vector<FakeGenerator::EmbeddingCall> FakeGenerator::embedding_calls() const {
  std::lock_guard lock(mutex_);
  return embedding_calls_;
}

model::EmbeddingResult FakeGenerator::embed(const model::EmbeddingRequest& req) {
  if (!supports_embeddings_) {
    throw std::runtime_error("embeddings are disabled on this FakeGenerator");
  }
  if (req.inputs.empty()) {
    throw std::invalid_argument("embed: inputs must not be empty");
  }
  if (req.tap_layer.has_value() && (*req.tap_layer < 0 || *req.tap_layer >= num_tap_layers_)) {
    throw std::invalid_argument("embed: tap_layer " + std::to_string(*req.tap_layer) + " out of range [0, " +
                                std::to_string(num_tap_layers_ - 1) + "]");
  }
  std::shared_ptr<const PlatosKey> pk;
  {
    std::lock_guard lock(mutex_);
    pk = platos_key_;
  }
  if (req.use_platos_key && !pk) {
    throw std::invalid_argument("embed: platos_key requested but no --platos-key is loaded");
  }
  if (req.use_platos_key) {
    pk->validate_request_tap(req.tap_layer);
    pk->validate_request_identity(req.cartridge_id.value_or(""), req.cartridge_sha256.value_or(""));
  }

  const auto t0 = std::chrono::steady_clock::now();
  const int B = static_cast<int>(req.inputs.size());
  const int H = embedding_dim_;
  const int effective_tap = req.tap_layer.value_or(-1);
  // Deterministic causal per-token synthetic hidden state H(seq, pos, layer, dim):
  // strictly causal (depends only on tokens 0..pos) and sensitive to tap_layer so tests verify pooling + early exit.
  const double layer_phase = effective_tap >= 0 ? static_cast<double>(effective_tap + 1) * 0.37 : 99.13;

  int total_tokens = 0;
  std::vector<float> pooled(static_cast<std::size_t>(B) * static_cast<std::size_t>(H), 0.0f);
  for (int b = 0; b < B; ++b) {
    const auto& seq = req.inputs[static_cast<std::size_t>(b)];
    if (seq.empty()) throw std::invalid_argument("embed: input sequence must not be empty");
    if (static_cast<int>(seq.size()) > max_model_len_) {
      throw std::invalid_argument("embed: input sequence length " + std::to_string(seq.size()) +
                                  " exceeds max_model_len " + std::to_string(max_model_len_));
    }
    total_tokens += static_cast<int>(seq.size());
    const int L = static_cast<int>(seq.size());
    auto token_hidden = [&](int pos, int d) -> double {
      std::uint64_t h = 0x9e3779b97f4a7c15ull ^ static_cast<std::uint64_t>(d + 1) * 0xbf58476d1ce4e5b9ull;
      for (int k = 0; k <= pos; ++k) {
        h ^= static_cast<std::uint64_t>(seq[static_cast<std::size_t>(k)] + 1315423911u) * 0x94d049bb133111ebull;
        h = (h << 13) | (h >> 51);
      }
      const double base = static_cast<double>(static_cast<std::int64_t>(h % 2001ull) - 1000ll) / 1000.0;
      return std::sin(base * 2.5 + layer_phase + static_cast<double>(d) * 0.11);
    };
    float* dst = pooled.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(H);
    for (int d = 0; d < H; ++d) {
      if (req.pooling == model::EmbeddingPooling::Cls) {
        dst[d] = static_cast<float>(token_hidden(0, d));
      } else if (req.pooling == model::EmbeddingPooling::LastToken) {
        dst[d] = static_cast<float>(token_hidden(L - 1, d));
      } else {
        double sum = 0.0;
        for (int pos = 0; pos < L; ++pos) sum += token_hidden(pos, d);
        dst[d] = static_cast<float>(sum / static_cast<double>(L));
      }
    }
  }
  const auto t1 = std::chrono::steady_clock::now();

  int out_dim = H;
  std::vector<float> final_flat;
  double pk_seconds = 0.0;
  if (req.use_platos_key) {
    if (req.dimensions.has_value()) {
      throw std::invalid_argument("embed: 'dimensions' cannot be combined with 'platos_key'");
    }
    const auto tp0 = std::chrono::steady_clock::now();
    out_dim = pk->out_dim();
    final_flat.resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(out_dim));
    pk->project_cpu(pooled.data(), B, final_flat.data(), req.normalize);
    pk_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
  } else {
    if (req.dimensions.has_value()) {
      const int d = *req.dimensions;
      if (d <= 0 || d > H) {
        throw std::invalid_argument("embed: dimensions " + std::to_string(d) + " out of range [1, " +
                                    std::to_string(H) + "]");
      }
      if (!matryoshka_dims_.empty() &&
          std::find(matryoshka_dims_.begin(), matryoshka_dims_.end(), d) == matryoshka_dims_.end()) {
        throw std::invalid_argument("embed: dimensions " + std::to_string(d) + " not in matryoshka_dims");
      }
      out_dim = d;
    }
    if (out_dim == H) {
      final_flat = std::move(pooled);
    } else {
      final_flat.resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(out_dim));
      for (int b = 0; b < B; ++b) {
        std::memcpy(final_flat.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim),
                    pooled.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(H),
                    static_cast<std::size_t>(out_dim) * sizeof(float));
      }
    }
    if (req.normalize) l2_normalize_rows_cpu(final_flat.data(), B, out_dim);
  }

  model::EmbeddingResult res;
  res.embeddings.resize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    const float* row = final_flat.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim);
    res.embeddings[static_cast<std::size_t>(b)].assign(row, row + out_dim);
  }
  res.stats.total_prompt_tokens = total_tokens;
  res.stats.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
  res.stats.platos_key_seconds = pk_seconds;
  res.stats.effective_tap_layer = effective_tap;
  res.stats.output_dim = out_dim;
  res.stats.kv_slots_allocated = 0;
  res.stats.lm_head_skipped = true;

  {
    std::lock_guard lock(mutex_);
    embedding_calls_.push_back(EmbeddingCall{req.inputs, req.pooling, req.normalize, req.tap_layer, req.use_platos_key,
                                             req.dimensions, req.prompt_name, res.stats.kv_slots_allocated,
                                             res.stats.lm_head_skipped});
  }
  return res;
}

}  // namespace ingine::engine
