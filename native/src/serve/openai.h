// OpenAI Chat Completions request parsing and JSON shapes, exactly as engine/ingine/server.py
// (Handler._parse_request, _num, _send_error, _delta_json).
#pragma once

#include "image/image.h"
#include "ingine/generator.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::serve {

using Json = nlohmann::ordered_json;

// Invalid request -> HTTP `status` with an OpenAI error body naming `param`.
struct BadRequest : std::runtime_error {
  BadRequest(const std::string& message, std::optional<std::string> param_ = std::nullopt, int status_ = 400)
      : std::runtime_error(message), param(std::move(param_)), status(status_) {}
  std::optional<std::string> param;
  int status;
};

// Image input of chat requests.
struct ImageOptions {
  bool enabled = true;          // false: requests with images are rejected (no vision tower)
  image::Limits limits;         // per image: token limit (larger images are reduced) and source size limits
  int max_images = 16;          // per request
  int max_total_tokens = 32768;  // image tokens per request
};

struct ChatRequest {
  std::vector<int32_t> prompt_ids;   // full prompt, prefill included; images expanded to their tokens
  std::vector<int32_t> prefill_ids;  // tail of prompt_ids the model would otherwise have generated
  std::vector<model::PromptImage> images;  // by position in prompt_ids
  int root_prefix_tokens = 0;        // verified leading system+tools+effort boundary in prompt_ids
  int max_new_tokens = 0;
  model::Sampling sampling;
  std::vector<std::string> stop;
  std::string persona_name;  // optional per-request persona profile selector (empty = server default)
  text::RenderStats render_stats;  // CPU tokenization cache stats (distinct from GPU prefix reuse)
  bool has_isolation_hint = false;  // true if request carries "user" or "prompt_cache_key" (signals tenant isolation)
  bool stream = false;
  bool include_usage = false;
  bool has_user_or_cache_key = false;  // true if JSON body specifies non-empty `user` or `prompt_cache_key`
};

// Validates `body` (a JSON object) and renders the prompt. Throws BadRequest or text::ChatError. Sampling fields
// the request omits (temperature, top_p, top_k) take their values from `defaults`.
//
// Images are content parts of user and tool messages: {"type": "image_url", "image_url": {"url": URI} | URI},
// {"type": "input_image", "image_url": URI}, {"type": "image", "image" | "url": URI} or the Anthropic shape
// {"type": "image", "source": {"type": "base64", "data": BASE64}}; URI is a base64 data: URI (nothing is
// fetched). The template renders one placeholder token per image; each is replaced by the image's tokens
// (image::token_id), so prompt_ids, the context check and the usage count the expanded prompt.
ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults = {}, const ImageOptions& images = {},
                          std::string_view raw_body = {}, bool allow_logit_masker = true);

// {"error": {"message", "type", "param", "code"}} with the type server.py maps from the status.
Json error_body(int status, std::string_view message, const std::optional<std::string>& param = std::nullopt);

// One streamed delta: reasoning_content, content or a tool_calls entry.
Json delta_json(const text::Delta& delta);

// json.dumps(ensure_ascii=False) equivalent (compact separators); invalid UTF-8 becomes U+FFFD.
std::string dump(const Json& value);

// Python truthiness of a JSON value.
bool truthy(const Json& value);

enum class EmbeddingEncodingFormat : std::uint8_t {
  Float = 0,
  Base64 = 1,
};

struct ParsedEmbeddingRequest {
  model::EmbeddingRequest request;
  EmbeddingEncodingFormat encoding_format = EmbeddingEncodingFormat::Float;
};

struct EmbeddingParseOptions {
  int max_model_len = 0;
  int embedding_dim = 0;
  int tap_embedding_dim = 0;     // intermediate tap dim when distinct from embedding_dim (0 => embedding_dim)
  int num_tap_layers = 0;
  bool has_platos_key = false;
  const PlatosKey* loaded_platos_key = nullptr;
  bool supports_images = false;
  bool supports_audio = false;
  int vision_soft_tokens_per_image = 280;
  int video_soft_tokens_per_frame = 140;
  image::Limits image_limits;
  int max_images = 16;           // per-request image + video frame limit across the entire batch (<0 disables)
  int max_total_tokens = 32768;  // per-request image + video soft token limit across the entire batch (<0 disables)
  std::vector<int> matryoshka_dims;
  std::map<std::string, std::string> prompt_prefixes;
};

// Validates and tokenizes a POST /v1/embeddings request body (standard OpenAI + Matryoshka dimensions +
// optional task prompt_name + opt-in Atlas extensions). Throws BadRequest (HTTP 400) on invalid parameters.
// When `opts.loaded_platos_key` is non-null, enforces its bound tap_layer/tap_spec and cartridge identity
// fail-closed whenever `platos_key=true`.
ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt,
                                               const EmbeddingParseOptions& opts);
ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                               int num_tap_layers, const PlatosKey* loaded_platos_key);
ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                               int num_tap_layers, bool has_platos_key);

// Encodes contiguous little-endian IEEE-754 float32 values as RFC 4648 base64 (OpenAI encoding_format="base64").
std::string encode_embedding_base64(std::span<const float> vec);

// Formats exact OpenAI POST /v1/embeddings response JSON.
Json embedding_response_json(const std::string& model_id, const model::EmbeddingResult& result,
                             EmbeddingEncodingFormat format);

}  // namespace ingine::serve
