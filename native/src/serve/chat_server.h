// OpenAI-compatible HTTP server (mirror of engine/ingine/server.py App/Handler/ChatServer): POST
// /v1/chat/completions (streaming SSE and non-streaming), GET /v1/models[/<id>], GET /health.
//
// cpp-httplib gives each connection a pool thread that validates the request, renders the prompt and submits a
// Job to the Scheduler (full -> 503). The request thread drains the job's tokens, runs the StreamParser and
// writes the response; it polls the socket while waiting and cancels the job when the client disconnects (or a
// stop string / stop token ends the turn).
#pragma once

#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "ingine/persona.h"
#include "serve/openai.h"
#include "text/chat.h"

#include <httplib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace ingine::serve {

struct ServerOptions {
  int max_queue = 4;      // requests allowed to wait behind the running one
  int max_model_len = 0;  // context limit; 0: generator.max_model_len()
  std::string model_id;   // model name reported by the API; empty: generator.model_id()
  model::Sampling sampling;  // temperature / top_p / top_k of requests that omit them
  ImageOptions images;       // image input; in effect only when the generator supports images
  std::shared_ptr<const persona::PersonaCatalog> personas;  // optional loaded persona spec catalog
  // Optional trusted proxy HTTP header carrying tenant/scope isolation namespace for prefix caching.
  // When set: requests carrying the header are scoped to its value; missing/empty header fails closed (bypass_prefix_cache=true).
  // When unset (default single trusted scope): requests with Authorization header or JSON "user"/"prompt_cache_key"
  // fail closed (bypass_prefix_cache=true).
  std::string cache_namespace_header;
  // Optional startup system prompt pre-prefilled into the prefix cache before serving (#887).
  // When empty (default): zero startup prefill work and /health reports warm_start=false.
  std::string warm_start_prompt;
  // Cache namespace for startup warm-start; required when combining warm_start_prompt with cache_namespace_header,
  // and rejected when cache_namespace_header is unset.
  std::string warm_start_namespace;
};

class ChatServer {
 public:
  ChatServer(model::Generator& generator, const text::ChatFormat& fmt, ServerOptions options = {});
  ~ChatServer();
  ChatServer(const ChatServer&) = delete;
  ChatServer& operator=(const ChatServer&) = delete;

  // Binds the listening socket (port 0: any free port, see port()).
  bool bind(const std::string& host, int port);
  int port() const { return port_; }
  // Blocking accept loop on the bound socket; returns after stop().
  bool listen();
  // Serves on a background thread (tests).
  void start();
  // Fails waiting jobs, cancels the running one, stops accepting and joins the background thread.
  void stop();

  const std::string& model_id() const { return model_id_; }
  int max_model_len() const { return max_model_len_; }
  engine::Scheduler& scheduler() { return scheduler_; }

 private:
  void register_routes();
  void handle_get(const httplib::Request& req, httplib::Response& res);
  void handle_post(const httplib::Request& req, httplib::Response& res);
  void complete(const httplib::Request& req, httplib::Response& res, ChatRequest request);
  void complete_anthropic(const httplib::Request& req, httplib::Response& res, ChatRequest request);
  void complete_embedding(const httplib::Request& req, httplib::Response& res, ParsedEmbeddingRequest request);
  Json model_card() const;
  Json supported_kv_cache_modes_json() const;
  Json personas_list() const;

  const text::ChatFormat& fmt_;
  const std::string model_id_;
  const std::string model_family_;
  const int max_model_len_;
  const model::KvCacheMode kv_cache_mode_;
  const std::vector<model::KvCacheMode> supported_kv_cache_modes_;
  const int loop_passes_;
  const model::LoopCdMode loopcd_mode_;
  const bool supports_logit_masker_;
  const bool supports_audio_;
  const bool is_encoder_only_;
  const bool supports_embeddings_;
  const int embedding_dim_;
  const int tap_embedding_dim_;
  const int num_tap_layers_;
  const std::vector<int> matryoshka_dims_;
  const std::map<std::string, std::string> embedding_prompt_prefixes_;
  const bool has_platos_key_;
  const int platos_key_out_dim_;
  const std::string platos_key_tap_spec_;
  const std::shared_ptr<const PlatosKey> platos_key_;
  const model::Sampling sampling_defaults_;
  const ImageOptions image_options_;
  const std::shared_ptr<const persona::PersonaCatalog> personas_;
  const std::string cache_namespace_header_;
  bool warm_start_ = false;
  int warm_start_tokens_ = 0;
  const int64_t created_;
  engine::Scheduler scheduler_;
  httplib::Server server_;
  int port_ = 0;
  std::thread thread_;
};

}  // namespace ingine::serve
