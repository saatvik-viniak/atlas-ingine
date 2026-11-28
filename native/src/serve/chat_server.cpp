#include "serve/chat_server.h"

#include "engine/log.h"
#include "serve/anthropic.h"
#include "serve/http_transport.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ingine::serve {
namespace {

using engine::Clock;

constexpr auto kPollInterval = std::chrono::milliseconds(50);  // between client-disconnect checks

// self.path.split("?", 1)[0].rstrip("/")
std::string route_path(const httplib::Request& req) {
  std::string path = req.target.substr(0, req.target.find('?'));
  while (!path.empty() && path.back() == '/') path.pop_back();
  return path;
}

void send_json(httplib::Response& res, int status, const Json& body) {
  res.status = status;
  res.set_content(dump(body), "application/json");
}

void send_error(httplib::Response& res, int status, std::string_view message,
                const std::optional<std::string>& param = std::nullopt) {
  send_json(res, status, error_body(status, message, param));
}

void send_anthropic_error(httplib::Response& res, int status, std::string_view message) {
  send_json(res, status, anthropic_error_body(status, message));
}

std::string sse(std::string_view payload) {
  std::string out;
  out.reserve(payload.size() + 8);
  out.append("data: ").append(payload).append("\n\n");
  return out;
}

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

// One request's state on its request thread.
struct Exchange {
  Exchange(std::shared_ptr<engine::Job> job_, text::StreamParser parser_, ChatRequest req_)
      : job(std::move(job_)), parser(std::move(parser_)), req(std::move(req_)), created(std::time(nullptr)) {}

  std::shared_ptr<engine::Job> job;
  text::StreamParser parser;
  ChatRequest req;
  int64_t created;
  std::optional<std::string> failure;
  std::optional<Clock::time_point> t_first;
  std::optional<model::GenerateStats> prefill_stats;
};

// Drains the job's output into the parser until it reports a finish reason; `gone()` is polled every
// kPollInterval and ends the exchange with ClientDisconnected. `emit` receives every batch's deltas.
void pump(Exchange& x, const std::function<bool()>& gone, const std::function<void(std::vector<text::Delta>&)>& emit) {
  Clock::time_point last_check = Clock::now();
  for (;;) {
    engine::JobBatch batch = x.job->take(kPollInterval);
    const Clock::time_point now = Clock::now();
    if (now - last_check >= kPollInterval) {
      last_check = now;
      if (gone()) throw ClientDisconnected();
    }
    if (batch.empty()) continue;
    std::vector<text::Delta> deltas;
    for (int32_t tok : batch.tokens) {
      if (x.parser.finished()) continue;
      if (!x.t_first) x.t_first = Clock::now();
      std::vector<text::Delta> d = x.parser.feed(tok);
      deltas.insert(deltas.end(), std::make_move_iterator(d.begin()), std::make_move_iterator(d.end()));
      if (x.parser.finished()) x.job->cancel();  // stop string / stop token: nothing more to generate
    }
    if (batch.failure) x.failure = std::move(batch.failure);
    if (x.parser.finished() || batch.end) {
      std::vector<text::Delta> d = x.parser.finish();
      deltas.insert(deltas.end(), std::make_move_iterator(d.begin()), std::make_move_iterator(d.end()));
    }
    if (!deltas.empty()) emit(deltas);
    if (x.parser.finish_reason()) break;
  }
}

Json usage(const Exchange& x) {
  const int prompt = int(x.job->prompt_ids.size());
  const int completion = x.parser.num_tokens();
  Json details = Json::object();
  if (x.prefill_stats) {
    details["cached_tokens"] = std::clamp(x.prefill_stats->reused_tokens, 0, prompt);
  } else {
    details["cached_tokens"] = nullptr;
  }
  return Json{{"prompt_tokens", prompt},
              {"completion_tokens", completion},
              {"total_tokens", prompt + completion},
              {"prompt_tokens_details", std::move(details)}};
}

void log_done(const Exchange& x, model::KvCacheMode kv_cache_mode) {
  const Clock::time_point now = Clock::now();
  const engine::Job& job = *x.job;
  const double queued = seconds(job.t_start().value_or(now) - job.t_submit);
  const int prompt = int(job.prompt_ids.size());
  const int n = x.parser.num_tokens();
  const auto& pst = x.prefill_stats;
  const int cached = pst ? std::clamp(pst->reused_tokens, 0, prompt) : 0;
  const int recomputed = std::max(0, prompt - cached);
  const char* reuse = pst ? model::to_string(pst->reuse) : "none";
  const std::string restored_bytes =
      (pst && pst->restored_bytes) ? std::to_string(*pst->restored_bytes) : std::string("n/a");
  const std::string rate =
      x.t_first && n > 1 ? fmt::format("{:.1f} tok/s", (n - 1) / std::max(seconds(now - *x.t_first), 1e-9)) : "-";
  const std::string ttft = x.t_first ? fmt::format("{:.3f}s", seconds(*x.t_first - job.t_submit)) : "-";
  const auto& finish = x.parser.finish_reason();
  engine::server_log().info(
      "{}: kv_cache_mode={} prompt={} cached={} recomputed={} reuse={} restored_bytes={} render_reused={} "
      "render_ms={:.2f} completion={} finish={} queued={:.3f}s ttft={} decode={}{}",
      job.id, model::to_string(kv_cache_mode), prompt, cached, recomputed, reuse, restored_bytes,
      x.req.render_stats.reused_text_tokens, x.req.render_stats.render_ms, n, finish ? *finish : std::string("None"),
      queued, ttft, rate, x.failure ? " error=" + *x.failure : std::string());
}

void log_disconnected(const Exchange& x) {
  engine::server_log().info("{}: client disconnected after {} tokens; cancelled", x.job->id, x.parser.num_tokens());
}

Json persona_meta_json(const persona::PersonaProfile& p) {
  return Json{{"profile", p.name},
              {"persona_fingerprint", p.fingerprint_sha256},
              {"status", persona::to_string(p.effective_status)},
              {"spec_sha256", p.spec_sha256}};
}

Json chunk(const Exchange& x, const std::string& model_id, Json delta, const std::optional<std::string>& finish) {
  Json c{{"id", x.job->id},
         {"object", "chat.completion.chunk"},
         {"created", x.created},
         {"model", model_id},
         {"system_fingerprint", nullptr},
         {"choices", Json::array({Json{{"index", 0},
                                       {"delta", std::move(delta)},
                                       {"logprobs", nullptr},
                                       {"finish_reason", finish ? Json(*finish) : Json()}}})}};
  if (x.req.include_usage) c["usage"] = nullptr;
  if (finish && x.req.sampling.persona) c["persona_metadata"] = persona_meta_json(*x.req.sampling.persona);
  return c;
}

struct StreamState {
  explicit StreamState(std::shared_ptr<Exchange> x_) : x(std::move(x_)) {}
  std::shared_ptr<Exchange> x;
  std::atomic<bool> cancelled{false};
  std::atomic<bool> started{false};
};

}  // namespace

ChatServer::ChatServer(model::Generator& generator, const text::ChatFormat& fmt, ServerOptions options)
    : fmt_(fmt),
      model_id_(options.model_id.empty() ? generator.model_id() : options.model_id),
      model_family_(generator.model_family()),
      max_model_len_(options.max_model_len ? options.max_model_len : generator.max_model_len()),
      kv_cache_mode_(generator.kv_cache_mode()),
      supported_kv_cache_modes_(generator.supported_kv_cache_modes()),
      loop_passes_(generator.loop_passes()),
      loopcd_mode_(generator.loopcd_mode()),
      supports_logit_masker_(generator.supports_logit_masker()),
      supports_audio_(generator.supports_audio()),
      is_encoder_only_(generator.is_encoder_only()),
      supports_embeddings_(generator.supports_embeddings()),
      embedding_dim_(generator.embedding_dim()),
      num_tap_layers_(generator.num_tap_layers()),
      matryoshka_dims_(generator.matryoshka_dims()),
      embedding_prompt_prefixes_(generator.embedding_prompt_prefixes()),
      has_platos_key_(generator.has_platos_key()),
      platos_key_out_dim_(generator.platos_key_out_dim()),
      platos_key_tap_spec_(generator.platos_key_tap_spec()),
      platos_key_(generator.platos_key()),
      sampling_defaults_(options.sampling),
      image_options_([&] {
        ImageOptions images = options.images;
        images.enabled = images.enabled && generator.supports_images();
        return images;
      }()),
      personas_(std::move(options.personas)),
      cache_namespace_header_(std::move(options.cache_namespace_header)),
      created_(std::time(nullptr)),
      scheduler_(generator, fmt.stop_ids(), options.max_queue) {
  if (platos_key_) {
    platos_key_->validate_binding(model_id_, model_family_, embedding_dim_, num_tap_layers_);
  }
  if (!options.warm_start_prompt.empty()) {
    if (!cache_namespace_header_.empty() && options.warm_start_namespace.empty()) {
      throw std::invalid_argument(
          "warm_start_prompt with cache_namespace_header requires a non-empty warm_start_namespace");
    }
    if (cache_namespace_header_.empty() && !options.warm_start_namespace.empty()) {
      throw std::invalid_argument("warm_start_namespace requires cache_namespace_header to be configured");
    }
    const text::RenderedPrompt rendered = fmt_.render_system_prefix(options.warm_start_prompt);
    if (int(rendered.ids.size()) >= max_model_len_) {
      throw std::invalid_argument("warm-start system prompt has " + std::to_string(rendered.ids.size()) +
                                  " tokens; leaves no generation room within max_model_len " +
                                  std::to_string(max_model_len_));
    }
    model::Sampling ws_sampling = sampling_defaults_;
    if (personas_) {
      ws_sampling.persona = personas_->resolve("");
    }
    const model::Prompt ws_prompt{rendered.ids, {}, rendered.root_prefix_tokens, options.warm_start_namespace,
                                  /*bypass_prefix_cache=*/false};
    const model::GenerateStats ws_stats = generator.warm_start(ws_prompt, ws_sampling);
    warm_start_ = true;
    warm_start_tokens_ = int(rendered.ids.size());
    engine::server_log().info("warm-start prefilled {} system-prompt tokens in {:.3f}s (namespace='{}')",
                              warm_start_tokens_, ws_stats.prefill_seconds, options.warm_start_namespace);
  } else if (!options.warm_start_namespace.empty()) {
    throw std::invalid_argument("warm_start_namespace requires warm_start_prompt");
  }
  // Size the HTTP pool for concurrent active slots + waiting requests + /health probes (never starve at B=64).
  const int http_threads = std::max(16, generator.max_batch_size() + options.max_queue + 16);
  server_.new_task_queue = [http_threads] { return new httplib::ThreadPool(std::size_t(http_threads), 1024); };
  server_.set_socket_options(configure_http_server_socket);
  register_routes();
}

ChatServer::~ChatServer() { stop(); }

void ChatServer::register_routes() {
  server_.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    engine::server_log().debug("{} \"{} {} {}\" {} -", req.remote_addr, req.method, req.target, req.version,
                               res.status);
  });
  server_.Get(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) { handle_get(req, res); });
  server_.Post(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) { handle_post(req, res); });
}

Json ChatServer::supported_kv_cache_modes_json() const {
  Json arr = Json::array();
  for (model::KvCacheMode m : supported_kv_cache_modes_) {
    const std::string s = model::to_string(m);
    if (std::find(arr.begin(), arr.end(), s) == arr.end()) arr.push_back(s);
  }
  const std::string active = model::to_string(kv_cache_mode_);
  if (std::find(arr.begin(), arr.end(), active) == arr.end()) arr.push_back(active);
  return arr;
}

Json ChatServer::model_card() const {
  Json modalities = Json::array({"text"});
  if (image_options_.enabled) modalities.push_back("image");
  if (supports_audio_) modalities.push_back("audio");
  Json card{{"id", model_id_},
            {"object", "model"},
            {"created", created_},
            {"owned_by", "ingine"},
            {"model_family", model_family_},
            {"max_model_len", max_model_len_},
            {"kv_cache_mode", model::to_string(kv_cache_mode_)},
            {"supported_kv_cache_modes", supported_kv_cache_modes_json()},
            {"kv_cache_mode_mutable", false},
            {"loop_passes", loop_passes_},
            {"loopcd", model::to_string(loopcd_mode_)},
            {"supports_structured_output", supports_logit_masker_},
            {"supports_embeddings", supports_embeddings_},
            {"embedding_dim", embedding_dim_},
            {"num_tap_layers", num_tap_layers_},
            {"platos_key_loaded", has_platos_key_},
            {"platos_key_out_dim", platos_key_out_dim_},
            {"input_modalities", std::move(modalities)}};
  if (is_encoder_only_) card["encoder_only"] = true;
  if (!matryoshka_dims_.empty()) card["matryoshka_dims"] = matryoshka_dims_;
  if (!embedding_prompt_prefixes_.empty()) {
    Json pnames = Json::array();
    for (const auto& [k, _] : embedding_prompt_prefixes_) pnames.push_back(k);
    card["supported_prompt_names"] = std::move(pnames);
  }
  if (!platos_key_tap_spec_.empty()) card["platos_key_tap_spec"] = platos_key_tap_spec_;
  if (platos_key_) {
    const auto& pm = platos_key_->metadata();
    card["platos_key_sha256"] = pm.file_sha256;
    card["platos_key_layout"] = pm.tensor_layout;
    if (!pm.donor_model.empty()) card["platos_key_donor_model"] = pm.donor_model;
    if (!pm.cartridge_id.empty()) card["platos_key_cartridge_id"] = pm.cartridge_id;
    if (pm.expected_tap_layer.has_value()) card["platos_key_tap_layer"] = *pm.expected_tap_layer;
  }
  return card;
}

Json ChatServer::personas_list() const {
  if (!personas_) {
    return Json{{"object", "list"},
                {"default", "none"},
                {"probe_mode", false},
                {"spec_sha256", nullptr},
                {"data", Json::array()}};
  }
  Json data = Json::array();
  for (const auto& [name, prof] : personas_->profiles) {
    Json entry{{"id", name},
               {"object", "persona"},
               {"status", persona::to_string(prof->effective_status)},
               {"declared_category", persona::to_string(prof->declared_category)},
               {"persona_fingerprint", prof->fingerprint_sha256},
               {"interventions", prof->interventions.size()},
               {"description", prof->description}};
    if (!prof->downgrade_reason.empty()) entry["downgrade_reason"] = prof->downgrade_reason;
    data.push_back(std::move(entry));
  }
  return Json{{"object", "list"},
              {"default", personas_->default_profile},
              {"probe_mode", personas_->probe_mode},
              {"spec_sha256", personas_->spec_sha256},
              {"data", std::move(data)}};
}

void ChatServer::handle_get(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
  if (path == "/health") {
    const engine::SchedulerStats st = scheduler_.stats();
    res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
    res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
    res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
    Json health{{"status", "ok"},
                {"model", model_id_},
                {"model_family", model_family_},
                {"kv_cache_mode", model::to_string(kv_cache_mode_)},
                {"supported_kv_cache_modes", supported_kv_cache_modes_json()},
                {"kv_cache_mode_mutable", false},
                {"loop_passes", loop_passes_},
                {"loopcd", model::to_string(loopcd_mode_)},
                {"supports_structured_output", supports_logit_masker_},
                {"supports_embeddings", supports_embeddings_},
                {"embedding_dim", embedding_dim_},
                {"num_tap_layers", num_tap_layers_},
                {"platos_key_loaded", has_platos_key_},
                {"platos_key_out_dim", platos_key_out_dim_},
                {"warm_start", warm_start_},
                {"warm_start_tokens", warm_start_tokens_},
                {"waiting", st.waiting},
                {"running", st.running},
                {"active", st.active},
                {"embed_waiting", st.embed_waiting},
                {"embed_running", st.embed_running},
                {"max_batch", st.max_batch},
                {"max_queue", st.max_queue}};
    Json modalities = Json::array({"text"});
    if (image_options_.enabled) modalities.push_back("image");
    if (supports_audio_) modalities.push_back("audio");
    health["input_modalities"] = std::move(modalities);
    if (is_encoder_only_) health["encoder_only"] = true;
    if (!matryoshka_dims_.empty()) health["matryoshka_dims"] = matryoshka_dims_;
    if (!platos_key_tap_spec_.empty()) health["platos_key_tap_spec"] = platos_key_tap_spec_;
    if (platos_key_) {
      const auto& pm = platos_key_->metadata();
      health["platos_key_sha256"] = pm.file_sha256;
      health["platos_key_layout"] = pm.tensor_layout;
      if (!pm.donor_model.empty()) health["platos_key_donor_model"] = pm.donor_model;
      if (!pm.cartridge_id.empty()) health["platos_key_cartridge_id"] = pm.cartridge_id;
      if (pm.expected_tap_layer.has_value()) health["platos_key_tap_layer"] = *pm.expected_tap_layer;
    }
    send_json(res, 200, health);
  } else if (path == "/v1/models") {
    res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
    res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
    res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
    send_json(res, 200, Json{{"object", "list"}, {"data", Json::array({model_card()})}});
  } else if (path == "/v1/models/" + model_id_) {
    res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
    res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
    res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
    send_json(res, 200, model_card());
  } else if (path == "/v1/personas") {
    send_json(res, 200, personas_list());
  } else {
    send_error(res, 404, "no route for GET " + path);
  }
}

void ChatServer::handle_post(const httplib::Request& req, httplib::Response& res) {
  const std::string path = route_path(req);
  const bool is_openai = (path == "/v1/chat/completions");
  const bool is_embeddings = (path == "/v1/embeddings");
  const bool is_anthropic = (path == "/v1/messages");
  const bool is_anthropic_count = (path == "/v1/messages/count_tokens");
  if (!is_openai && !is_embeddings && !is_anthropic && !is_anthropic_count) {
    send_error(res, 404, "no route for POST " + path);
    return;
  }
  auto fail_post = [&](int status, std::string_view msg, const std::optional<std::string>& param = std::nullopt) {
    if (is_anthropic || is_anthropic_count) send_anthropic_error(res, status, msg);
    else send_error(res, status, msg, param);
  };
  if (!req.has_header("Content-Length")) {
    res.set_header("Connection", "close");
    fail_post(411, "Content-Length required");
    return;
  }
  if (is_embeddings) {
    if (!supports_embeddings_) {
      send_error(res, 400, "embeddings are not supported by this model backend");
      return;
    }
    ParsedEmbeddingRequest emb_req;
    try {
      Json body;
      try {
        body = Json::parse(req.body);
      } catch (const Json::parse_error& e) {
        send_error(res, 400, std::string("invalid JSON body: ") + e.what());
        return;
      }
      if (!body.is_object()) throw BadRequest("request body must be a JSON object");
      EmbeddingParseOptions popts;
      popts.max_model_len = max_model_len_;
      popts.embedding_dim = embedding_dim_;
      popts.num_tap_layers = num_tap_layers_;
      popts.has_platos_key = has_platos_key_;
      popts.loaded_platos_key = platos_key_.get();
      popts.supports_images = image_options_.enabled;
      popts.supports_audio = supports_audio_;
      popts.image_limits = image_options_.limits;
      popts.matryoshka_dims = matryoshka_dims_;
      popts.prompt_prefixes = embedding_prompt_prefixes_;
      emb_req = parse_embedding_request(body, fmt_, popts);
    } catch (const BadRequest& e) {
      send_error(res, e.status, e.what(), e.param);
      return;
    } catch (const text::ChatError& e) {
      send_error(res, 400, e.what(), e.param);
      return;
    }
    complete_embedding(req, res, std::move(emb_req));
    return;
  }
  if (is_encoder_only_) {
    fail_post(400, "model '" + model_id_ + "' is an embeddings-only encoder; use POST /v1/embeddings", "model");
    return;
  }
  if (req.has_header("X-Ingine-Expected-Loop-Passes")) {
    const std::string raw_lp = req.get_header_value("X-Ingine-Expected-Loop-Passes");
    int expected_lp = 0;
    bool valid_lp = !raw_lp.empty();
    for (char c : raw_lp) {
      if (c < '0' || c > '9') {
        valid_lp = false;
        break;
      }
      expected_lp = expected_lp * 10 + (c - '0');
      if (expected_lp > 8) {
        valid_lp = false;
        break;
      }
    }
    if (!valid_lp || expected_lp < 1 || expected_lp > 8) {
      res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
      res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
      fail_post(400, "invalid X-Ingine-Expected-Loop-Passes header: expected integer in 1..8",
                "X-Ingine-Expected-Loop-Passes");
      return;
    }
    if (expected_lp != loop_passes_) {
      res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
      res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
      fail_post(409,
                "worker loop_passes mismatch: expected=" + std::to_string(expected_lp) +
                    ", server=" + std::to_string(loop_passes_),
                "X-Ingine-Expected-Loop-Passes");
      return;
    }
  }
  ChatRequest request;
  try {
    Json body;
    try {
      body = Json::parse(req.body);
    } catch (const Json::parse_error& e) {
      fail_post(400, std::string("invalid JSON body: ") + e.what());
      return;
    }
    if (!body.is_object()) throw BadRequest("request body must be a JSON object");
    if (is_openai) {
      request = parse_request(body, fmt_, max_model_len_, sampling_defaults_, image_options_, req.body,
                              supports_logit_masker_);
      if (request.sampling.logit_masker && !supports_logit_masker_) {
        throw BadRequest(
            "structured JSON response_format ('json_object'/'json_schema') is not supported by this model backend",
            "response_format");
      }
    } else {
      request = parse_anthropic_request(body, fmt_, max_model_len_, sampling_defaults_, image_options_,
                                        /*require_max_tokens=*/is_anthropic);
    }
    if (is_anthropic_count) {
      res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
      res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
      send_json(res, 200, Json{{"input_tokens", int(request.prompt_ids.size())}});
      return;
    }
    if (personas_) {
      try {
        request.sampling.persona = personas_->resolve(request.persona_name);
      } catch (const std::invalid_argument& e) {
        throw BadRequest(e.what(), "persona");
      }
    } else if (!request.persona_name.empty() && request.persona_name != "none") {
      throw BadRequest("unknown persona '" + request.persona_name + "' (no --persona-spec loaded)", "persona");
    }
  } catch (const BadRequest& e) {
    fail_post(e.status, e.what(), e.param);
    return;
  } catch (const text::ChatError& e) {
    fail_post(400, e.what(), e.param);
    return;
  }
  if (is_anthropic) complete_anthropic(req, res, std::move(request));
  else complete(req, res, std::move(request));
}

void ChatServer::complete(const httplib::Request& req, httplib::Response& res, ChatRequest request) {
  std::string cache_namespace;
  bool bypass_prefix_cache = false;
  if (!cache_namespace_header_.empty()) {
    // Trusted reverse-proxy scope mode: require the configured header; fail closed if absent/empty.
    if (req.has_header(cache_namespace_header_)) {
      cache_namespace = req.get_header_value(cache_namespace_header_);
    }
    if (cache_namespace.empty()) bypass_prefix_cache = true;
  } else {
    // Single-trusted-scope server mode: fail closed for requests signaling caller/tenant isolation.
    const bool has_auth = (req.has_header("Authorization") && !req.get_header_value("Authorization").empty()) ||
                          (req.has_header("x-api-key") && !req.get_header_value("x-api-key").empty());
    if (request.has_isolation_hint || request.has_user_or_cache_key || has_auth) bypass_prefix_cache = true;
  }
  auto job = std::make_shared<engine::Job>(request.prompt_ids, request.max_new_tokens, request.sampling,
                                           std::move(request.images), request.root_prefix_tokens,
                                           std::move(cache_namespace), bypass_prefix_cache);
  if (!scheduler_.submit(job)) {
    res.set_header("Retry-After", "1");
    send_error(res, 503, "server busy: request queue is full");
    return;
  }
  text::StreamParser parser = fmt_.parser(request.prefill_ids, request.stop);
  auto x = std::make_shared<Exchange>(std::move(job), std::move(parser), std::move(request));

  if (!x->req.stream) {
    try {
      pump(*x, [&req] { return req.is_connection_closed(); }, [](std::vector<text::Delta>&) {});
    } catch (const ClientDisconnected&) {
      scheduler_.withdraw(x->job);
      log_disconnected(*x);
      res.set_header("Connection", "close");
      send_error(res, 500, "client disconnected");
      return;
    }
    x->prefill_stats = x->job->wait_prefill_stats();
    log_done(*x, kv_cache_mode_);
    if (x->failure) {
      send_error(res, 500, *x->failure);
      return;
    }
    const text::Completion result = x->parser.result();
    Json message{{"role", "assistant"},
                 {"content", !result.content.empty() || result.tool_calls.empty() ? Json(result.content) : Json()},
                 {"reasoning_content", result.reasoning_content.empty() ? Json() : Json(result.reasoning_content)},
                 {"refusal", nullptr}};
    if (!result.tool_calls.empty()) {
      Json calls = Json::array();
      for (const text::ToolCall& tc : result.tool_calls) {
        calls.push_back(Json{{"id", tc.id},
                             {"type", "function"},
                             {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
      }
      message["tool_calls"] = std::move(calls);
    }
    Json resp{{"id", x->job->id},
              {"object", "chat.completion"},
              {"created", x->created},
              {"model", model_id_},
              {"kv_cache_mode", model::to_string(kv_cache_mode_)},
              {"loop_passes", loop_passes_},
              {"system_fingerprint", nullptr},
              {"choices", Json::array({Json{{"index", 0},
                                            {"message", std::move(message)},
                                            {"logprobs", nullptr},
                                            {"finish_reason", result.finish_reason ? Json(*result.finish_reason)
                                                                                   : Json()}}})},
              {"usage", usage(*x)}};
    if (x->req.sampling.persona) resp["persona_metadata"] = persona_meta_json(*x->req.sampling.persona);
    res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
    res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
    res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
    send_json(res, 200, resp);
    return;
  }

  auto state = std::make_shared<StreamState>(std::move(x));
  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
  res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
  res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, state](size_t, httplib::DataSink& sink) -> bool {
        if (state->started.exchange(true, std::memory_order_acq_rel)) {
          sink.done();
          return true;
        }
        Exchange& x = *state->x;
        SseTransport transport(sink, state->cancelled);
        try {
          transport.write(sse(dump(chunk(x, model_id_, Json{{"role", "assistant"}, {"content", ""}}, std::nullopt))));
          pump(x, [&] { return transport.poll(); },
               [&](std::vector<text::Delta>& deltas) {
                 for (const text::Delta& d : text::merge_deltas(deltas)) {
                   transport.write(sse(dump(chunk(x, model_id_, delta_json(d), std::nullopt))));
                 }
               });
        } catch (const ClientDisconnected&) {
          scheduler_.withdraw(x.job);
          log_disconnected(x);
          return false;
        }
        x.prefill_stats = x.job->wait_prefill_stats();
        log_done(x, kv_cache_mode_);
        try {
          if (x.failure) {
            transport.write(sse(dump(error_body(500, *x.failure))));
          } else {
            transport.write(sse(dump(chunk(x, model_id_, Json::object(), x.parser.finish_reason()))));
            if (x.req.include_usage) {
              transport.write(sse(dump(Json{{"id", x.job->id},
                                            {"object", "chat.completion.chunk"},
                                            {"created", x.created},
                                            {"model", model_id_},
                                            {"system_fingerprint", nullptr},
                                            {"choices", Json::array()},
                                            {"usage", usage(x)}})));
            }
          }
          transport.write(sse("[DONE]"));
        } catch (const ClientDisconnected&) {
          return false;
        }
        sink.done();
        return true;
      },
      [this, state](bool success) {
        // The client left before or while streaming: the job must not keep (or start) generating.
        if (!success || !state->started.load(std::memory_order_acquire)) scheduler_.withdraw(state->x->job);
      });
}

void ChatServer::complete_anthropic(const httplib::Request& req, httplib::Response& res, ChatRequest request) {
  std::string cache_namespace;
  bool bypass_prefix_cache = false;
  if (!cache_namespace_header_.empty()) {
    if (req.has_header(cache_namespace_header_)) {
      cache_namespace = req.get_header_value(cache_namespace_header_);
    }
    if (cache_namespace.empty()) bypass_prefix_cache = true;
  } else {
    const bool has_auth = (req.has_header("Authorization") && !req.get_header_value("Authorization").empty()) ||
                          (req.has_header("x-api-key") && !req.get_header_value("x-api-key").empty());
    if (request.has_isolation_hint || request.has_user_or_cache_key || has_auth) bypass_prefix_cache = true;
  }
  auto job = std::make_shared<engine::Job>(request.prompt_ids, request.max_new_tokens, request.sampling,
                                           std::move(request.images), request.root_prefix_tokens,
                                           std::move(cache_namespace), bypass_prefix_cache);
  if (!scheduler_.submit(job)) {
    res.set_header("Retry-After", "1");
    send_anthropic_error(res, 503, "server busy: request queue is full");
    return;
  }
  text::StreamParser parser = fmt_.parser(request.prefill_ids, request.stop);
  auto x = std::make_shared<Exchange>(std::move(job), std::move(parser), std::move(request));
  const std::string msg_id = "msg_" + x->job->id;

  if (!x->req.stream) {
    try {
      pump(*x, [&req] { return req.is_connection_closed(); }, [](std::vector<text::Delta>&) {});
    } catch (const ClientDisconnected&) {
      scheduler_.withdraw(x->job);
      log_disconnected(*x);
      res.set_header("Connection", "close");
      send_anthropic_error(res, 500, "client disconnected");
      return;
    }
    x->prefill_stats = x->job->wait_prefill_stats();
    log_done(*x, kv_cache_mode_);
    if (x->failure) {
      send_anthropic_error(res, 500, *x->failure);
      return;
    }
    const int prompt_tokens = int(x->job->prompt_ids.size());
    const int cached = x->prefill_stats ? std::clamp(x->prefill_stats->reused_tokens, 0, prompt_tokens) : 0;
    const Json resp = anthropic_message_response(msg_id, model_id_, x->parser.result(), prompt_tokens,
                                                 x->parser.num_tokens(), cached);
    res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
    res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
    res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
    send_json(res, 200, resp);
    return;
  }

  auto state = std::make_shared<StreamState>(std::move(x));
  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Ingine-Kv-Cache-Mode", model::to_string(kv_cache_mode_));
  res.set_header("X-Ingine-Loop-Passes", std::to_string(loop_passes_));
  res.set_header("X-Ingine-LoopCD", model::to_string(loopcd_mode_));
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, state, msg_id](size_t, httplib::DataSink& sink) -> bool {
        if (state->started.exchange(true, std::memory_order_acq_rel)) {
          sink.done();
          return true;
        }
        Exchange& x = *state->x;
        SseTransport transport(sink, state->cancelled);
        const int prompt_tokens = int(x.job->prompt_ids.size());

        enum class BlockKind { None, Thinking, Text, ToolUse };
        BlockKind cur_kind = BlockKind::None;
        int cur_tool_idx = -1;
        int cur_block = -1;
        int next_block = 0;

        auto close_block = [&] {
          if (cur_kind != BlockKind::None) {
            transport.write(anthropic_sse("content_block_stop",
                                          Json{{"type", "content_block_stop"}, {"index", cur_block}}));
            cur_kind = BlockKind::None;
            cur_tool_idx = -1;
            cur_block = -1;
          }
        };

        try {
          transport.write(anthropic_sse(
              "message_start",
              Json{{"type", "message_start"},
                   {"message",
                    {{"id", msg_id},
                     {"type", "message"},
                     {"role", "assistant"},
                     {"model", model_id_},
                     {"content", Json::array()},
                     {"stop_reason", nullptr},
                     {"stop_sequence", nullptr},
                     {"usage",
                      {{"input_tokens", prompt_tokens},
                       {"output_tokens", 0},
                       {"cache_creation_input_tokens", 0},
                       {"cache_read_input_tokens", 0}}}}}}));

          pump(x, [&] { return transport.poll(); },
               [&](std::vector<text::Delta>& deltas) {
                 for (const text::Delta& d : text::merge_deltas(deltas)) {
                   if (!d.reasoning_content.empty()) {
                     if (cur_kind != BlockKind::Thinking) {
                       close_block();
                       cur_kind = BlockKind::Thinking;
                       cur_block = next_block++;
                       transport.write(anthropic_sse(
                           "content_block_start",
                           Json{{"type", "content_block_start"},
                                {"index", cur_block},
                                {"content_block", {{"type", "thinking"}, {"thinking", ""}}}}));
                     }
                     transport.write(anthropic_sse(
                         "content_block_delta",
                         Json{{"type", "content_block_delta"},
                              {"index", cur_block},
                              {"delta", {{"type", "thinking_delta"}, {"thinking", d.reasoning_content}}}}));
                   } else if (!d.content.empty()) {
                     if (cur_kind != BlockKind::Text) {
                       close_block();
                       cur_kind = BlockKind::Text;
                       cur_block = next_block++;
                       transport.write(anthropic_sse(
                           "content_block_start",
                           Json{{"type", "content_block_start"},
                                {"index", cur_block},
                                {"content_block", {{"type", "text"}, {"text", ""}}}}));
                     }
                     transport.write(anthropic_sse(
                         "content_block_delta",
                         Json{{"type", "content_block_delta"},
                              {"index", cur_block},
                              {"delta", {{"type", "text_delta"}, {"text", d.content}}}}));
                   } else if (d.tool_call) {
                     const text::ToolCallDelta& tc = *d.tool_call;
                     if (cur_kind != BlockKind::ToolUse || cur_tool_idx != tc.index) {
                       close_block();
                       cur_kind = BlockKind::ToolUse;
                       cur_tool_idx = tc.index;
                       cur_block = next_block++;
                       transport.write(anthropic_sse(
                           "content_block_start",
                           Json{{"type", "content_block_start"},
                                {"index", cur_block},
                                {"content_block",
                                 {{"type", "tool_use"},
                                  {"id", tc.id ? *tc.id : ("call_" + std::to_string(tc.index))},
                                  {"name", tc.name ? *tc.name : ""},
                                  {"input", Json::object()}}}}));
                     }
                     if (!tc.arguments.empty()) {
                       transport.write(anthropic_sse(
                           "content_block_delta",
                           Json{{"type", "content_block_delta"},
                                {"index", cur_block},
                                {"delta", {{"type", "input_json_delta"}, {"partial_json", tc.arguments}}}}));
                     }
                   }
                 }
               });
        } catch (const ClientDisconnected&) {
          scheduler_.withdraw(x.job);
          log_disconnected(x);
          return false;
        }
        x.prefill_stats = x.job->wait_prefill_stats();
        log_done(x, kv_cache_mode_);
        try {
          close_block();
          if (x.failure) {
            transport.write(anthropic_sse("error", anthropic_error_body(500, *x.failure)));
          } else {
            const int cached = x.prefill_stats ? std::clamp(x.prefill_stats->reused_tokens, 0, prompt_tokens) : 0;
            const auto& matched_stop = x.parser.stop_sequence();
            const auto stop = anthropic_stop_reason(x.parser.finish_reason(), matched_stop);
            transport.write(anthropic_sse(
                "message_delta",
                Json{{"type", "message_delta"},
                     {"delta",
                      {{"stop_reason", stop ? Json(*stop) : Json()},
                       {"stop_sequence", matched_stop ? Json(*matched_stop) : Json()}}},
                     {"usage",
                      {{"input_tokens", prompt_tokens},
                       {"output_tokens", x.parser.num_tokens()},
                       {"cache_creation_input_tokens", 0},
                       {"cache_read_input_tokens", cached}}}}));
            transport.write(anthropic_sse("message_stop", Json{{"type", "message_stop"}}));
          }
        } catch (const ClientDisconnected&) {
          return false;
        }
        sink.done();
        return true;
      },
      [this, state](bool success) {
        if (!success || !state->started.load(std::memory_order_acquire)) scheduler_.withdraw(state->x->job);
      });
}

void ChatServer::complete_embedding(const httplib::Request& req, httplib::Response& res,
                                    ParsedEmbeddingRequest parsed) {
  const EmbeddingEncodingFormat encoding_format = parsed.encoding_format;
  auto job = std::make_shared<engine::EmbeddingJob>(std::move(parsed.request));
  if (!scheduler_.submit_embedding(job)) {
    res.set_header("Retry-After", "1");
    send_error(res, 503, "server busy: embedding request queue is full");
    return;
  }
  while (!job->wait(kPollInterval)) {
    if (req.is_connection_closed()) {
      scheduler_.withdraw_embedding(job);
      engine::server_log().info("{}: embedding client disconnected; cancelled", job->id);
      res.set_header("Connection", "close");
      send_error(res, 500, "client disconnected");
      return;
    }
  }
  if (job->failure()) {
    send_error(res, 500, *job->failure());
    return;
  }
  const model::EmbeddingResult& result = *job->result();
  const double total_s = seconds(Clock::now() - job->t_submit);
  const double queued_s = seconds(job->t_start().value_or(Clock::now()) - job->t_submit);
  engine::server_log().info(
      "{}: embeddings batch={} tokens={} dim={} tap={} platos_key={} prefill_ms={:.2f} pk_ms={:.3f} "
      "kv_slots={} lm_head_skipped={} queued={:.3f}s total={:.3f}s",
      job->id, result.embeddings.size(), result.stats.total_prompt_tokens, result.stats.output_dim,
      result.stats.effective_tap_layer >= 0 ? std::to_string(result.stats.effective_tap_layer) : std::string("full"),
      job->request.use_platos_key ? "true" : "false", result.stats.prefill_seconds * 1e3,
      result.stats.platos_key_seconds * 1e3, result.stats.kv_slots_allocated,
      result.stats.lm_head_skipped ? "true" : "false", queued_s, total_s);

  res.set_header("X-Ingine-Kv-Slots-Allocated", std::to_string(result.stats.kv_slots_allocated));
  res.set_header("X-Ingine-Lm-Head-Skipped", result.stats.lm_head_skipped ? "true" : "false");
  res.set_header("X-Ingine-Embedding-Dim", std::to_string(result.stats.output_dim));
  res.set_header("X-Ingine-Tap-Layer", result.stats.effective_tap_layer >= 0
                                           ? std::to_string(result.stats.effective_tap_layer)
                                           : "full");
  res.set_header("X-Ingine-Platos-Key", job->request.use_platos_key ? "true" : "false");
  if (job->request.use_platos_key) {
    res.set_header("X-Ingine-Platos-Key-Ms", fmt::format("{:.4f}", result.stats.platos_key_seconds * 1e3));
  }
  send_json(res, 200, embedding_response_json(model_id_, result, encoding_format));
}

bool ChatServer::bind(const std::string& host, int port) {
  if (port == 0) {
    port_ = server_.bind_to_any_port(host);
    return port_ > 0;
  }
  port_ = port;
  return server_.bind_to_port(host, port);
}

bool ChatServer::listen() { return server_.listen_after_bind(); }

void ChatServer::start() {
  thread_ = std::thread([this] { server_.listen_after_bind(); });
  server_.wait_until_ready();
}

void ChatServer::stop() {
  scheduler_.close();
  server_.stop();
  if (thread_.joinable()) thread_.join();
}

}  // namespace ingine::serve
