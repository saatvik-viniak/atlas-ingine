// OpenAI-compatible server end to end against FakeGenerator (CPU only, real tokenizer/template): port of
// engine/tests/test_server.py.
#include "engine/fake_generator.h"
#include "ingine/persona.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::engine::FakeGenerator;
using ingine::serve::ChatServer;
using ingine::text::ChatFormat;

int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                              \
      throw std::runtime_error("check failed");                                  \
    }                                                                            \
  } while (0)

const Json MSGS = Json::array({{{"role", "user"}, {"content", "Hi 🎉"}}});
const std::string THINK_TEXT =
    "<|message_model|><|content_thinking|>Let me think… 𓀀 ok<|end_message|>"
    "<|message_model|><|content_text|>Hello, wörld 𝔘! STOP here<|end_message|>"
    "<|content_model_end_sampling|>";
const std::string TOOL_TEXT =
    "<|message_model|><|content_thinking|>Need the weather.<|end_message|>"
    "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
    "{\"city\":\"São Paulo\",\"days\":3,\"opts\":{\"units\":\"metric\"}}}<|end_message|>"
    "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{\"q\":\"rain\"}}"
    "<|end_message|><|content_model_end_sampling|>";
std::string long_text() {
  std::string s = "<|message_model|><|content_text|>";
  for (int i = 0; i < 3000; ++i) s += "word ";
  return s + "<|end_message|><|content_model_end_sampling|>";
}
const Json TOOLS = Json::array(
    {{{"type", "function"}, {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}},
     {{"type", "function"}, {"function", {{"name", "search"}, {"parameters", {{"type", "object"}}}}}}});
const std::set<std::string> FINISH = {"stop", "length", "tool_calls"};

const ChatFormat* g_fmt = nullptr;

ingine::text::RenderedPrompt render(const Json& messages, Json effort = Json(), bool enable_thinking = true) {
  ingine::text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = enable_thinking;
  return g_fmt->render(messages, Json(), options);
}

struct Served {
  std::unique_ptr<FakeGenerator> gen;
  std::unique_ptr<ChatServer> srv;
  int port() const { return srv->port(); }
  ~Served() {
    if (srv) srv->stop();
  }
};

std::unique_ptr<Served> serve(std::string text = THINK_TEXT, double delay = 0.0, int max_queue = 4,
                              int max_model_len = 0, FakeGenerator::Responder responder = {},
                              ingine::model::Sampling sampling = {}, int loop_passes = 3) {
  auto s = std::make_unique<Served>();
  if (!responder) responder = [text](std::span<const int32_t>) { return FakeGenerator::Output(text); };
  s->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test", 1 << 20, responder, delay, loop_passes);
  ingine::serve::ServerOptions options;
  options.max_queue = max_queue;
  options.max_model_len = max_model_len;
  options.sampling = sampling;
  s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, options);
  if (!s->srv->bind("127.0.0.1", 0)) throw std::runtime_error("bind failed");
  s->srv->start();
  return s;
}

struct Response {
  int status = 0;
  httplib::Headers headers;
  std::string body;
};

Response request(int port, const std::string& method, const std::string& path, const std::string& body = "",
                 const httplib::Headers& headers = {}) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(20, 0);
  httplib::Result r = method == "GET" ? cli.Get(path, headers) : cli.Post(path, headers, body, "application/json");
  if (!r) throw std::runtime_error("request failed: " + httplib::to_string(r.error()));
  return {r->status, r->headers, r->body};
}

std::vector<Json> parse_sse(const std::string& data) {
  std::vector<Json> events;
  size_t pos = 0;
  while (pos < data.size()) {
    size_t end = data.find("\n\n", pos);
    if (end == std::string::npos) end = data.size();
    const std::string block = data.substr(pos, end - pos);
    pos = end + 2;
    if (block.find_first_not_of(" \t\r\n") == std::string::npos || block.starts_with(":")) continue;
    CHECK(block.starts_with("data: "));
    const std::string payload = block.substr(6);
    events.push_back(payload == "[DONE]" ? Json("[DONE]") : Json::parse(payload));
  }
  return events;
}

struct Chat {
  int status;
  httplib::Headers headers;
  Json body;  // response object, or the SSE events of a successful stream
};

Chat chat(int port, const Json& extra) {
  Json body{{"messages", MSGS}};
  for (auto& [k, v] : extra.items()) body[k] = v;
  Response r = request(port, "POST", "/v1/chat/completions", body.dump());
  const bool stream = extra.contains("stream") && extra["stream"] == true;
  return {r.status, r.headers, stream && r.status == 200 ? Json(parse_sse(r.body)) : Json::parse(r.body)};
}

std::string header(const httplib::Headers& h, const std::string& key) {
  auto it = h.find(key);
  return it == h.end() ? "" : it->second;
}

int raw_request(int port, const Json& extra) {
  Json body{{"messages", MSGS}};
  for (auto& [k, v] : extra.items()) body[k] = v;
  const std::string data = body.dump();
  const int s = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) throw std::runtime_error("connect failed");
  const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                          "Content-Length: " +
                          std::to_string(data.size()) + "\r\n\r\n" + data;
  CHECK(::send(s, req.data(), req.size(), MSG_NOSIGNAL) == ssize_t(req.size()));
  return s;
}

std::pair<std::string, std::string> read_raw_response(int s) {
  timeval tv{10, 0};
  ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  std::string buf;
  char tmp[65536];
  while (buf.find("\r\n\r\n") == std::string::npos) {
    const ssize_t n = ::recv(s, tmp, sizeof tmp, 0);
    if (n <= 0) throw std::runtime_error("connection closed");
    buf.append(tmp, size_t(n));
  }
  const size_t split = buf.find("\r\n\r\n");
  std::string head = buf.substr(0, split), body = buf.substr(split + 4);
  std::string lower = head;
  for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
  const size_t cl = lower.find("content-length:");
  if (cl == std::string::npos) throw std::runtime_error("no content-length");
  const size_t length = std::stoul(head.substr(cl + 15));
  while (body.size() < length) {
    const ssize_t n = ::recv(s, tmp, sizeof tmp, 0);
    if (n <= 0) throw std::runtime_error("connection closed");
    body.append(tmp, size_t(n));
  }
  return {head, body};
}

void wait_for(const std::function<bool()>& cond, double timeout = 10.0) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < timeout) {
    if (cond()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(!"condition not reached");
}

Json health(int port) { return Json::parse(request(port, "GET", "/health").body); }

// ---- OpenAI schema checks ---------------------------------------------------------------------------------

void check_usage(const Json& u) {
  CHECK(u.size() == 4 && u.contains("prompt_tokens") && u.contains("completion_tokens") && u.contains("total_tokens") &&
        u.contains("prompt_tokens_details"));
  for (const char* k : {"prompt_tokens", "completion_tokens", "total_tokens"}) {
    CHECK(u[k].is_number_integer() && u[k].get<int64_t>() >= 0);
  }
  CHECK(u["total_tokens"] == u["prompt_tokens"].get<int>() + u["completion_tokens"].get<int>());
  const Json& details = u["prompt_tokens_details"];
  CHECK(details.is_object() && details.contains("cached_tokens") && details["cached_tokens"].is_number_integer());
  const int cached = details["cached_tokens"].get<int>();
  CHECK(cached >= 0 && cached <= u["prompt_tokens"].get<int>());
}

void check_completion(const Json& r) {
  CHECK(r["id"].is_string() && r["id"].get<std::string>().starts_with("chatcmpl-"));
  CHECK(r["object"] == "chat.completion" && r["created"].is_number_integer() && r["model"] == "inkling-test");
  CHECK(r["choices"].size() == 1);
  const Json& c = r["choices"][0];
  CHECK(c["index"] == 0 && FINISH.contains(c["finish_reason"].get<std::string>()) && c["logprobs"].is_null());
  const Json& m = c["message"];
  CHECK(m["role"] == "assistant");
  CHECK(m["content"].is_null() || m["content"].is_string());
  CHECK(m["reasoning_content"].is_null() || m["reasoning_content"].is_string());
  if (m.contains("tool_calls")) {
    CHECK(!m["tool_calls"].empty());
    for (const Json& tc : m["tool_calls"]) {
      CHECK(tc.size() == 3 && tc["type"] == "function" && tc["id"].is_string());
      CHECK(tc["function"].size() == 2 && tc["function"]["name"].is_string() &&
            tc["function"]["arguments"].is_string());
    }
  }
  check_usage(r["usage"]);
}

struct CallOut {
  std::string id, name, arguments;
  bool operator==(const CallOut&) const = default;
};
struct Stream {
  std::string reasoning, content;
  std::vector<CallOut> calls;
  std::string finish;
  Json usage;
};

// Validates a chunk stream and reassembles it.
Stream check_chunks(Json events, bool include_usage = false) {
  CHECK(events.back() == "[DONE]");
  Json chunks = Json::array();
  for (size_t i = 0; i + 1 < events.size(); ++i) chunks.push_back(events[i]);
  std::set<std::string> ids;
  for (const Json& c : chunks) ids.insert(c["id"].get<std::string>());
  CHECK(ids.size() == 1 && ids.begin()->starts_with("chatcmpl-"));
  Stream out;
  if (include_usage) {
    Json last = chunks.back();
    chunks.erase(chunks.size() - 1);
    CHECK(last["choices"] == Json::array() && last["object"] == "chat.completion.chunk");
    check_usage(last["usage"]);
    out.usage = last["usage"];
  }
  CHECK((chunks[0]["choices"][0]["delta"] == Json{{"role", "assistant"}, {"content", ""}}));
  std::map<int, CallOut> calls;
  bool finished = false;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Json& c = chunks[i];
    CHECK(c["object"] == "chat.completion.chunk" && c["created"].is_number_integer());
    CHECK(c["model"] == "inkling-test");
    CHECK(include_usage ? (c.contains("usage") && c["usage"].is_null()) : !c.contains("usage"));
    CHECK(c["choices"].size() == 1);
    const Json& ch = c["choices"][0];
    CHECK(ch["index"] == 0 && ch["logprobs"].is_null());
    const Json& d = ch["delta"];
    for (auto& [k, v] : d.items()) {
      CHECK(k == "role" || k == "content" || k == "reasoning_content" || k == "tool_calls");
    }
    if (!ch["finish_reason"].is_null()) {
      CHECK(i == chunks.size() - 1 && FINISH.contains(ch["finish_reason"].get<std::string>()) && d.empty());
      out.finish = ch["finish_reason"];
      finished = true;
      continue;
    }
    if (i > 0) CHECK(d.size() == 1 && !d.contains("role"));
    if (d.contains("reasoning_content")) out.reasoning += d["reasoning_content"].get<std::string>();
    if (d.contains("content") && d["content"].is_string()) out.content += d["content"].get<std::string>();
    if (d.contains("tool_calls")) {
      for (const Json& tc : d["tool_calls"]) {
        CHECK(tc["index"].is_number_integer());
        const int index = tc["index"];
        if (tc.contains("id")) {
          CHECK(!calls.contains(index) && tc["type"] == "function");
          CHECK(tc["function"]["name"].is_string() && !tc["function"]["name"].get<std::string>().empty());
          calls[index] = CallOut{tc["id"], tc["function"]["name"], ""};
        } else {
          CHECK(tc.size() == 2 && tc.contains("function") && tc["function"].size() == 1 &&
                tc["function"].contains("arguments"));
        }
        calls[index].arguments += tc["function"]["arguments"].get<std::string>();
      }
    }
  }
  CHECK(finished);
  CHECK((out.reasoning + out.content).find("\xEF\xBF\xBD") == std::string::npos);
  for (auto& [k, v] : calls) out.calls.push_back(v);
  return out;
}

int encoded_len(const std::string& text) { return int(g_fmt->tokenizer().encode(text, true).size()); }

// ---- tests ---------------------------------------------------------------------------------------------------

void test_models_and_health() {
  auto s = serve();
  Response r = request(s->port(), "GET", "/v1/models");
  Json body = Json::parse(r.body);
  CHECK(r.status == 200 && body["object"] == "list");
  CHECK(body["data"].size() == 1);
  const Json& m = body["data"][0];
  CHECK(m["id"] == "inkling-test" && m["object"] == "model" && m["created"].is_number_integer());
  CHECK(m["owned_by"].is_string() && m["max_model_len"] == (1 << 20) && m["loop_passes"] == 3);
  CHECK(m["kv_cache_mode"] == "per_pass");
  CHECK(m["supported_kv_cache_modes"] == Json::array({"per_pass", "last_full"}));
  CHECK(m["kv_cache_mode_mutable"] == false);
  CHECK(header(r.headers, "X-Ingine-Kv-Cache-Mode") == "per_pass");
  CHECK(Json::parse(request(s->port(), "GET", "/v1/models/inkling-test").body) == m);
  Response hr = request(s->port(), "GET", "/health");
  CHECK(hr.status == 200 && header(hr.headers, "X-Ingine-Kv-Cache-Mode") == "per_pass");
  Json h = Json::parse(hr.body);
  CHECK(h["status"] == "ok" && h["model"] == "inkling-test" && h["loop_passes"] == 3 &&
        h["kv_cache_mode"] == "per_pass" &&
        h["supported_kv_cache_modes"] == Json::array({"per_pass", "last_full"}) &&
        h["kv_cache_mode_mutable"] == false && h["warm_start"] == false && h["warm_start_tokens"] == 0 &&
        h["waiting"] == 0 && h["running"] == false && h["max_queue"] == 4);
  CHECK(request(s->port(), "GET", "/health/").status == 200);
  Response nope = request(s->port(), "GET", "/nope");
  CHECK(nope.status == 404 && Json::parse(nope.body)["error"]["type"] == "not_found_error");
  CHECK(Json::parse(nope.body)["error"]["message"] == "no route for GET /nope");
  CHECK(request(s->port(), "POST", "/v1/completions", "{}").status == 404);

  // Non-default loop_passes metadata propagation (/health, /v1/models, /v1/models/<id>) and last_full restriction
  auto s5 = serve(THINK_TEXT, 0.0, 4, 0, {}, {}, 5);
  Json m5 = Json::parse(request(s5->port(), "GET", "/v1/models").body)["data"][0];
  CHECK(m5["loop_passes"] == 5);
  CHECK(m5["kv_cache_mode"] == "per_pass");
  CHECK(m5["supported_kv_cache_modes"] == Json::array({"per_pass"}));
  CHECK(Json::parse(request(s5->port(), "GET", "/v1/models/inkling-test").body)["loop_passes"] == 5);
  Json h5 = health(s5->port());
  CHECK(h5["loop_passes"] == 5);
  CHECK(h5["kv_cache_mode"] == "per_pass");
  CHECK(h5["supported_kv_cache_modes"] == Json::array({"per_pass"}));

  // FakeGenerator rejects incompatible loop_passes != 3 with last_full, matching real serving.
  bool fake_rejected_l5_lf = false;
  try {
    FakeGenerator bad(g_fmt->tokenizer(), "inkling-test", 1 << 20, {}, 0.0, 5, ingine::model::KvCacheMode::LastFull);
    (void)bad;
  } catch (const std::invalid_argument&) {
    fake_rejected_l5_lf = true;
  }
  CHECK(fake_rejected_l5_lf);

  // L=3 FakeGenerator with explicit last_full serves last_full consistently.
  FakeGenerator fg_lf(g_fmt->tokenizer(), "inkling-test", 1 << 20, {}, 0.0, 3, ingine::model::KvCacheMode::LastFull);
  CHECK(fg_lf.kv_cache_mode() == ingine::model::KvCacheMode::LastFull);
  CHECK(fg_lf.supported_kv_cache_modes().size() == 2);

  // X-Ingine-Loop-Passes response header + completion body field + X-Ingine-Expected-Loop-Passes precondition.
  CHECK(header(r.headers, "X-Ingine-Loop-Passes") == "3");
  CHECK(header(hr.headers, "X-Ingine-Loop-Passes") == "3");
  Response c5_mismatch = request(
      s5->port(), "POST", "/v1/chat/completions",
      Json{{"model", "inkling-test"}, {"messages", MSGS}}.dump(),
      {{"Content-Type", "application/json"}, {"X-Ingine-Expected-Loop-Passes", "1"}});
  CHECK(c5_mismatch.status == 409);
  CHECK(header(c5_mismatch.headers, "X-Ingine-Loop-Passes") == "5");
  CHECK(s5->gen->calls().empty());  // rejected BEFORE generation!
  Response c5_bad = request(
      s5->port(), "POST", "/v1/chat/completions",
      Json{{"model", "inkling-test"}, {"messages", MSGS}}.dump(),
      {{"Content-Type", "application/json"}, {"X-Ingine-Expected-Loop-Passes", "99"}});
  CHECK(c5_bad.status == 400);
  CHECK(s5->gen->calls().empty());
  Response c5_ok = request(
      s5->port(), "POST", "/v1/chat/completions",
      Json{{"model", "inkling-test"}, {"messages", MSGS}}.dump(),
      {{"Content-Type", "application/json"}, {"X-Ingine-Expected-Loop-Passes", "5"}});
  CHECK(c5_ok.status == 200);
  CHECK(header(c5_ok.headers, "X-Ingine-Loop-Passes") == "5");
  CHECK(Json::parse(c5_ok.body)["loop_passes"] == 5);
  CHECK(s5->gen->calls().size() == 1);
}

Json stats(int port) { return Json::parse(request(port, "GET", "/stats").body); }

// GET /stats: the live counters behind the dashboard (idle shape, an in-flight request, cleanup on disconnect).
void test_stats() {
  auto s = serve(long_text(), 0.002);
  const int port = s->port();
  Json idle = stats(port);
  CHECK(idle["now"].is_number());
  CHECK(idle["requests"].is_array() && idle["requests"].empty());
  CHECK(idle["totals"]["requests"] == 0 && idle["totals"]["prompt_tokens"] == 0 &&
        idle["totals"]["output_tokens"] == 0);
  CHECK(health(port)["max_context"] == (1 << 20));

  // one streaming answer in flight: generated advances, elapsed is non-negative, the prompt/max are known
  const int sock = raw_request(port, {{"stream", true}, {"max_tokens", 2000}});
  wait_for([&] {
    Json j = stats(port);
    return j["requests"].size() == 1 && j["requests"][0]["generated"].get<int>() > 0;
  });
  Json live = stats(port)["requests"][0];
  CHECK(live["id"].is_string() && live["id"].get<std::string>().starts_with("chatcmpl-"));
  CHECK(live["prompt_tokens"].get<int>() > 0 && live["max_tokens"] == 2000);
  CHECK(live["first_token"] == true);
  CHECK(live["generated"].get<int>() > 0 && live["generated"].get<int>() <= 2000);
  CHECK(live["started"].is_number() && live["elapsed_s"].is_number() && live["elapsed_s"].get<double>() >= 0.0);
  const int seen = live["generated"].get<int>();
  wait_for([&] { return stats(port)["requests"][0]["generated"].get<int>() > seen; });

  // the client leaves: the live entry is removed and the totals keep the request
  ::close(sock);
  wait_for([&] { return stats(port)["requests"].empty(); });
  Json after = stats(port);
  CHECK(after["totals"]["requests"] == 1);
  CHECK(after["totals"]["prompt_tokens"].get<int>() > 0);
  CHECK(after["totals"]["output_tokens"].get<int>() > 0);

  // a second disconnect cleans up the same way
  const int sock2 = raw_request(port, {{"stream", true}, {"max_tokens", 2000}});
  wait_for([&] { return stats(port)["requests"].size() == 1; });
  ::close(sock2);
  wait_for([&] { return stats(port)["requests"].empty(); });
  CHECK(stats(port)["totals"]["requests"] >= 2);
}

void test_non_streaming() {
  auto s = serve();
  Chat r = chat(s->port(), {{"model", "whatever"}});
  CHECK(r.status == 200);
  check_completion(r.body);
  const Json& m = r.body["choices"][0]["message"];
  CHECK(m["reasoning_content"] == "Let me think… 𓀀 ok");
  CHECK(m["content"] == "Hello, wörld 𝔘! STOP here" && !m.contains("tool_calls"));
  CHECK(m["refusal"].is_null() && r.body["system_fingerprint"].is_null());
  CHECK(r.body["choices"][0]["finish_reason"] == "stop");
  const auto calls = s->gen->calls();
  const auto ids = render(MSGS).ids;
  CHECK(r.body["usage"]["prompt_tokens"] == ids.size() && calls[0].prompt_ids.size() == ids.size());
  CHECK(r.body["usage"]["completion_tokens"] == encoded_len(THINK_TEXT));
  CHECK(calls[0].prompt_ids == ids);
}

void test_streaming_matches_non_streaming(bool include_usage) {
  auto s = serve();
  Chat ref = chat(s->port(), Json::object());
  Chat r = chat(s->port(), {{"stream", true}, {"stream_options", {{"include_usage", include_usage}}}});
  CHECK(r.status == 200 && header(r.headers, "Content-Type").starts_with("text/event-stream"));
  Stream st = check_chunks(r.body, include_usage);
  const Json& m = ref.body["choices"][0]["message"];
  CHECK(Json(st.reasoning) == m["reasoning_content"] && Json(st.content) == m["content"] && st.calls.empty() &&
        st.finish == "stop");
  if (include_usage) CHECK(st.usage == ref.body["usage"]);
}

void test_tool_call_streaming() {
  auto s = serve(TOOL_TEXT, 0.003);  // paced so deltas are not coalesced into one chunk
  Chat r = chat(s->port(), {{"tools", TOOLS}});
  check_completion(r.body);
  const Json& m = r.body["choices"][0]["message"];
  CHECK(r.body["choices"][0]["finish_reason"] == "tool_calls" && m["content"].is_null());
  CHECK(m["tool_calls"].size() == 2);
  CHECK(m["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK((Json::parse(m["tool_calls"][0]["function"]["arguments"].get<std::string>()) ==
         Json{{"city", "São Paulo"}, {"days", 3}, {"opts", {{"units", "metric"}}}}));
  CHECK(m["tool_calls"][1]["function"]["name"] == "search");
  CHECK((Json::parse(m["tool_calls"][1]["function"]["arguments"].get<std::string>()) == Json{{"q", "rain"}}));
  Chat sr = chat(s->port(), {{"tools", TOOLS}, {"stream", true}, {"stream_options", {{"include_usage", true}}}});
  Stream st = check_chunks(sr.body, true);
  CHECK(st.finish == "tool_calls" && st.reasoning == "Need the weather." && st.content.empty());
  CHECK(st.calls.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    CHECK(Json(st.calls[i].name) == m["tool_calls"][i]["function"]["name"]);
    CHECK(Json(st.calls[i].arguments) == m["tool_calls"][i]["function"]["arguments"]);
  }
  CHECK(st.calls[0].id != st.calls[1].id && st.usage == r.body["usage"]);
  int arg_chunks = 0;
  for (size_t i = 0; i + 2 < sr.body.size(); ++i) {
    const Json& d = sr.body[i]["choices"][0]["delta"];
    if (!d.contains("tool_calls")) continue;
    for (const Json& tc : d["tool_calls"]) arg_chunks += !tc.contains("id") && tc["index"] == 0;
  }
  CHECK(arg_chunks > 1);  // arguments really stream
}

void test_tool_round_trip_prompt() {
  Json msgs = MSGS;
  msgs.push_back({{"role", "assistant"},
                  {"content", nullptr},
                  {"reasoning_content", "r"},
                  {"tool_calls",
                   Json::array({{{"id", "call_a"},
                                 {"type", "function"},
                                 {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"Paris\"}"}}}}})}});
  msgs.push_back({{"role", "tool"}, {"tool_call_id", "call_a"}, {"content", "sunny"}});
  auto s = serve();
  Chat r = chat(s->port(), {{"messages", msgs}, {"tools", TOOLS}});
  CHECK(r.status == 200);
  const std::string text = g_fmt->tokenizer().decode(s->gen->calls()[0].prompt_ids);
  CHECK(text.find("<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":"
                  "{\"city\":\"Paris\"}}<|end_message|><|content_model_end_sampling|>"
                  "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>") != std::string::npos);
}

void test_thinking_controls() {
  auto s = serve("Plain answer.<|end_message|><|content_model_end_sampling|>");
  for (const Json& body : {Json{{"chat_template_kwargs", {{"enable_thinking", false}}}}, Json{{"enable_thinking", false}}}) {
    Json req = body;
    req["reasoning_effort"] = 0.3;
    Chat r = chat(s->port(), req);
    CHECK(r.status == 200);
    const Json& m = r.body["choices"][0]["message"];
    CHECK(m["content"] == "Plain answer." && m["reasoning_content"].is_null());
    const auto want = render(MSGS, 0.3, false);
    CHECK(s->gen->calls().back().prompt_ids == want.ids);
    CHECK(r.body["usage"]["prompt_tokens"] == want.ids.size());
  }
  chat(s->port(), {{"reasoning_effort", "max"}});
  CHECK(s->gen->calls().back().prompt_ids == render(MSGS, "max").ids);
  chat(s->port(), {{"chat_template_kwargs", {{"reasoning_effort", "low"}}}});
  CHECK(s->gen->calls().back().prompt_ids == render(MSGS, "low").ids);
}

void test_sampling_and_max_tokens() {
  auto s = serve();
  Chat r = chat(s->port(), {{"temperature", 0.6}, {"top_p", 0.95}, {"top_k", 20}, {"seed", 7}, {"max_tokens", 5}});
  CHECK(r.status == 200 && r.body["choices"][0]["finish_reason"] == "length");
  CHECK(r.body["usage"]["completion_tokens"] == 5);
  auto call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.6f && call.sampling.top_p == 0.95f && call.sampling.top_k == 20 &&
        call.sampling.seed == uint64_t(7));
  CHECK(call.max_new_tokens == 5);
  chat(s->port(), {{"max_completion_tokens", 3}, {"max_tokens", 9}, {"top_k", -1}});
  call = s->gen->calls().back();
  CHECK(call.max_new_tokens == 3 && call.sampling.top_k == 0 && !call.sampling.seed);
  CHECK(call.sampling.temperature == 1.0f && call.sampling.top_p == 1.0f);
  Chat sr = chat(s->port(), {{"stream", true}, {"max_tokens", 4}});
  CHECK(check_chunks(sr.body).finish == "length");
}

// ServerOptions::sampling (ingine-serve --temperature / --top-p / --top-k) fills the fields a request omits;
// fields the request sets win.
void test_sampling_defaults() {
  ingine::model::Sampling defaults;
  defaults.temperature = 0.7f;
  defaults.top_p = 0.95f;
  defaults.top_k = 40;
  auto s = serve(THINK_TEXT, 0.0, 4, 0, {}, defaults);
  chat(s->port(), {{"max_tokens", 2}});
  auto call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.7f && call.sampling.top_p == 0.95f && call.sampling.top_k == 40);
  chat(s->port(), {{"max_tokens", 2}, {"temperature", 1.0}, {"top_p", 1.0}, {"top_k", 0}});
  call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 1.0f && call.sampling.top_p == 1.0f && call.sampling.top_k == 0);
  chat(s->port(), {{"max_tokens", 2}, {"top_p", 0.5}});
  call = s->gen->calls().back();
  CHECK(call.sampling.temperature == 0.7f && call.sampling.top_p == 0.5f && call.sampling.top_k == 40);
}

void test_context_limit() {
  const int n = int(render(MSGS).ids.size());
  {
    auto s = serve(THINK_TEXT, 0.0, 4, n + 6);
    Chat r = chat(s->port(), Json::object());
    CHECK(r.body["choices"][0]["finish_reason"] == "length" && s->gen->calls().back().max_new_tokens == 6);
    chat(s->port(), {{"max_tokens", 100}});
    CHECK(s->gen->calls().back().max_new_tokens == 6);
  }
  auto s = serve(THINK_TEXT, 0.0, 4, n);
  Chat r = chat(s->port(), Json::object());
  CHECK(r.status == 400 && r.body["error"]["param"] == "messages");
}

void test_stop_strings_cancel_generation(bool stream) {
  std::string text = "<|message_model|><|content_text|>alpha beta<END> gamma";
  for (int i = 0; i < 200; ++i) text += " delta";
  text += "<|end_message|>";
  auto s = serve(text, 0.002);
  Chat r = chat(s->port(), {{"stop", Json::array({"<END>", "never"})},
                            {"stream", stream},
                            {"stream_options", {{"include_usage", true}}}});
  CHECK(r.status == 200);
  std::string content, finish;
  Json usage;
  if (stream) {
    Stream st = check_chunks(r.body, true);
    content = st.content, finish = st.finish, usage = st.usage;
  } else {
    check_completion(r.body);
    content = r.body["choices"][0]["message"]["content"];
    finish = r.body["choices"][0]["finish_reason"];
    usage = r.body["usage"];
  }
  CHECK(content == "alpha beta" && finish == "stop");
  CHECK(usage["completion_tokens"].get<int>() < 20);
  wait_for([&] { return s->gen->calls().back().done; });
  CHECK(s->gen->calls().back().cancelled && s->gen->calls().back().yielded < 50);
  Chat r2 = chat(s->port(), {{"stop", "beta"}});
  CHECK(r2.body["choices"][0]["message"]["content"] == "alpha ");
}

void test_disconnect_cancels_generation(bool stream) {
  auto s = serve(long_text(), 0.002);
  const int sock = raw_request(s->port(), {{"stream", stream}});
  wait_for([&] {
    auto calls = s->gen->calls();
    return !calls.empty() && calls.back().yielded > 20;
  });
  if (stream) {
    char buf[4096];
    const ssize_t n = ::recv(sock, buf, sizeof buf, 0);
    CHECK(n > 0 && std::string(buf, size_t(n)).starts_with("HTTP/1.1 200"));
  }
  ::close(sock);
  wait_for([&] { return s->gen->calls().back().done; });
  CHECK(s->gen->calls().back().cancelled && s->gen->calls().back().yielded < 2000);
  wait_for([&] { return !health(s->port())["running"].get<bool>(); });
  Chat r = chat(s->port(), {{"max_tokens", 3}});  // server still healthy
  CHECK(r.status == 200 && r.body["usage"]["completion_tokens"] == 3);
}

void test_queue_full_503_and_withdraw() {
  auto s = serve(long_text(), 0.005, 1);
  const int port = s->port();
  const int a = raw_request(port, {{"stream", true}});
  wait_for([&] { return health(port)["running"].get<bool>(); });
  const int b = raw_request(port, Json::object());
  wait_for([&] { return health(port)["waiting"] == 1; });
  Chat r = chat(port, Json::object());
  CHECK(r.status == 503 && r.body["error"]["code"] == 503 && r.body["error"]["message"].is_string());
  CHECK(r.body["error"]["type"] == "server_busy");
  CHECK(header(r.headers, "Retry-After") == "1");
  ::close(b);  // a queued request whose client leaves is withdrawn without ever running
  wait_for([&] { return health(port)["waiting"] == 0; });
  const int c = raw_request(port, {{"max_tokens", 2}});
  wait_for([&] { return health(port)["waiting"] == 1; });
  ::close(a);
  auto [head, body] = read_raw_response(c);
  ::close(c);
  CHECK(head.starts_with("HTTP/1.1 200"));
  CHECK(Json::parse(body)["usage"]["completion_tokens"] == 2);
  const auto calls = s->gen->calls();
  CHECK(calls.size() == 2);  // a and c ran; b never reached the generator
  CHECK(calls[0].cancelled && calls[0].yielded < 2000);
}

void test_fifo_order() {
  auto s = serve(THINK_TEXT, 0.001, 8);
  std::mutex mutex;
  std::vector<std::pair<int, int>> order;
  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&, i] {
      Chat r = chat(s->port(), {{"messages", Json::array({{{"role", "user"}, {"content", "req " + std::to_string(i)}}})},
                                {"max_tokens", 4}});
      std::lock_guard lock(mutex);
      order.emplace_back(i, r.status);
    });
    wait_for([&] { return int(s->gen->calls().size()) + health(s->port())["waiting"].get<int>() >= i + 1; });
  }
  for (auto& t : threads) t.join();
  std::sort(order.begin(), order.end());
  for (int i = 0; i < 5; ++i) CHECK(order[size_t(i)] == std::make_pair(i, 200));
  const auto calls = s->gen->calls();
  CHECK(calls.size() == 5);
  for (int i = 0; i < 5; ++i) {
    CHECK(g_fmt->tokenizer().decode(calls[size_t(i)].prompt_ids).find("req " + std::to_string(i)) != std::string::npos);
  }
}

void test_generator_failure() {
  auto s = serve(THINK_TEXT, 0.0, 4, 0,
                 [](std::span<const int32_t>) -> FakeGenerator::Output { throw std::runtime_error("kaboom"); });
  Chat r = chat(s->port(), Json::object());
  CHECK(r.status == 500 && r.body["error"]["message"].get<std::string>().find("kaboom") != std::string::npos);
  CHECK(r.body["error"]["type"] == "server_error" && r.body["error"]["code"] == 500);
  Chat sr = chat(s->port(), {{"stream", true}});
  CHECK(sr.status == 200 && sr.body.back() == "[DONE]");
  CHECK(sr.body[sr.body.size() - 2]["error"]["message"].get<std::string>().find("kaboom") != std::string::npos);
  wait_for([&] { return !health(s->port())["running"].get<bool>(); });
}

void test_bad_requests() {
  const std::vector<std::pair<Json, std::string>> cases = {
      {{{"n", 2}}, "n"},
      {{{"temperature", -1}}, "temperature"},
      {{{"temperature", "hot"}}, "temperature"},
      {{{"top_p", 0}}, "top_p"},
      {{{"max_tokens", 0}}, "max_tokens"},
      {{{"max_tokens", 1.5}}, "max_tokens"},
      {{{"seed", true}}, "seed"},
      {{{"stream", "yes"}}, "stream"},
      {{{"stop", 5}}, "stop"},
      {{{"messages", Json::array()}}, "messages"},
      {{{"messages", Json::array({{{"role", "user"},
                                   {"content", Json::array({{{"type", "image_url"}, {"image_url", {{"url", "x"}}}}})}}})}},
       "messages"},
      {{{"reasoning_effort", 2}}, "reasoning_effort"},
      {{{"chat_template_kwargs", {{"enable_thinking", "no"}}}}, "enable_thinking"},
  };
  auto s = serve();
  for (const auto& [body, param] : cases) {
    Chat r = chat(s->port(), body);
    if (!(r.status == 400 && r.body["error"]["param"] == param && r.body["error"]["type"] == "invalid_request_error")) {
      std::fprintf(stderr, "bad request %s -> %d %s\n", body.dump().c_str(), r.status, r.body.dump().c_str());
    }
    CHECK(r.status == 400 && r.body["error"]["param"] == param && r.body["error"]["type"] == "invalid_request_error");
    CHECK(r.body["error"]["code"] == 400);
  }
  Chat r = chat(s->port(), {{"temperature", -1}});
  CHECK(r.body["error"]["message"] == "'temperature' must be in [0.0, 2.0]");
  r = chat(s->port(), {{"max_tokens", 0}});
  CHECK(r.body["error"]["message"] == "'max_tokens' must be in [1, None]");
  CHECK(s->gen->calls().empty());
}

void test_invalid_json() {
  auto s = serve();
  Response r = request(s->port(), "POST", "/v1/chat/completions", "{nope");
  CHECK(r.status == 400 && Json::parse(r.body)["error"]["message"].get<std::string>().find("invalid JSON") !=
                               std::string::npos);
  r = request(s->port(), "POST", "/v1/chat/completions", "[1]");
  CHECK(r.status == 400 && Json::parse(r.body)["error"]["message"] == "request body must be a JSON object");
}

// The default responder: echo, tool call, thinking disabled.
void test_echo() {
  auto gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test");
  ChatServer srv(*gen, *g_fmt);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();
  Chat r = chat(srv.port(), Json::object());
  const Json& m = r.body["choices"][0]["message"];
  CHECK(m["content"] == "You said: Hi 🎉" && m["reasoning_content"] == "The user wrote 4 characters.");
  r = chat(srv.port(), {{"enable_thinking", false}});
  CHECK(r.body["choices"][0]["message"]["content"] == "You said: Hi 🎉");
  CHECK(r.body["choices"][0]["message"]["reasoning_content"].is_null());
  r = chat(srv.port(), {{"messages", Json::array({{{"role", "user"}, {"content", "call get_weather {\"city\": \"Paris\"}"}}})},
                        {"tools", TOOLS}});
  const Json& t = r.body["choices"][0]["message"];
  CHECK(r.body["choices"][0]["finish_reason"] == "tool_calls" && t["reasoning_content"] == "Calling get_weather.");
  CHECK(t["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK((Json::parse(t["tool_calls"][0]["function"]["arguments"].get<std::string>()) == Json{{"city", "Paris"}}));
  srv.stop();
}

}  // namespace

// ---- images

std::string data_uri(const std::string& fixture) {
  const std::filesystem::path path = std::filesystem::path(INGINE_GOLDEN_DIR) / "vision" / "fixtures" / fixture;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  const std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  return "data:image/png;base64," + httplib::detail::base64_encode(bytes);
}

Json image_part(const std::string& uri) { return {{"type", "image_url"}, {"image_url", {{"url", uri}}}}; }

Json user(Json content) { return Json::array({{{"role", "user"}, {"content", std::move(content)}}}); }

std::vector<int32_t> image_ids(const std::vector<int32_t>& prompt) {
  std::vector<int32_t> out;
  for (const int32_t id : prompt) {
    if (id < 0) out.push_back(id);
  }
  return out;
}

void test_images() {
  const std::string red = data_uri("solid_red_40x40.png"), blue = data_uri("solid_blue_40x40.png");
  const std::string grad = data_uri("grad_100x60.png");  // 2 x 3 tiles: 6 tokens
  auto s = serve();
  CHECK(Json::parse(request(s->port(), "GET", "/v1/models").body)["data"][0]["input_modalities"] ==
        Json::array({"text", "image"}));

  // one image: its placeholder becomes 6 negative ids; the usage counts them
  const Json one = user(Json::array({{{"type", "text"}, {"text", "What is this?"}}, image_part(grad)}));
  Chat r = chat(s->port(), {{"messages", one}});
  CHECK(r.status == 200);
  const std::vector<int32_t> rendered = render(one).ids;  // with the template's single placeholder
  CHECK(std::count(rendered.begin(), rendered.end(), 200054) == 1);
  FakeGenerator::Call call = s->gen->calls().back();
  CHECK(call.images == 1 && call.image_tokens == 6);
  CHECK(call.prompt_ids.size() == rendered.size() - 1 + 6);
  CHECK(std::count(call.prompt_ids.begin(), call.prompt_ids.end(), 200054) == 0);
  const std::vector<int32_t> grad_ids = image_ids(call.prompt_ids);
  CHECK(grad_ids.size() == 6 && std::set<int32_t>(grad_ids.begin(), grad_ids.end()).size() == 6);
  CHECK(r.body["usage"]["prompt_tokens"] == int(call.prompt_ids.size()));
  // the ids take the placeholder's place, between <|content_image|> (200005) and <|end_message|> (200010)
  const auto first = std::find(call.prompt_ids.begin(), call.prompt_ids.end(), grad_ids[0]);
  CHECK(first != call.prompt_ids.begin() && *(first - 1) == 200005 && *(first + 6) == 200010);
  CHECK(std::equal(grad_ids.begin(), grad_ids.end(), first));

  // the same image again gives the same prompt (the prompt caches can reuse it), another image another one
  r = chat(s->port(), {{"messages", one}});
  CHECK(r.status == 200 && s->gen->calls().back().prompt_ids == call.prompt_ids);
  r = chat(s->port(), {{"messages", user(Json::array({{{"type", "text"}, {"text", "What is this?"}}, image_part(red)}))}});
  CHECK(r.status == 200);
  const std::vector<int32_t> red_ids = image_ids(s->gen->calls().back().prompt_ids);
  CHECK(red_ids.size() == 1 && red_ids[0] != grad_ids[0]);
  r = chat(s->port(), {{"messages", user(Json::array({image_part(blue)}))}});
  CHECK(r.status == 200 && image_ids(s->gen->calls().back().prompt_ids) != red_ids);

  // several images keep their order; other part shapes carry the same image
  r = chat(s->port(), {{"messages", user(Json::array({image_part(red), {{"type", "text"}, {"text", "and"}}, image_part(grad)}))}});
  CHECK(r.status == 200 && s->gen->calls().back().images == 2 && s->gen->calls().back().image_tokens == 7);
  std::vector<int32_t> both = red_ids;
  both.insert(both.end(), grad_ids.begin(), grad_ids.end());
  CHECK(image_ids(s->gen->calls().back().prompt_ids) == both);
  const std::string base64 = grad.substr(grad.find(',') + 1);
  const Json shapes = Json::array({
      {{"type", "image_url"}, {"image_url", grad}},
      {{"type", "input_image"}, {"image_url", grad}},
      {{"type", "image"}, {"image", grad}},
      {{"type", "image"}, {"url", grad}},
      {{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", base64}}}},
  });
  for (const Json& shape : shapes) {
    r = chat(s->port(), {{"messages", user(Json::array({shape}))}});
    CHECK(r.status == 200 && image_ids(s->gen->calls().back().prompt_ids) == grad_ids);
  }
  // an image in a tool message, and streaming with usage
  r = chat(s->port(), {{"messages", Json::array({{{"role", "user"}, {"content", "look"}},
                                                 {{"role", "tool"}, {"content", Json::array({image_part(red)})}}})}});
  CHECK(r.status == 200 && image_ids(s->gen->calls().back().prompt_ids) == red_ids);
  r = chat(s->port(), {{"messages", one}, {"stream", true}, {"stream_options", {{"include_usage", true}}}});
  CHECK(r.status == 200);
  bool usage_seen = false;
  for (const Json& event : r.body) {
    if (event.is_object() && event.contains("usage") && event["usage"].is_object()) {
      usage_seen = event["usage"]["prompt_tokens"] == int(call.prompt_ids.size());
    }
  }
  CHECK(usage_seen);

  // rejected with 400, param messages, before anything reaches the generator
  const std::size_t calls_before = s->gen->calls().size();
  const std::vector<std::pair<Json, const char*>> bad = {
      {user(Json::array({image_part("https://example.com/cat.png")})), "data: URIs"},
      {user(Json::array({image_part("data:image/png;base64,!!!")})), "base64"},
      {user(Json::array({image_part("data:image/png;base64,aGVsbG8gd29ybGQ=")})), "PNG, JPEG, WebP, GIF or BMP"},
      {user(Json::array({image_part("data:image/png," + base64)})), "base64-encoded"},
      {user(Json::array({{{"type", "image_url"}}})), "needs a base64 data: URI"},
      {user(Json::array({{{"type", "text"}, {"text", "a <|unused_200054|> b"}}, image_part(red)})), "placeholder"},
      {Json::array({{{"role", "user"}, {"content", "hi"}},
                    {{"role", "assistant"}, {"content", Json::array({image_part(red)})}}}),
       "user and tool"},
      {user(Json::array({{{"type", "input_audio"}, {"input_audio", {{"data", "x"}}}}})), "text and images only"},
  };
  for (const auto& [messages, needle] : bad) {
    r = chat(s->port(), {{"messages", messages}});
    const std::string message = r.status == 400 ? r.body["error"]["message"].get<std::string>() : std::string();
    if (r.status != 400 || message.find(needle) == std::string::npos) {
      std::fprintf(stderr, "image request %s -> %d %s (expected '%s')\n", messages.dump().substr(0, 200).c_str(),
                   r.status, r.body.dump().c_str(), needle);
    }
    CHECK(r.status == 400 && r.body["error"]["param"] == "messages" && message.find(needle) != std::string::npos);
  }
  CHECK(s->gen->calls().size() == calls_before);
  // without an image the placeholder's spelling is ordinary text, as before
  r = chat(s->port(), {{"messages", user("a <|unused_200054|> b")}});
  CHECK(r.status == 200 && s->gen->calls().back().images == 0);

  // limits: images per request, image tokens per request, tokens per image (reduced, not rejected), context
  auto limited = [](ingine::serve::ImageOptions images, int max_model_len = 0) {
    auto served = std::make_unique<Served>();
    served->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test");
    ingine::serve::ServerOptions options;
    options.images = images;
    options.max_model_len = max_model_len;
    served->srv = std::make_unique<ChatServer>(*served->gen, *g_fmt, options);
    if (!served->srv->bind("127.0.0.1", 0)) throw std::runtime_error("bind failed");
    served->srv->start();
    return served;
  };
  ingine::serve::ImageOptions options;
  options.max_images = 1;
  auto few = limited(options);
  CHECK(chat(few->port(), {{"messages", user(Json::array({image_part(red)}))}}).status == 200);
  r = chat(few->port(), {{"messages", user(Json::array({image_part(red), image_part(blue)}))}});
  CHECK(r.status == 400 && r.body["error"]["message"].get<std::string>().find("at most 1") != std::string::npos);
  options = {};
  options.max_total_tokens = 6;
  auto tight = limited(options);
  CHECK(chat(tight->port(), {{"messages", user(Json::array({image_part(grad)}))}}).status == 200);
  r = chat(tight->port(), {{"messages", user(Json::array({image_part(grad), image_part(red)}))}});
  CHECK(r.status == 400 && r.body["error"]["message"].get<std::string>().find("7 tokens") != std::string::npos);
  options = {};
  options.limits.max_tokens = 4;
  auto small = limited(options);
  r = chat(small->port(), {{"messages", user(Json::array({image_part(grad)}))}});
  CHECK(r.status == 200 && small->gen->calls().back().images == 1);
  CHECK(small->gen->calls().back().image_tokens <= 4 && small->gen->calls().back().image_tokens >= 2);
  // the context check counts the image's tokens: 306 of them do not fit 200 positions
  auto narrow = limited({}, 200);
  CHECK(chat(narrow->port(), {{"messages", user(Json::array({image_part(red)}))}}).status == 200);
  r = chat(narrow->port(), {{"messages", user(Json::array({image_part(data_uri("medium_720x680.png"))}))}});
  CHECK(r.status == 400 && r.body["error"]["message"].get<std::string>().find("the model context is 200") !=
                               std::string::npos);

  // a server without the vision tower rejects images and says so in its model card
  options = {};
  options.enabled = false;
  auto text_only = limited(options);
  r = chat(text_only->port(), {{"messages", user(Json::array({image_part(red)}))}});
  CHECK(r.status == 400 && r.body["error"]["message"].get<std::string>().find("does not take images") !=
                               std::string::npos);
  CHECK(text_only->gen->calls().empty());
  CHECK(Json::parse(request(text_only->port(), "GET", "/v1/models").body)["data"][0]["input_modalities"] ==
        Json::array({"text"}));
  CHECK(chat(text_only->port(), Json::object()).status == 200);
}

void test_response_format_and_tool_ban() {
  auto s = serve();
  const int32_t tool_tok = g_fmt->content_invoke_tool_json_id();

  // No tools -> <|content_invoke_tool_json|> is banned in Sampling::banned_tokens
  Chat r = chat(s->port(), Json::object());
  CHECK(r.status == 200);
  {
    const auto calls = s->gen->calls();
    const auto& ban = calls.back().sampling.banned_tokens;
    CHECK(std::find(ban.begin(), ban.end(), tool_tok) != ban.end());
  }

  // With tools -> <|content_invoke_tool_json|> is NOT banned
  r = chat(s->port(), {{"tools", TOOLS}});
  CHECK(r.status == 200);
  {
    const auto calls = s->gen->calls();
    const auto& ban = calls.back().sampling.banned_tokens;
    CHECK(std::find(ban.begin(), ban.end(), tool_tok) == ban.end());
  }

  // response_format: text / json_object / json_schema
  CHECK(chat(s->port(), {{"response_format", {{"type", "text"}}}}).status == 200);
  CHECK(s->gen->calls().back().sampling.logit_masker == nullptr);
  const size_t base_len = s->gen->calls().back().prompt_ids.size();
  r = chat(s->port(), {{"response_format", {{"type", "json_object"}}}});
  CHECK(r.status == 200 && s->gen->calls().back().prompt_ids.size() > base_len);
  CHECK(s->gen->calls().back().sampling.logit_masker != nullptr);
  {
    const std::string content = r.body["choices"][0]["message"]["content"].get<std::string>();
    const Json parsed = Json::parse(content);
    CHECK(parsed.is_object());
  }
  const std::string decoded_obj = g_fmt->decode(s->gen->calls().back().prompt_ids);
  CHECK(decoded_obj.find("single valid JSON object") != std::string::npos);

  const Json schema = {
      {"type", "object"},
      {"additionalProperties", false},
      {"$defs", {{"Score", {{"type", "object"}, {"properties", {{"joy", {{"type", "number"}}}}}}}}},
      {"properties",
       {{"polarity", {{"type", "string"}, {"enum", Json::array({"positive", "negative", "neutral", "mixed"})}}},
        {"tags", {{"type", "array"}, {"items", {{"type", "string"}}}}},
        {"scores", {{"$ref", "#/$defs/Score"}}},
        {"answer", {{"type", "string"}}}}},
      {"required", Json::array({"polarity", "tags", "scores", "answer"})}};
  r = chat(s->port(), {{"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "ans"}, {"schema", schema}}}}}});
  CHECK(r.status == 200);
  CHECK(s->gen->calls().back().sampling.logit_masker != nullptr);
  {
    const std::string content = r.body["choices"][0]["message"]["content"].get<std::string>();
    const Json parsed = Json::parse(content);
    CHECK(parsed.is_object());
    CHECK(parsed.contains("polarity") && parsed.contains("tags") && parsed.contains("scores") &&
          parsed.contains("answer"));
    const std::string pol = parsed["polarity"].get<std::string>();
    CHECK(pol == "positive" || pol == "negative" || pol == "neutral" || pol == "mixed");
    CHECK(parsed["tags"].is_array());
    CHECK(parsed["scores"].is_object());
    CHECK(parsed["answer"].is_string());
  }
  const std::string decoded_schema = g_fmt->decode(s->gen->calls().back().prompt_ids);
  // Compiles into an instance skeleton + field constraints instead of echoing raw JSON Schema meta-object.
  CHECK(decoded_schema.find("\"answer\":\"<string>\"") != std::string::npos);
  CHECK(decoded_schema.find("\"polarity\":\"positive\"") != std::string::npos);
  CHECK(decoded_schema.find("\"tags\":[\"<string>\"]") != std::string::npos);
  CHECK(decoded_schema.find("\"scores\":{\"joy\":0.0}") != std::string::npos);
  CHECK(decoded_schema.find("polarity: one of [\"positive\",\"negative\",\"neutral\",\"mixed\"]") != std::string::npos);
  CHECK(decoded_schema.find("\"$defs\"") == std::string::npos);
  CHECK(decoded_schema.find("\"additionalProperties\":false") == std::string::npos);

  // Array length bounds (minItems/maxItems), numeric bounds, and string pattern/maxLength are preserved in constraints.
  const Json bounded_schema = {
      {"type", "object"},
      {"additionalProperties", false},
      {"properties",
       {{"city", {{"type", "string"}, {"maxLength", 40}}},
        {"population_millions", {{"type", "number"}, {"minimum", 0.0}, {"maximum", 50.0}}},
        {"landmarks", {{"type", "array"}, {"items", {{"type", "string"}}}, {"minItems", 2}, {"maxItems", 4}}}}},
      {"required", Json::array({"city", "population_millions", "landmarks"})}};
  r = chat(s->port(), {{"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "city"}, {"schema", bounded_schema}}}}}});
  CHECK(r.status == 200);
  const std::string decoded_bounded = g_fmt->decode(s->gen->calls().back().prompt_ids);
  CHECK(decoded_bounded.find("\"landmarks\":[\"<string>\",\"<string>\"]") != std::string::npos);
  CHECK(decoded_bounded.find("landmarks: array length between 2 and 4 items (inclusive)") != std::string::npos);
  CHECK(decoded_bounded.find("population_millions: >= 0.0") != std::string::npos);
  CHECK(decoded_bounded.find("population_millions: <= 50.0") != std::string::npos);
  CHECK(decoded_bounded.find("city: maxLength 40") != std::string::npos);

  // Issue #2: enable_thinking=false without reasoning_effort clamps template effort to 0 and bans <|content_thinking|>.
  const int32_t think_tok = g_fmt->content_thinking_id();
  r = chat(s->port(), {{"enable_thinking", false}});
  CHECK(r.status == 200);
  {
    const auto calls = s->gen->calls();
    const auto& call = calls.back();
    const std::string decoded_nt = g_fmt->decode(call.prompt_ids);
    CHECK(decoded_nt.find("Thinking effort level: 0<|end_message|>") != std::string::npos);
    CHECK(decoded_nt.find("Thinking effort level: 0.9") == std::string::npos);
    CHECK(std::find(call.sampling.banned_tokens.begin(), call.sampling.banned_tokens.end(), think_tok) !=
          call.sampling.banned_tokens.end());
  }
  // When enable_thinking=true (default), <|content_thinking|> is NOT banned.
  r = chat(s->port(), Json::object());
  CHECK(r.status == 200);
  {
    const auto calls = s->gen->calls();
    const auto& ban = calls.back().sampling.banned_tokens;
    CHECK(std::find(ban.begin(), ban.end(), think_tok) == ban.end());
  }

  // Invalid response_format & unsupported/unsatisfiable json_schema -> explicit 400 before generation
  CHECK(chat(s->port(), {{"response_format", "json_object"}}).status == 400);
  CHECK(chat(s->port(), {{"response_format", {{"type", "xml"}}}}).status == 400);
  CHECK(chat(s->port(), {{"response_format", {{"type", "json_schema"}}}}).status == 400);
  CHECK(chat(s->port(), {{"response_format",
                          {{"type", "json_schema"},
                           {"json_schema", {{"name", "bad_ref"}, {"schema", {{"$ref", "#/$defs/absent"}}}}}}}})
            .status == 400);
  CHECK(chat(s->port(), {{"response_format",
                          {{"type", "json_schema"},
                           {"json_schema",
                            {{"name", "bad_kw"}, {"schema", {{"type", "string"}, {"pattern", "^[a-z]+$"}}}}}}}})
            .status == 400);
  CHECK(chat(s->port(), {{"response_format",
                          {{"type", "json_schema"},
                           {"json_schema",
                            {{"name", "unsat"}, {"schema", {{"type", "integer"}, {"minimum", 10}, {"maximum", 5}}}}}}}})
            .status == 400);

  // Literal wire-byte HTTP regressions: prove exact decimal spellings survive HTTP transport without binary64 rounding!
  auto run_wire_http = [&](const std::string& schema_raw, const std::string& candidate_reply) -> std::string {
    // enable_thinking=false prefills <|message_model|><|content_text|>, so completion begins directly in content.
    const std::string forced = candidate_reply + "<|end_message|><|content_model_end_sampling|>";
    auto ws = serve(forced);
    const std::string raw_req =
        R"({"model":"inkling-test","enable_thinking":false,"max_tokens":32,"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"wire","schema":)" +
        schema_raw + R"(}}})";
    Response resp = request(ws->port(), "POST", "/v1/chat/completions", raw_req);
    CHECK(resp.status == 200);
    return Json::parse(resp.body)["choices"][0]["message"]["content"].get<std::string>();
  };
  // minimum 0.50000000000000001 must NOT admit 0.5, and MUST admit 0.50000000000000001
  CHECK(run_wire_http(R"({"type":"number","minimum":0.50000000000000001})", "0.5") != "0.5");
  CHECK(run_wire_http(R"({"type":"number","minimum":0.50000000000000001})", "0.50000000000000001") ==
        "0.50000000000000001");
  // maximum 0.49999999999999999 must NOT admit 0.5, and MUST admit 0.49999999999999999
  CHECK(run_wire_http(R"({"type":"number","maximum":0.49999999999999999})", "0.5") != "0.5");
  CHECK(run_wire_http(R"({"type":"number","maximum":0.49999999999999999})", "0.49999999999999999") ==
        "0.49999999999999999");
  // const 0.50000000000000001 must NOT admit 0.5, and MUST admit 0.50000000000000001
  CHECK(run_wire_http(R"({"const":0.50000000000000001})", "0.5") != "0.5");
  CHECK(run_wire_http(R"({"const":0.50000000000000001})", "0.50000000000000001") == "0.50000000000000001");
  // minimum 9007199254740993.0 (would round to 2^53 in double) must NOT admit 9007199254740992
  CHECK(run_wire_http(R"({"type":"number","minimum":9007199254740993.0})", "9007199254740992") != "9007199254740992");
  CHECK(run_wire_http(R"({"type":"number","minimum":9007199254740993.0})", "9007199254740993") == "9007199254740993");
  // nested oneOf with exclusiveMinimum 0.5 must NOT admit {"x":0.50000000000000001}, MUST admit {"x":0.5}
  const std::string nested_schema_raw =
      R"({"oneOf":[{"type":"object"},{"type":"object","required":["x"],"properties":{"x":{"type":"number","exclusiveMinimum":0.5}}}]})";
  CHECK(run_wire_http(nested_schema_raw, R"({"x":0.50000000000000001})") != R"({"x":0.50000000000000001})");
  CHECK(run_wire_http(nested_schema_raw, R"({"x":0.5})") == R"({"x":0.5})");
}

void test_persona_control_plane() {
  using ingine::persona::ModelRuntimeIdentity;
  using ingine::persona::PersonaCatalog;
  using ingine::persona::PersonaStatus;
  using ingine::persona::RefusalMap;

  const std::string hex64 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  auto make_prov = [&](const std::string& status = "validated") -> Json {
    return Json{
        {"status", status},
        {"purpose", "verified unit test intervention"},
        {"experiment", "unit_test_exp"},
        {"evidence_file", "results/unit.json"},
        {"evidence_split", "confirmation"},
        {"reaches", "tone"},
        {"origin", {{"character", "archivist"}, {"source_text", "train_prompt_01"}}},
        {"noop_check",
         {{"flag", "--force-unit 10:61:0.0:1.0"},
          {"kl", 0},
          {"delta_nll", 0},
          {"fused_path", "dense::router_hooked+prefill::moe_topk_hooked"},
          {"hardware", "RTX PRO 6000 Blackwell (sm_120a)"},
          {"visits", Json::array({3, 6, 10, 18, 29, 44})}}},
        {"measured_effect",
         {{"summary", "style shift above slot-permute floor"},
          {"metric", "style_score"},
          {"engine", "atlas-ingine"},
          {"weights_sha256", hex64},
          {"value", 0.42},
          {"floor", 0.05},
          {"floor_kind", "slot_permute_null"},
          {"floor_n", 4},
          {"sample_count", 16}}},
        {"refusal_checked",
         {{"passed", true}, {"evidence_file", "results/refusal.json"}, {"rate_before", 0.95}, {"rate_after", 0.95}}},
    };
  };
  auto make_spec = [&](const Json& profiles, const std::string& def_prof = "none") -> Json {
    return Json{
        {"schema_version", "atlas-persona-spec/v1"},
        {"spec_metadata", {{"created", "2026-11-24T01:00:00Z"}, {"author", "test-suite"}}},
        {"default_profile", def_prof},
        {"profiles", profiles},
    };
  };

  // 1. Valid 'validated' profile with full provenance + 6-visit noop_check + model_binding + leakage_check.
  Json val_profile = {
      {"description", "validated persona profile"},
      {"category", "validated"},
      {"model_binding",
       {{"weights_sha256", hex64},
        {"quantisation", "NVFP4"},
        {"engine_commit", "f3848d8"},
        {"prompt_renderer", "ingine_chat_format_v1"},
        {"chat_template_sha256", hex64}}},
      {"leakage_check", {{"passed", true}, {"evidence_file", "results/leakage.json"}}},
      {"interventions",
       Json::array({Json{{"id", "fit_1"},
                         {"type", "router_force"},
                         {"role", "probe"},
                         {"visit", 10},
                         {"expert", 61},
                         {"bias", 0.0},
                         {"gain", 1.25},
                         {"provenance", make_prov("validated")}}})},
  };
  const Json valid_doc = make_spec(Json{{"calm", val_profile}}, "none");
  const PersonaCatalog cat_strict = PersonaCatalog::parse_json(valid_doc.dump(), /*probe_mode=*/false);
  CHECK(cat_strict.default_profile == "none");
  CHECK(cat_strict.profiles.at("calm")->effective_status == PersonaStatus::Validated);
  CHECK(cat_strict.profiles.at("calm")->fingerprint_sha256.size() == 64);

  // 1b. Constant push claiming status='product' is rejected (probes are never product).
  {
    Json bad_prod = valid_doc;
    bad_prod["profiles"]["calm"]["category"] = "product";
    bad_prod["profiles"]["calm"]["interventions"][0]["provenance"]["status"] = "product";
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(bad_prod.dump(), true);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 1c. default_profile naming a non-product profile ('validated') is rejected outside probe mode.
  {
    Json bad_def = make_spec(Json{{"calm", val_profile}}, "calm");
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(bad_def.dump(), false);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 2. Missing one of the 6 required visits in noop_check.visits -> strict mode rejects (downgraded to probe).
  {
    Json bad = valid_doc;
    bad["profiles"]["calm"]["interventions"][0]["provenance"]["noop_check"]["visits"] = Json::array({3, 6, 10, 29, 44});
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(bad.dump(), false);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
    // In probe mode it loads, downgraded to probe with explicit reason.
    const PersonaCatalog cat_probe = PersonaCatalog::parse_json(bad.dump(), true);
    CHECK(cat_probe.profiles.at("calm")->effective_status == PersonaStatus::Probe);
    CHECK(!cat_probe.profiles.at("calm")->downgrade_reason.empty());
  }

  // 3. model_binding mismatch (quantisation, prompt_renderer, chat_template_sha256) downgrades to probe.
  {
    for (const auto& mismatch : {
             ModelRuntimeIdentity{"", "Q8_0", "", "", ""},
             ModelRuntimeIdentity{"", "", "", "atlas_eval_dump:reasoning_effort=none", ""},
             ModelRuntimeIdentity{"", "", "", "", std::string(64, 'a')},
         }) {
      bool threw = false;
      try {
        (void)PersonaCatalog::parse_json(valid_doc.dump(), false, RefusalMap::default_map(), mismatch);
      } catch (const std::exception&) {
        threw = true;
      }
      CHECK(threw);
      const PersonaCatalog cat_p = PersonaCatalog::parse_json(valid_doc.dump(), true, RefusalMap::default_map(), mismatch);
      CHECK(cat_p.profiles.at("calm")->effective_status == PersonaStatus::Probe);
    }
  }

  // 4. fit_method="trained" is unconditionally rejected even in probe mode.
  {
    Json trained = valid_doc;
    trained["profiles"]["calm"]["interventions"][0]["fit_method"] = "trained";
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(trained.dump(), true);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 5. Coordinate d_2027 in residual_steer is unconditionally rejected even in probe mode.
  {
    std::vector<float> vec(2304, 0.0f);
    vec[2027] = 0.01f;
    Json bad_steer = make_spec(Json{
        {"steer_bad",
         {{"category", "probe"},
          {"interventions",
           Json::array({{{"id", "s1"},
                         {"type", "residual_steer"},
                         {"role", "probe"},
                         {"visit", 10},
                         {"site", "mlp_write"},
                         {"scale", 1.0},
                         {"vector", vec}}})}}}});
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(bad_steer.dump(), true);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 6. Refusal-map protected unit is unconditionally rejected even in probe mode.
  {
    RefusalMap rmap = RefusalMap::default_map();
    rmap.protected_units.push_back({10, -1, 61, -1});
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(valid_doc.dump(), true, rmap);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 7. Sealed/heldout donor rejection even in probe mode.
  {
    Json sealed_donor = make_spec(Json{
        {"bad_replay",
         {{"category", "probe"},
          {"interventions",
           Json::array({{{"id", "r1"},
                         {"type", "route_replay"},
                         {"role", "probe"},
                         {"visit", 10},
                         {"donor",
                          {{"manifest_sha256", hex64},
                           {"row_id", "row_0"},
                           {"split", "heldout"},
                           {"raw_order", true}}}}})}}}});
    bool threw = false;
    try {
      (void)PersonaCatalog::parse_json(sealed_donor.dump(), true);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 8. HTTP control plane: GET /v1/personas, per-request "persona", persona_metadata on active only, 400 on unknown.
  {
    auto gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-test");
    ingine::serve::ServerOptions opts;
    opts.personas = std::make_shared<PersonaCatalog>(cat_strict);
    ChatServer srv(*gen, *g_fmt, opts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();

    const Response pr = request(srv.port(), "GET", "/v1/personas");
    CHECK(pr.status == 200);
    const Json pj = Json::parse(pr.body);
    CHECK(pj["object"] == "list" && pj["default"] == "none" && pj["probe_mode"] == false);
    CHECK(pj["data"].size() == 1 && pj["data"][0]["id"] == "calm" && pj["data"][0]["status"] == "validated");

    // Default persona ("none") uses unhooked fast path and omits persona_metadata.
    Chat r_def = chat(srv.port(), Json::object());
    CHECK(r_def.status == 200 && !r_def.body.contains("persona_metadata"));
    CHECK(gen->calls().back().sampling.persona == nullptr);

    // Explicit "persona": "calm" is applied and returns persona_metadata.
    Chat r_calm = chat(srv.port(), {{"persona", "calm"}});
    CHECK(r_calm.status == 200 && r_calm.body.contains("persona_metadata"));
    CHECK(r_calm.body["persona_metadata"]["profile"] == "calm");
    CHECK(r_calm.body["persona_metadata"]["status"] == "validated");
    CHECK(r_calm.body["persona_metadata"]["persona_fingerprint"] == cat_strict.profiles.at("calm")->fingerprint_sha256);
    CHECK(gen->calls().back().sampling.persona != nullptr &&
          gen->calls().back().sampling.persona->name == "calm");

    // Explicit "persona": "none" uses unhooked fast path and omits persona_metadata.
    Chat r_none = chat(srv.port(), {{"persona", "none"}});
    CHECK(r_none.status == 200 && !r_none.body.contains("persona_metadata"));
    CHECK(gen->calls().back().sampling.persona == nullptr);

    // Streaming with active persona emits persona_metadata on final finish chunk.
    Chat r_stream = chat(srv.port(), {{"persona", "calm"}, {"stream", true}});
    CHECK(r_stream.status == 200);
    bool saw_meta = false;
    for (const Json& ev : r_stream.body) {
      if (ev.is_object() && ev.contains("persona_metadata")) {
        CHECK(ev["persona_metadata"]["profile"] == "calm");
        saw_meta = true;
      }
    }
    CHECK(saw_meta);

    // Unknown persona or non-string persona -> 400 with param="persona".
    Chat r_bad = chat(srv.port(), {{"persona", "nonexistent"}});
    CHECK(r_bad.status == 400 && r_bad.body["error"]["param"] == "persona");
    Chat r_type = chat(srv.port(), {{"persona", 42}});
    CHECK(r_type.status == 400 && r_type.body["error"]["param"] == "persona");

    srv.stop();
  }
}

void test_cached_tokens_accounting() {
  using ingine::model::GenerateStats;
  using ingine::model::Reuse;

  const std::string long_reply =
      "<|message_model|><|content_text|>alpha beta<END> gamma delta epsilon zeta eta theta iota kappa lambda mu "
      "nu xi omicron pi rho sigma tau upsilon<|end_message|><|content_model_end_sampling|>";
  // Non-zero token delay guarantees pump() cancels on "<END>" while FakeGenerator is still in its decode loop,
  // verifying prefill stats are published strictly before token 0 (zero race on early stop-string finish).
  auto s = serve(long_reply, 0.01);

  std::mutex hook_mu;
  Reuse next_reuse = Reuse::None;
  int next_reused = 0;
  std::uint64_t next_restored = 0;
  s->gen->set_prefill_stats_hook([&](std::span<const int32_t>, GenerateStats& st) {
    std::lock_guard lock(hook_mu);
    st.reuse = next_reuse;
    st.reused_tokens = next_reused;
    st.restored_bytes = next_restored;
  });

  auto set_next = [&](Reuse r, int reused, std::uint64_t bytes = 0) {
    std::lock_guard lock(hook_mu);
    next_reuse = r;
    next_reused = reused;
    next_restored = bytes;
  };

  const Json conv_turn1 = Json::array({
      {{"role", "system"}, {"content", "You are a concise coding assistant. Follow exact rules."}},
      {{"role", "user"}, {"content", "Inspect main.cpp and summarize the bug."}},
  });
  const Json conv_turn2 = Json::array({
      {{"role", "system"}, {"content", "You are a concise coding assistant. Follow exact rules."}},
      {{"role", "user"}, {"content", "Inspect main.cpp and summarize the bug."}},
      {{"role", "assistant"}, {"content", "Found an off-by-one bug in parser.cpp line 42."}},
      {{"role", "user"}, {"content", "Now write the fix and verify."}},
  });

  // 1. Cold request: cached_tokens == 0 on both non-streaming and streaming (include_usage=true).
  g_fmt->clear_render_cache();
  set_next(Reuse::None, 0, 0);
  Chat r_cold = chat(s->port(), {{"messages", conv_turn1}});
  CHECK(r_cold.status == 200);
  check_completion(r_cold.body);
  const int p1 = r_cold.body["usage"]["prompt_tokens"].get<int>();
  CHECK(p1 > 10);
  CHECK(r_cold.body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);

  set_next(Reuse::None, 0, 0);
  Chat s_cold = chat(s->port(), {{"messages", conv_turn1},
                                 {"stream", true},
                                 {"stream_options", {{"include_usage", true}}}});
  CHECK(s_cold.status == 200);
  Stream st_cold = check_chunks(s_cold.body, true);
  // CPU RenderCache hit on repeated conv_turn1 must NOT create a GPU cached_tokens hit!
  CHECK(st_cold.usage["prompt_tokens"] == p1);
  CHECK(st_cold.usage["prompt_tokens_details"]["cached_tokens"] == 0);

  // 2. Known warm request: exact expected cached_tokens on both JSON and SSE across reuse kinds.
  const int expected_warm = p1;
  for (Reuse kind : {Reuse::Extend, Reuse::Checkpoint, Reuse::Parked, Reuse::RootCheckpoint, Reuse::HostParked,
                     Reuse::PagedPrefix}) {
    set_next(kind, expected_warm, 4096);
    Chat r_warm = chat(s->port(), {{"messages", conv_turn2}});
    CHECK(r_warm.status == 200);
    check_completion(r_warm.body);
    CHECK(r_warm.body["usage"]["prompt_tokens"].get<int>() > expected_warm);
    CHECK(r_warm.body["usage"]["prompt_tokens_details"]["cached_tokens"] == expected_warm);

    set_next(kind, expected_warm, 4096);
    Chat s_warm = chat(s->port(), {{"messages", conv_turn2},
                                   {"stream", true},
                                   {"stream_options", {{"include_usage", true}}}});
    CHECK(s_warm.status == 200);
    Stream st_warm = check_chunks(s_warm.body, true);
    CHECK(st_warm.usage["prompt_tokens_details"]["cached_tokens"] == expected_warm);
  }

  // 3. Deliberate miss after warm hits -> cached_tokens == 0 (both JSON and SSE).
  set_next(Reuse::None, 0, 0);
  Chat r_miss = chat(s->port(), {{"messages", conv_turn2}});
  CHECK(r_miss.status == 200 && r_miss.body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);
  set_next(Reuse::None, 0, 0);
  Chat s_miss = chat(s->port(), {{"messages", conv_turn2},
                                 {"stream", true},
                                 {"stream_options", {{"include_usage", true}}}});
  CHECK(s_miss.status == 200);
  CHECK(check_chunks(s_miss.body, true).usage["prompt_tokens_details"]["cached_tokens"] == 0);

  // 4. Early stop-string finish (non-streaming and streaming): prefill stats must already be visible!
  set_next(Reuse::Checkpoint, expected_warm, 8192);
  Chat r_stop = chat(s->port(), {{"messages", conv_turn2}, {"stop", Json::array({"<END>"})}});
  CHECK(r_stop.status == 200);
  check_completion(r_stop.body);
  CHECK(r_stop.body["choices"][0]["message"]["content"] == "alpha beta");
  CHECK(r_stop.body["choices"][0]["finish_reason"] == "stop");
  CHECK(r_stop.body["usage"]["prompt_tokens_details"]["cached_tokens"] == expected_warm);

  set_next(Reuse::Checkpoint, expected_warm, 8192);
  Chat s_stop = chat(s->port(), {{"messages", conv_turn2},
                                 {"stop", Json::array({"<END>"})},
                                 {"stream", true},
                                 {"stream_options", {{"include_usage", true}}}});
  CHECK(s_stop.status == 200);
  Stream st_stop = check_chunks(s_stop.body, true);
  CHECK(st_stop.content == "alpha beta" && st_stop.finish == "stop");
  CHECK(st_stop.usage["prompt_tokens_details"]["cached_tokens"] == expected_warm);

  // 5. Defensive clamping of out-of-range reused_tokens.
  set_next(Reuse::Extend, -7, 0);
  Chat r_neg = chat(s->port(), {{"messages", conv_turn1}});
  CHECK(r_neg.status == 200 && r_neg.body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);

  set_next(Reuse::Extend, 9999999, 0);
  Chat r_hi = chat(s->port(), {{"messages", conv_turn1}});
  CHECK(r_hi.status == 200 && r_hi.body["usage"]["prompt_tokens_details"]["cached_tokens"] == p1);
}

void test_legacy_generator_early_stop_cached_tokens() {
  // Regression for review item #1: a legacy Generator subclass implementing ONLY the 5-arg generate()
  // (never overriding the 6-arg on_prefill overload) that reports a warm hit and delays after an early
  // stop-string cancellation must STILL report exact cached_tokens on both non-streaming JSON and SSE.
  class LegacyWarmGenerator final : public ingine::model::Generator {
   public:
    explicit LegacyWarmGenerator(const ingine::text::Tokenizer& tok) : tok_(tok) {}
    std::string model_id() const override { return "inkling-test"; }
    int max_model_len() const override { return 1 << 20; }
    ingine::model::GenerateStats generate(
        std::span<const int32_t> prompt, int max_new_tokens, const ingine::model::Sampling&,
        std::span<const int32_t> stop_ids,
        const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
      const std::string reply =
          "<|message_model|><|content_text|>alpha beta<END> trailing words that should be cancelled"
          "<|end_message|><|content_model_end_sampling|>";
      const std::vector<int32_t> out = tok_.encode(reply, /*special=*/true);
      const std::unordered_set<int32_t> stops(stop_ids.begin(), stop_ids.end());
      const std::size_t lim = std::min(out.size(), std::size_t(std::max(max_new_tokens, 0)));
      int yielded = 0;
      for (std::size_t i = 0; i < lim; ++i) {
        ++yielded;
        const bool more = on_tokens(std::span<const int32_t>(&out[i], 1));
        if (stops.contains(out[i]) || !more) break;
      }
      // Deliberately delay AFTER early-stop cancellation returns false from on_tokens(): without
      // wait_prefill_stats() synchronization, usage() would race and observe cached_tokens == 0.
      std::this_thread::sleep_for(std::chrono::milliseconds(80));
      ingine::model::GenerateStats st;
      st.prompt_tokens = int(prompt.size());
      st.reused_tokens = std::max(0, int(prompt.size()) - 3);
      st.reuse = ingine::model::Reuse::Checkpoint;
      st.generated_tokens = yielded;
      return st;
    }

   private:
    const ingine::text::Tokenizer& tok_;
  };

  LegacyWarmGenerator gen(g_fmt->tokenizer());
  ChatServer srv(gen, *g_fmt);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();

  const Json msgs = Json::array({
      {{"role", "system"}, {"content", "Legacy generator system prompt."}},
      {{"role", "user"}, {"content", "Emit alpha beta<END>."}},
  });
  const int p_len = int(render(msgs).ids.size());
  const int expected_cached = std::max(0, p_len - 3);

  Chat r_json = chat(srv.port(), {{"messages", msgs}, {"stop", Json::array({"<END>"})}});
  CHECK(r_json.status == 200);
  check_completion(r_json.body);
  CHECK(r_json.body["choices"][0]["message"]["content"] == "alpha beta");
  CHECK(r_json.body["choices"][0]["finish_reason"] == "stop");
  CHECK(r_json.body["usage"]["prompt_tokens_details"]["cached_tokens"] == expected_cached);

  Chat r_sse = chat(srv.port(), {{"messages", msgs},
                                 {"stop", Json::array({"<END>"})},
                                 {"stream", true},
                                 {"stream_options", {{"include_usage", true}}}});
  CHECK(r_sse.status == 200);
  Stream st_sse = check_chunks(r_sse.body, true);
  CHECK(st_sse.content == "alpha beta" && st_sse.finish == "stop");
  CHECK(st_sse.usage["prompt_tokens_details"]["cached_tokens"] == expected_cached);

  srv.stop();
}

void test_prompt_stability_and_render_stats() {
  // Verify actual post-template token IDs under Atlas's chat_template.jinja so harness prompt-stability rules
  // are grounded in measured token IDs, not assumptions.
  auto common_prefix_len = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) -> std::size_t {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
  };

  ingine::text::ChatOptions opts;
  opts.reasoning_effort = "medium";
  opts.enable_thinking = true;

  const Json base_msgs = Json::array({
      {{"role", "system"}, {"content", "Static system prompt for coding tasks."}},
      {{"role", "user"}, {"content", "Run tests in /workspace/app."}},
  });
  g_fmt->clear_render_cache();
  const auto p_base = g_fmt->render(base_msgs, TOOLS, opts);
  CHECK(!p_base.stats.cache_hit && p_base.stats.reused_text_tokens == 0 &&
        p_base.stats.encoded_text_tokens == int(p_base.ids.size()));

  // A) Repeating the same prompt hits CPU RenderCache (exact_hit=true) and reports non-zero cache occupancy.
  const auto p_repeat = g_fmt->render(base_msgs, TOOLS, opts);
  CHECK(p_repeat.ids == p_base.ids && p_repeat.stats.cache_hit && p_repeat.stats.exact_hit &&
        p_repeat.stats.reused_text_tokens == int(p_base.ids.size()) && p_repeat.stats.encoded_text_tokens == 0);
  const auto cstats = g_fmt->render_cache_stats();
  CHECK(cstats.entries >= 1 && cstats.bytes > 0);

  // B) Appending a mid-history system delta (e.g. dynamic cwd/runtime note) preserves 100% of p_base.ids.
  Json with_mid_sys = base_msgs;
  with_mid_sys.push_back({{"role", "system"}, {"content", "Current working directory: /workspace/app/sub"}});
  with_mid_sys.push_back({{"role", "user"}, {"content", "Continue."}});
  const auto p_mid_sys = g_fmt->render(with_mid_sys, TOOLS, opts);
  CHECK(common_prefix_len(p_base.ids, p_mid_sys.ids) == p_base.ids.size());
  CHECK(p_mid_sys.stats.cache_hit && p_mid_sys.stats.reused_text_tokens > 0);

  // C) Mutating `tools` (rendered at position 0 by chat_template.jinja) invalidates the prefix near token 0.
  Json extra_tools = TOOLS;
  extra_tools.push_back(
      {{"type", "function"}, {"function", {{"name", "run_shell"}, {"parameters", {{"type", "object"}}}}}});
  const auto p_tools_changed = g_fmt->render(base_msgs, extra_tools, opts);
  // Only the opening <|message_system|>tool_declare<|content_xml|> + identical first tool prefix survives;
  // it breaks BEFORE base_msgs system/user content begins.
  CHECK(common_prefix_len(p_base.ids, p_tools_changed.ids) < p_base.ids.size());

  // D) Changing reasoning_effort mid-conversation invalidates everything after the leading system message.
  ingine::text::ChatOptions opts_high = opts;
  opts_high.reasoning_effort = "high";
  const auto p_effort_changed = g_fmt->render(base_msgs, TOOLS, opts_high);
  CHECK(common_prefix_len(p_base.ids, p_effort_changed.ids) < p_base.ids.size());

  // E) Preserving vs dropping assistant reasoning_content determines exact prefix continuation length.
  Json with_reasoning = base_msgs;
  with_reasoning.push_back({{"role", "assistant"},
                            {"reasoning_content", "Let me check pytest output."},
                            {"content", "Running pytest now."}});
  with_reasoning.push_back({{"role", "user"}, {"content", "Any failures?"}});
  Json without_reasoning = base_msgs;
  without_reasoning.push_back({{"role", "assistant"}, {"content", "Running pytest now."}});
  without_reasoning.push_back({{"role", "user"}, {"content", "Any failures?"}});
  const auto p_with_r = g_fmt->render(with_reasoning, TOOLS, opts);
  const auto p_without_r = g_fmt->render(without_reasoning, TOOLS, opts);
  CHECK(common_prefix_len(p_base.ids, p_with_r.ids) == p_base.ids.size());
  CHECK(common_prefix_len(p_base.ids, p_without_r.ids) == p_base.ids.size());
  CHECK(common_prefix_len(p_with_r.ids, p_without_r.ids) == p_base.ids.size() + 1);  // diverges right at <|content_thinking|> vs <|content_text|>
}

void test_cache_scope_isolation() {
  // 1. Default single trusted scope (cache_namespace_header unset):
  //    - Plain unauthenticated request shares default scope (bypass=false, namespace="").
  //    - Request with Authorization header, JSON `user`, or JSON `prompt_cache_key` fails closed (bypass=true).
  {
    auto s = serve();
    const Json msgs = Json::array({{{"role", "system"}, {"content", "Be concise."}}, {{"role", "user"}, {"content", "hi"}}});
    Chat r = chat(s->port(), {{"messages", msgs}});
    CHECK(r.status == 200);
    const auto c0 = s->gen->calls().back();
    CHECK(c0.root_prefix_tokens > 0 && c0.cache_namespace.empty() && !c0.bypass_prefix_cache);

    // JSON `user` -> fail-closed bypass
    r = chat(s->port(), {{"messages", msgs}, {"user", "alice"}});
    CHECK(r.status == 200 && s->gen->calls().back().bypass_prefix_cache);

    // JSON `prompt_cache_key` -> fail-closed bypass
    r = chat(s->port(), {{"messages", msgs}, {"prompt_cache_key", "sess-42"}});
    CHECK(r.status == 200 && s->gen->calls().back().bypass_prefix_cache);

    // HTTP `Authorization` header -> fail-closed bypass
    httplib::Client cli("127.0.0.1", s->port());
    httplib::Headers hdrs{{"Authorization", "Bearer tok-xyz"}};
    auto res = cli.Post("/v1/chat/completions", hdrs, ingine::serve::dump(Json{{"messages", msgs}}), "application/json");
    CHECK(res && res->status == 200 && s->gen->calls().back().bypass_prefix_cache);
  }

  // 2. Trusted proxy header mode (cache_namespace_header = "X-Tenant-Scope"):
  //    - Present header propagates exact namespace with bypass=false.
  //    - Missing or empty header fails closed with bypass=true.
  {
    auto s = std::make_unique<Served>();
    s->gen = std::make_unique<FakeGenerator>(
        g_fmt->tokenizer(), "inkling-test", 1 << 20,
        [](std::span<const int32_t>) { return FakeGenerator::Output(THINK_TEXT); }, 0.0);
    ingine::serve::ServerOptions opt;
    opt.cache_namespace_header = "X-Tenant-Scope";
    s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, opt);
    if (!s->srv->bind("127.0.0.1", 0)) throw std::runtime_error("bind failed");
    s->srv->start();
    const Json msgs = Json::array({{{"role", "system"}, {"content", "Be concise."}}, {{"role", "user"}, {"content", "hi"}}});
    httplib::Client cli("127.0.0.1", s->port());

    auto res1 = cli.Post("/v1/chat/completions", httplib::Headers{{"X-Tenant-Scope", "tenant-a"}},
                         ingine::serve::dump(Json{{"messages", msgs}}), "application/json");
    CHECK(res1 && res1->status == 200);
    const auto c1 = s->gen->calls().back();
    CHECK(c1.cache_namespace == "tenant-a" && !c1.bypass_prefix_cache);

    auto res2 = cli.Post("/v1/chat/completions", httplib::Headers{},
                         ingine::serve::dump(Json{{"messages", msgs}}), "application/json");
    CHECK(res2 && res2->status == 200);
    const auto c2 = s->gen->calls().back();
    CHECK(c2.cache_namespace.empty() && c2.bypass_prefix_cache);
  }
}

void test_kv_cache_mode_advertisement() {
  // 1. Active last_full worker advertising both per_pass and last_full.
  {
    auto s = std::make_unique<Served>();
    auto responder = [](std::span<const int32_t>) { return FakeGenerator::Output(THINK_TEXT); };
    s->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-lf", 1 << 20, responder, 0.0);
    s->gen->set_kv_cache_mode(ingine::model::KvCacheMode::LastFull);
    s->gen->set_supported_kv_cache_modes({ingine::model::KvCacheMode::PerPass, ingine::model::KvCacheMode::LastFull});
    s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, ingine::serve::ServerOptions{});
    CHECK(s->srv->bind("127.0.0.1", 0));
    s->srv->start();

    Response hr = request(s->port(), "GET", "/health");
    CHECK(hr.status == 200 && header(hr.headers, "X-Ingine-Kv-Cache-Mode") == "last_full");
    Json h = Json::parse(hr.body);
    CHECK(h["kv_cache_mode"] == "last_full");
    CHECK(h["supported_kv_cache_modes"] == Json::array({"per_pass", "last_full"}));
    CHECK(h["kv_cache_mode_mutable"] == false);

    Response mr = request(s->port(), "GET", "/v1/models");
    CHECK(mr.status == 200 && header(mr.headers, "X-Ingine-Kv-Cache-Mode") == "last_full");
    const Json m = Json::parse(mr.body)["data"][0];
    CHECK(m["kv_cache_mode"] == "last_full");
    CHECK(m["supported_kv_cache_modes"] == Json::array({"per_pass", "last_full"}));
    CHECK(m["kv_cache_mode_mutable"] == false);

    Chat r = chat(s->port(), {{"model", "inkling-lf"}});
    CHECK(r.status == 200);
    CHECK(header(r.headers, "X-Ingine-Kv-Cache-Mode") == "last_full");
    CHECK(r.body["kv_cache_mode"] == "last_full");

    Chat sr = chat(s->port(), {{"model", "inkling-lf"}, {"stream", true}});
    CHECK(sr.status == 200);
    CHECK(header(sr.headers, "X-Ingine-Kv-Cache-Mode") == "last_full");
  }

  // 2. Restricted worker (e.g. --spec, B>1, or TP>1) advertising only per_pass.
  {
    auto s = std::make_unique<Served>();
    auto responder = [](std::span<const int32_t>) { return FakeGenerator::Output(THINK_TEXT); };
    s->gen = std::make_unique<FakeGenerator>(g_fmt->tokenizer(), "inkling-pp-only", 1 << 20, responder, 0.0);
    s->gen->set_kv_cache_mode(ingine::model::KvCacheMode::PerPass);
    s->gen->set_supported_kv_cache_modes({ingine::model::KvCacheMode::PerPass});
    s->srv = std::make_unique<ChatServer>(*s->gen, *g_fmt, ingine::serve::ServerOptions{});
    CHECK(s->srv->bind("127.0.0.1", 0));
    s->srv->start();

    Json h = health(s->port());
    CHECK(h["kv_cache_mode"] == "per_pass");
    CHECK(h["supported_kv_cache_modes"] == Json::array({"per_pass"}));
    CHECK(h["kv_cache_mode_mutable"] == false);

    Chat r = chat(s->port(), {{"model", "inkling-pp-only"}});
    CHECK(r.status == 200 && header(r.headers, "X-Ingine-Kv-Cache-Mode") == "per_pass");
    CHECK(r.body["kv_cache_mode"] == "per_pass");
  }
}

void test_anthropic_messages_api() {
  auto parse_anthropic_sse = [](const std::string& raw) {
    std::vector<std::pair<std::string, Json>> events;
    std::string cur_ev;
    std::size_t pos = 0;
    while (pos < raw.size()) {
      std::size_t nl = raw.find('\n', pos);
      std::string line = (nl == std::string::npos) ? raw.substr(pos) : raw.substr(pos, nl - pos);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      pos = (nl == std::string::npos) ? raw.size() : nl + 1;
      if (line.starts_with("event: ")) cur_ev = line.substr(7);
      else if (line.starts_with("data: ")) events.emplace_back(cur_ev, Json::parse(line.substr(6)));
    }
    return events;
  };

  // 1. Non-streaming text + thinking + system block array + count_tokens + stop_sequences.
  {
    auto s = serve(THINK_TEXT);
    const Json body = {
        {"model", "inkling-test"},
        {"max_tokens", 256},
        {"system", Json::array({{{"type", "text"}, {"text", "Be brief."}, {"cache_control", {{"type", "ephemeral"}}}}})},
        {"messages", Json::array({{{"role", "user"}, {"content", Json::array({{{"type", "text"}, {"text", "Hi!"}}})}}})},
        {"stop_sequences", Json::array({"STOP"})},
    };
    Response r = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(body));
    CHECK(r.status == 200);
    const Json j = Json::parse(r.body);
    CHECK(j["type"] == "message" && j["role"] == "assistant" && j["model"] == "inkling-test");
    CHECK(j["id"].get<std::string>().starts_with("msg_"));
    CHECK(j["stop_reason"] == "stop_sequence");
    CHECK(j["stop_sequence"] == "STOP");
    CHECK(j["content"].size() == 2);
    CHECK(j["content"][0]["type"] == "thinking");
    CHECK(j["content"][0]["thinking"].get<std::string>().find("Let me think") != std::string::npos);
    CHECK(j["content"][1]["type"] == "text");
    CHECK(j["content"][1]["text"].get<std::string>().find("STOP") == std::string::npos);
    CHECK(j["usage"]["input_tokens"].get<int>() > 0 && j["usage"]["output_tokens"].get<int>() > 0);

    // /v1/messages/count_tokens matches input_tokens without requiring max_tokens.
    Json count_body = body;
    count_body.erase("max_tokens");
    Response cr = request(s->port(), "POST", "/v1/messages/count_tokens", ingine::serve::dump(count_body));
    CHECK(cr.status == 200);
    CHECK(Json::parse(cr.body)["input_tokens"].get<int>() == j["usage"]["input_tokens"].get<int>());

    // Normal end_turn (no stop sequence match) returns stop_reason="end_turn" and stop_sequence=null.
    Json no_stop = body;
    no_stop.erase("stop_sequences");
    Response nr2 = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(no_stop));
    CHECK(nr2.status == 200);
    const Json j2 = Json::parse(nr2.body);
    CHECK(j2["stop_reason"] == "end_turn" && j2["stop_sequence"].is_null());
  }

  // 2. Streaming SSE reconstruction matches non-streaming for thinking + tool_use blocks.
  {
    auto s = serve(TOOL_TEXT);
    const Json tools = Json::array({
        {{"name", "get_weather"},
         {"description", "Get weather"},
         {"input_schema", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}}}},
        {{"name", "search"}, {"input_schema", {{"type", "object"}}}},
    });
    const Json base = {
        {"model", "inkling-test"},
        {"max_tokens", 256},
        {"tools", tools},
        {"messages", Json::array({{{"role", "user"}, {"content", "Weather in Sao Paulo?"}}})},
    };
    Response nr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(base));
    CHECK(nr.status == 200);
    const Json nj = Json::parse(nr.body);
    CHECK(nj["stop_reason"] == "tool_use");
    CHECK(nj["content"].size() == 3);
    CHECK(nj["content"][0]["type"] == "thinking");
    CHECK(nj["content"][1]["type"] == "tool_use" && nj["content"][1]["name"] == "get_weather");
    CHECK(nj["content"][1]["input"]["days"] == 3);
    CHECK(nj["content"][2]["type"] == "tool_use" && nj["content"][2]["name"] == "search");
    CHECK(nj["content"][2]["input"]["q"] == "rain");

    Json sbody = base;
    sbody["stream"] = true;
    Response sr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(sbody));
    CHECK(sr.status == 200);
    const auto evs = parse_anthropic_sse(sr.body);
    CHECK(!evs.empty() && evs.front().first == "message_start" && evs.back().first == "message_stop");
    std::string s_thinking, s_stop;
    std::map<int, std::string> tool_names, tool_json;
    for (const auto& [ev, d] : evs) {
      if (ev == "content_block_start" && d["content_block"]["type"] == "tool_use") {
        tool_names[d["index"].get<int>()] = d["content_block"]["name"].get<std::string>();
      } else if (ev == "content_block_delta") {
        const auto& dl = d["delta"];
        if (dl["type"] == "thinking_delta") s_thinking += dl["thinking"].get<std::string>();
        else if (dl["type"] == "input_json_delta") tool_json[d["index"].get<int>()] += dl["partial_json"].get<std::string>();
      } else if (ev == "message_delta") {
        s_stop = d["delta"]["stop_reason"].get<std::string>();
      }
    }
    CHECK(s_thinking == nj["content"][0]["thinking"].get<std::string>());
    CHECK(s_stop == "tool_use");
    CHECK(tool_names.size() == 2 && tool_names[1] == "get_weather" && tool_names[2] == "search");
    CHECK(Json::parse(tool_json[1]) == nj["content"][1]["input"]);
    CHECK(Json::parse(tool_json[2]) == nj["content"][2]["input"]);

    // Multi-turn tool_use + tool_result round trip renders cleanly.
    const Json follow = {
        {"model", "inkling-test"},
        {"max_tokens", 64},
        {"tools", tools},
        {"messages",
         Json::array({
             {{"role", "user"}, {"content", "Weather?"}},
             {{"role", "assistant"}, {"content", nj["content"]}},
             {{"role", "user"},
              {"content",
               Json::array({{{"type", "tool_result"},
                             {"tool_use_id", nj["content"][1]["id"]},
                             {"content", Json::array({{{"type", "text"}, {"text", "24C sunny"}}})}},
                            {{"type", "text"}, {"text", "Summarize."}}})}},
         })},
    };
    Response fr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(follow));
    CHECK(fr.status == 200);
  }

  // 3. Thinking disabled, error envelope, and x-api-key / metadata.user_id prefix-cache fail-closed isolation.
  {
    auto s = serve(THINK_TEXT);
    const Json missing_max = {{"model", "inkling-test"}, {"messages", MSGS}};
    Response br = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(missing_max));
    CHECK(br.status == 400);
    const Json bj = Json::parse(br.body);
    CHECK(bj["type"] == "error" && bj["error"]["type"] == "invalid_request_error");

    const Json no_think = {
        {"model", "inkling-test"},
        {"max_tokens", 32},
        {"thinking", {{"type", "disabled"}}},
        {"messages", MSGS},
    };
    Response tr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(no_think));
    CHECK(tr.status == 200);
    const auto calls_tr = s->gen->calls();
    const auto& banned = calls_tr.back().sampling.banned_tokens;
    CHECK(std::find(banned.begin(), banned.end(), g_fmt->content_thinking_id()) != banned.end());

    // x-api-key header fails closed for prefix caching when cache_namespace_header is unset.
    const Json req_ok = {{"model", "inkling-test"}, {"max_tokens", 32}, {"messages", MSGS}};
    Response kr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(req_ok),
                          httplib::Headers{{"x-api-key", "sk-ant-test"}});
    CHECK(kr.status == 200 && s->gen->calls().back().bypass_prefix_cache);

    // metadata.user_id also triggers fail-closed prefix-cache isolation.
    Json req_uid = req_ok;
    req_uid["metadata"] = {{"user_id", "omarchy-user-1"}};
    Response ur = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(req_uid));
    CHECK(ur.status == 200 && s->gen->calls().back().bypass_prefix_cache);

    // Unsupported tool_choice ('any' and named 'tool') fail closed before inference (HTTP 400).
    const std::size_t calls_before = s->gen->calls().size();
    const Json tools = Json::array({{{"name", "lookup"}, {"input_schema", {{"type", "object"}}}}});
    for (const Json& tc : {Json{{"type", "any"}}, Json{{"type", "tool"}, {"name", "lookup"}}}) {
      Json bad_tc = {{"model", "inkling-test"}, {"max_tokens", 32}, {"tools", tools}, {"tool_choice", tc}, {"messages", MSGS}};
      Response r_bad = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(bad_tc));
      CHECK(r_bad.status == 400);
      const Json ej = Json::parse(r_bad.body);
      CHECK(ej["type"] == "error" && ej["error"]["type"] == "invalid_request_error");
    }
    CHECK(s->gen->calls().size() == calls_before);

    // tool_choice={"type":"none"} strips tools and bans <|content_invoke_tool_json|>.
    Json none_tc = {
        {"model", "inkling-test"},
        {"max_tokens", 32},
        {"tools", tools},
        {"tool_choice", {{"type", "none"}}},
        {"messages", MSGS},
    };
    Response r_none = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(none_tc));
    CHECK(r_none.status == 200);
    const auto calls_none = s->gen->calls();
    const auto& banned_none = calls_none.back().sampling.banned_tokens;
    CHECK(std::find(banned_none.begin(), banned_none.end(), g_fmt->content_invoke_tool_json_id()) != banned_none.end());
  }

  // 4. User stop_sequences (including cross-token matches) emit stop_reason="stop_sequence" and exact stop_sequence
  //    in both non-streaming JSON and streaming SSE.
  {
    // "END_MARK_XYZ" spans multiple BPE tokens so StreamParser must match across token boundaries via hold_.
    const std::string stop_seq = "END_MARK_XYZ";
    CHECK(g_fmt->tokenizer().encode(stop_seq, false).size() >= 2);
    const std::string stop_text =
        "<|message_model|><|content_text|>alpha beta " + stop_seq + " trailing<|end_message|><|content_model_end_sampling|>";
    auto s = serve(stop_text);

    const Json req_sync = {
        {"model", "inkling-test"},
        {"max_tokens", 64},
        {"stop_sequences", Json::array({"OTHER_STOP", stop_seq})},
        {"messages", MSGS},
    };
    Response nr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(req_sync));
    CHECK(nr.status == 200);
    const Json nj = Json::parse(nr.body);
    CHECK(nj["stop_reason"] == "stop_sequence");
    CHECK(nj["stop_sequence"] == stop_seq);
    CHECK(nj["content"].size() == 1 && nj["content"][0]["type"] == "text");
    CHECK(nj["content"][0]["text"] == "alpha beta ");

    Json req_stream = req_sync;
    req_stream["stream"] = true;
    Response sr = request(s->port(), "POST", "/v1/messages", ingine::serve::dump(req_stream));
    CHECK(sr.status == 200);
    const auto evs = parse_anthropic_sse(sr.body);
    CHECK(!evs.empty() && evs.front().first == "message_start" && evs.back().first == "message_stop");
    std::string s_text, s_reason, s_seq;
    for (const auto& [ev, d] : evs) {
      if (ev == "content_block_delta" && d["delta"]["type"] == "text_delta") {
        s_text += d["delta"]["text"].get<std::string>();
      } else if (ev == "message_delta") {
        s_reason = d["delta"]["stop_reason"].get<std::string>();
        s_seq = d["delta"]["stop_sequence"].get<std::string>();
      }
    }
    CHECK(s_text == "alpha beta ");
    CHECK(s_reason == "stop_sequence");
    CHECK(s_seq == stop_seq);
  }
}

void test_loopcd_advertisement() {
  using ingine::model::LoopCdMode;
  for (LoopCdMode mode : {LoopCdMode::Off, LoopCdMode::LogitsAdaptive, LoopCdMode::Hidden}) {
    auto s = serve();
    s->gen->set_loopcd_mode(mode);
    // Construct a fresh server bound to the generator with this LoopCD mode.
    ingine::serve::ServerOptions opts;
    opts.model_id = "inkling-loopcd-test";
    ingine::serve::ChatServer srv(*s->gen, *g_fmt, opts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    const std::string expected = ingine::model::to_string(mode);
    Response h = request(srv.port(), "GET", "/health");
    CHECK(h.status == 200);
    CHECK(Json::parse(h.body)["loopcd"] == expected);
    CHECK(header(h.headers, "X-Ingine-LoopCD") == expected);

    Response m = request(srv.port(), "GET", "/v1/models");
    CHECK(m.status == 200);
    CHECK(Json::parse(m.body)["data"][0]["loopcd"] == expected);
    CHECK(header(m.headers, "X-Ingine-LoopCD") == expected);

    const Json req_body = {{"model", "inkling-loopcd-test"}, {"messages", MSGS}, {"max_tokens", 16}};
    Response c = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(req_body));
    CHECK(c.status == 200);
    CHECK(header(c.headers, "X-Ingine-LoopCD") == expected);
    srv.stop();
  }
}

void test_warm_start_system_prompt() {
  const std::string sys_prompt = "You are a concise, accurate systems engineering assistant.";

  // 1. Unset baseline: warm_start=false, warm_start_tokens=0, zero warm_start calls, Turn-1 cached_tokens=0.
  {
    FakeGenerator gen(g_fmt->tokenizer(), "inkling-ws-off", 1 << 20);
    ingine::serve::ServerOptions opts;
    opts.model_id = "inkling-ws-off";
    ingine::serve::ChatServer srv(gen, *g_fmt, opts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    CHECK(gen.warm_start_calls().empty());
    const Json h = health(srv.port());
    CHECK(h["warm_start"] == false && h["warm_start_tokens"] == 0);
    const Json body = {
        {"model", "inkling-ws-off"},
        {"messages", Json::array({{{"role", "system"}, {"content", sys_prompt}}, {{"role", "user"}, {"content", "Hi"}}})},
        {"max_tokens", 16}};
    const Response r = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(body));
    CHECK(r.status == 200);
    CHECK(Json::parse(r.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);
    const Json abody = {{"model", "inkling-ws-off"},
                        {"max_tokens", 16},
                        {"system", sys_prompt},
                        {"messages", Json::array({{{"role", "user"}, {"content", "Hi"}}})}};
    const Response ra = request(srv.port(), "POST", "/v1/messages", ingine::serve::dump(abody));
    CHECK(ra.status == 200);
    CHECK(Json::parse(ra.body)["usage"]["cache_read_input_tokens"] == 0);
    srv.stop();
  }

  // 2. Warm-start enabled (single trusted scope): /health reports warm_start=true + tokens, Turn-1 hits across
  //    default effort, explicit reasoning_effort="medium", and enable_thinking=false; mismatched system misses;
  //    Authorization header still fails closed to cached_tokens=0.
  {
    FakeGenerator gen(g_fmt->tokenizer(), "inkling-ws-on", 1 << 20);
    ingine::serve::ServerOptions opts;
    opts.model_id = "inkling-ws-on";
    opts.warm_start_prompt = sys_prompt;
    ingine::serve::ChatServer srv(gen, *g_fmt, opts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    CHECK(gen.warm_start_calls().size() == 1);
    const Json h = health(srv.port());
    CHECK(h["warm_start"] == true);
    const int ws_toks = h["warm_start_tokens"].get<int>();
    CHECK(ws_toks > 0 && ws_toks == int(gen.warm_start_calls().front().prompt_ids.size()));

    // Turn-1 hit (default effort)
    const Json b1 = {
        {"model", "inkling-ws-on"},
        {"messages", Json::array({{{"role", "system"}, {"content", sys_prompt}}, {{"role", "user"}, {"content", "Explain RAII."}}})},
        {"max_tokens", 16}};
    const Response r1 = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b1));
    CHECK(r1.status == 200);
    CHECK(Json::parse(r1.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == ws_toks);

    // Turn-1 hit with different reasoning_effort and enable_thinking=false
    Json b_med = b1;
    b_med["reasoning_effort"] = "medium";
    const Response r_med = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b_med));
    CHECK(r_med.status == 200);
    CHECK(Json::parse(r_med.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == ws_toks);

    Json b_nt = b1;
    b_nt["chat_template_kwargs"] = {{"enable_thinking", false}};
    const Response r_nt = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b_nt));
    CHECK(r_nt.status == 200);
    CHECK(Json::parse(r_nt.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == ws_toks);

    // Anthropic /v1/messages Turn-1 with matching system prompt also hits the warmed prefix cache.
    const Json ab1 = {{"model", "inkling-ws-on"},
                      {"max_tokens", 16},
                      {"system", sys_prompt},
                      {"messages", Json::array({{{"role", "user"}, {"content", "Explain RAII."}}})}};
    const Response ra1 = request(srv.port(), "POST", "/v1/messages", ingine::serve::dump(ab1));
    CHECK(ra1.status == 200);
    CHECK(Json::parse(ra1.body)["usage"]["cache_read_input_tokens"] == ws_toks);

    // Different system prompt misses
    const Json b_diff = {
        {"model", "inkling-ws-on"},
        {"messages", Json::array({{{"role", "system"}, {"content", "Different persona."}}, {{"role", "user"}, {"content", "Hi"}}})},
        {"max_tokens", 16}};
    const Response r_diff = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b_diff));
    CHECK(r_diff.status == 200);
    CHECK(Json::parse(r_diff.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);

    // Fail-closed Authorization isolation still holds on warm-started server
    const Response r_auth = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b1),
                                    {{"Authorization", "Bearer tenant-token"}});
    CHECK(r_auth.status == 200);
    CHECK(Json::parse(r_auth.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);
    srv.stop();
  }

  // 3. Namespace-scoped warm-start: hits matching tenant (even with Authorization), misses other tenant / missing header.
  {
    FakeGenerator gen(g_fmt->tokenizer(), "inkling-ws-ns", 1 << 20);
    ingine::serve::ServerOptions opts;
    opts.model_id = "inkling-ws-ns";
    opts.cache_namespace_header = "X-Tenant";
    opts.warm_start_prompt = sys_prompt;
    opts.warm_start_namespace = "team-a";
    ingine::serve::ChatServer srv(gen, *g_fmt, opts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    const int ws_toks = health(srv.port())["warm_start_tokens"].get<int>();
    CHECK(health(srv.port())["warm_start"] == true && ws_toks > 0);
    const Json b = {
        {"model", "inkling-ws-ns"},
        {"messages", Json::array({{{"role", "system"}, {"content", sys_prompt}}, {{"role", "user"}, {"content", "Hi"}}})},
        {"max_tokens", 16}};
    const Response r_hit = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b),
                                   {{"X-Tenant", "team-a"}, {"Authorization", "Bearer none"}});
    CHECK(r_hit.status == 200);
    CHECK(Json::parse(r_hit.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == ws_toks);

    const Response r_other = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b),
                                     {{"X-Tenant", "team-b"}});
    CHECK(r_other.status == 200);
    CHECK(Json::parse(r_other.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);

    const Response r_missing = request(srv.port(), "POST", "/v1/chat/completions", ingine::serve::dump(b));
    CHECK(r_missing.status == 200);
    CHECK(Json::parse(r_missing.body)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0);
    srv.stop();
  }

  // 4. Fail-closed invalid warm-start configurations throw in ChatServer constructor.
  auto expect_throws = [&](ingine::serve::ServerOptions bad, int max_len = 1 << 20) {
    FakeGenerator gen(g_fmt->tokenizer(), "inkling-ws-bad", max_len);
    bool threw = false;
    try {
      ingine::serve::ChatServer srv(gen, *g_fmt, std::move(bad));
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  };
  {
    ingine::serve::ServerOptions o;
    o.warm_start_prompt = "   \n ";
    expect_throws(o);
  }
  {
    ingine::serve::ServerOptions o;
    o.cache_namespace_header = "X-Tenant";
    o.warm_start_prompt = sys_prompt;
    expect_throws(o);  // missing warm_start_namespace
  }
  {
    ingine::serve::ServerOptions o;
    o.warm_start_prompt = sys_prompt;
    o.warm_start_namespace = "team-a";
    expect_throws(o);  // warm_start_namespace without cache_namespace_header
  }
  {
    ingine::serve::ServerOptions o;
    o.cache_namespace_header = "X-Tenant";
    o.warm_start_namespace = "team-a";
    expect_throws(o);  // warm_start_namespace without warm_start_prompt
  }
  {
    ingine::serve::ServerOptions o;
    o.warm_start_prompt = sys_prompt;
    expect_throws(o, /*max_len=*/4);  // exceeds max_model_len
  }

  // 5. Ordinary Generator subclass with no prefix cache (inherits default Generator::warm_start):
  //    must fail closed when warm_start_prompt is configured and never advertise warm_start=true,
  //    while continuing to serve cleanly with warm_start=false when unset.
  {
    class NoCacheCpuGenerator final : public ingine::model::Generator {
     public:
      explicit NoCacheCpuGenerator(const ingine::text::Tokenizer& tok) : tok_(tok) {}
      std::string model_id() const override { return "nocache-cpu-test"; }
      int max_model_len() const override { return 4096; }
      ingine::model::GenerateStats generate(
          std::span<const int32_t> prompt, int max_new_tokens, const ingine::model::Sampling&,
          std::span<const int32_t>,
          const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
        const std::vector<int32_t> out = tok_.encode(
            "<|message_model|><|content_text|>ok<|end_message|><|content_model_end_sampling|>", /*special=*/true);
        const std::size_t lim = std::min(out.size(), std::size_t(std::max(max_new_tokens, 0)));
        for (std::size_t i = 0; i < lim; ++i) {
          if (!on_tokens(std::span<const int32_t>(&out[i], 1))) break;
        }
        ingine::model::GenerateStats st;
        st.prompt_tokens = int(prompt.size());
        st.generated_tokens = int(lim);
        st.reuse = ingine::model::Reuse::None;
        st.reused_tokens = 0;
        return st;
      }

     private:
      const ingine::text::Tokenizer& tok_;
    };

    NoCacheCpuGenerator nc_gen(g_fmt->tokenizer());
    {
      ingine::serve::ServerOptions o;
      o.model_id = "nocache-cpu-test";
      o.warm_start_prompt = sys_prompt;
      bool threw = false;
      try {
        ingine::serve::ChatServer srv(nc_gen, *g_fmt, std::move(o));
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      CHECK(threw);
    }
    {
      ingine::serve::ServerOptions o;
      o.model_id = "nocache-cpu-test";
      ingine::serve::ChatServer srv(nc_gen, *g_fmt, std::move(o));
      CHECK(srv.bind("127.0.0.1", 0));
      srv.start();
      const Json h = health(srv.port());
      CHECK(h["warm_start"] == false && h["warm_start_tokens"] == 0);
      srv.stop();
    }
  }
}

int main() {
  const char* env = std::getenv("INGINE_MODEL_DIR");
  const std::filesystem::path model_dir = env && *env ? env : INGINE_MODEL_DIR;
  if (!std::filesystem::exists(model_dir / "tokenizer.json") || !std::filesystem::exists(model_dir / "chat_template.jinja")) {
    std::printf("skip: no tokenizer/chat template in %s\n", model_dir.c_str());
    return 77;
  }
  const ChatFormat fmt = ChatFormat::load(model_dir);
  g_fmt = &fmt;
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"models_and_health", test_models_and_health},
      {"stats", test_stats},
      {"non_streaming", test_non_streaming},
      {"streaming_matches_non_streaming[False]", [] { test_streaming_matches_non_streaming(false); }},
      {"streaming_matches_non_streaming[True]", [] { test_streaming_matches_non_streaming(true); }},
      {"tool_call_streaming", test_tool_call_streaming},
      {"tool_round_trip_prompt", test_tool_round_trip_prompt},
      {"thinking_controls", test_thinking_controls},
      {"sampling_and_max_tokens", test_sampling_and_max_tokens},
      {"sampling_defaults", test_sampling_defaults},
      {"context_limit", test_context_limit},
      {"stop_strings_cancel_generation[False]", [] { test_stop_strings_cancel_generation(false); }},
      {"stop_strings_cancel_generation[True]", [] { test_stop_strings_cancel_generation(true); }},
      {"disconnect_cancels_generation[False]", [] { test_disconnect_cancels_generation(false); }},
      {"disconnect_cancels_generation[True]", [] { test_disconnect_cancels_generation(true); }},
      {"queue_full_503_and_withdraw", test_queue_full_503_and_withdraw},
      {"fifo_order", test_fifo_order},
      {"generator_failure", test_generator_failure},
      {"bad_requests", test_bad_requests},
      {"invalid_json", test_invalid_json},
      {"echo", test_echo},
      {"images", test_images},
      {"response_format_and_tool_ban", test_response_format_and_tool_ban},
      {"persona_control_plane", test_persona_control_plane},
      {"cached_tokens_accounting", test_cached_tokens_accounting},
      {"legacy_generator_early_stop_cached_tokens", test_legacy_generator_early_stop_cached_tokens},
      {"prompt_stability_and_render_stats", test_prompt_stability_and_render_stats},
      {"cache_scope_isolation", test_cache_scope_isolation},
      {"kv_cache_mode_advertisement", test_kv_cache_mode_advertisement},
      {"anthropic_messages_api", test_anthropic_messages_api},
      {"loopcd_advertisement", test_loopcd_advertisement},
      {"warm_start_system_prompt", test_warm_start_system_prompt},
  };
  int failed = 0;
  for (const auto& [name, fn] : tests) {
    const auto t0 = std::chrono::steady_clock::now();
    try {
      fn();
      std::printf("PASS %s (%.2fs)\n", name.c_str(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("FAIL %s: %s\n", name.c_str(), e.what());
    }
    std::fflush(stdout);
  }
  std::printf("%d/%zu passed\n", int(tests.size()) - failed, tests.size());
  return failed ? 1 : 0;
}
