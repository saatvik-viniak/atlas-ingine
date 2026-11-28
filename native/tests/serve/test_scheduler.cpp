// Scheduler unit tests with a generator that emits several tokens per callback (like a speculative round):
// the scheduler must cut a batch at a stop id, at max_new_tokens and on cancellation, run jobs in FIFO order,
// bound the waiting list and report generator failures.
#include "engine/scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::GenerateStats;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

// Emits tokens 100, 101, ... in rounds of `round` tokens until on_tokens refuses; throws if prompt[0] < 0.
class RoundGenerator final : public ingine::model::Generator {
 public:
  explicit RoundGenerator(int round, std::chrono::milliseconds pause = {}) : round_(round), pause_(pause) {}
  std::string model_id() const override { return "round"; }
  int max_model_len() const override { return 1 << 20; }
  GenerateStats generate(std::span<const int32_t> prompt, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    if (!prompt.empty() && prompt[0] < 0) throw std::invalid_argument("prompt does not fit");
    order.push_back(prompt.empty() ? -1 : prompt[0]);
    for (int32_t next = 100;; next += round_) {
      std::vector<int32_t> toks;
      for (int i = 0; i < round_; ++i) toks.push_back(next + i);
      if (pause_.count()) std::this_thread::sleep_for(pause_);
      if (!on_tokens(toks)) break;
    }
    return {};
  }
  std::vector<int32_t> order;

 private:
  int round_;
  std::chrono::milliseconds pause_;
};

std::vector<int32_t> drain(Job& job, std::optional<std::string>* failure = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(5));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && failure) *failure = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout");
  }
}

void test_cuts() {
  RoundGenerator gen(4);
  Scheduler s(gen, {105}, 4);
  auto stop = std::make_shared<Job>(std::vector<int32_t>{1}, 100, Sampling{});
  CHECK(s.submit(stop));
  CHECK((drain(*stop) == std::vector<int32_t>{100, 101, 102, 103, 104, 105}));  // stop id mid-round
  auto len = std::make_shared<Job>(std::vector<int32_t>{2}, 3, Sampling{});
  CHECK(s.submit(len));
  CHECK((drain(*len) == std::vector<int32_t>{100, 101, 102}));  // max_new_tokens mid-round
  CHECK(len->t_start().has_value());
}

void test_cancel_fifo_bound() {
  RoundGenerator gen(2, std::chrono::milliseconds(5));
  Scheduler s(gen, {}, 1);
  auto a = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto b = std::make_shared<Job>(std::vector<int32_t>{2}, 2, Sampling{});
  auto c = std::make_shared<Job>(std::vector<int32_t>{3}, 2, Sampling{});
  CHECK(s.submit(a));
  while (!s.stats().running) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(s.submit(b));
  CHECK(!s.submit(c));  // running + 1 waiting > max_queue 1
  CHECK(s.stats().waiting == 1 && s.stats().max_queue == 1);
  s.withdraw(b);  // never runs
  CHECK(s.stats().waiting == 0);
  CHECK(s.submit(c));
  a->cancel();
  CHECK(drain(*a).size() < 1000);
  CHECK(drain(*c).size() == 2);
  CHECK((gen.order == std::vector<int32_t>{1, 3}));
}

void test_failure_and_close() {
  RoundGenerator gen(1, std::chrono::milliseconds(2));
  Scheduler s(gen, {}, 4);
  auto bad = std::make_shared<Job>(std::vector<int32_t>{-1}, 5, Sampling{});
  CHECK(s.submit(bad));
  std::optional<std::string> failure;
  CHECK(drain(*bad, &failure).empty());
  CHECK(failure && failure->find("generation failed: std::invalid_argument: prompt does not fit") == 0);
  auto running = std::make_shared<Job>(std::vector<int32_t>{1}, 1000000, Sampling{});
  auto waiting = std::make_shared<Job>(std::vector<int32_t>{2}, 5, Sampling{});
  CHECK(s.submit(running));
  CHECK(s.submit(waiting));
  s.close();
  failure.reset();
  CHECK(drain(*waiting, &failure).empty() && failure == "server shutting down");
  CHECK(drain(*running).size() < 1000000);
  CHECK(!s.submit(std::make_shared<Job>(std::vector<int32_t>{3}, 1, Sampling{})));
}

class PrefillHookGenerator final : public ingine::model::Generator {
 public:
  std::string model_id() const override { return "prefill-hook"; }
  int max_model_len() const override { return 1 << 20; }
  GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const int32_t> stop_ids,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    return generate(ingine::model::Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens,
                    ingine::model::PrefillCallback{});
  }
  GenerateStats generate(const ingine::model::Prompt& prompt, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens,
                         const ingine::model::PrefillCallback& on_prefill) override {
    GenerateStats st;
    st.prompt_tokens = int(prompt.ids.size());
    st.reused_tokens = std::max(0, st.prompt_tokens - 1);
    st.reuse = ingine::model::Reuse::Checkpoint;
    st.restored_bytes = 12345;
    if (on_prefill) on_prefill(st);
    for (int32_t tok = 200; tok < 210; ++tok) {
      const int32_t out[1] = {tok};
      if (!on_tokens(out)) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return st;
  }
};

void test_prefill_stats_before_first_token() {
  PrefillHookGenerator gen;
  Scheduler s(gen, {}, 4);
  auto j = std::make_shared<Job>(std::vector<int32_t>{10, 20, 30, 40}, 10, Sampling{});
  CHECK(s.submit(j));
  // Receive the very first token and cancel immediately without waiting for generate() to return.
  JobBatch b = j->take(std::chrono::seconds(5));
  CHECK(!b.tokens.empty() && b.tokens[0] == 200);
  j->cancel();
  const auto st = j->prefill_stats();
  CHECK(st.has_value());
  CHECK(st->prompt_tokens == 4 && st->reused_tokens == 3 && st->reuse == ingine::model::Reuse::Checkpoint &&
        st->restored_bytes.has_value() && *st->restored_bytes == 12345);
  drain(*j);
}

class EmbedFairnessGenerator final : public ingine::model::Generator {
 public:
  std::string model_id() const override { return "embed-fair"; }
  int max_model_len() const override { return 1 << 20; }
  bool supports_embeddings() const override { return true; }
  int embedding_dim() const override { return 4; }
  int num_tap_layers() const override { return 8; }
  GenerateStats generate(std::span<const int32_t> prompt, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    events.push_back(prompt[0]);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const int32_t tok = 42;
    on_tokens(std::span<const int32_t>(&tok, 1));
    return {};
  }
  ingine::model::EmbeddingResult embed(const ingine::model::EmbeddingRequest& req) override {
    events.push_back(-req.inputs[0][0]);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ingine::model::EmbeddingResult res;
    res.embeddings.push_back({1.0f, 0.0f, 0.0f, 0.0f});
    res.stats.total_prompt_tokens = int(req.inputs[0].size());
    res.stats.output_dim = 4;
    res.stats.kv_slots_allocated = 0;
    res.stats.lm_head_skipped = true;
    return res;
  }
  std::vector<int32_t> events;
};

void test_embedding_fair_interleaving_and_close() {
  EmbedFairnessGenerator gen;
  Scheduler s(gen, {}, 4);
  auto c1 = std::make_shared<Job>(std::vector<int32_t>{1}, 1, Sampling{});
  auto c2 = std::make_shared<Job>(std::vector<int32_t>{2}, 1, Sampling{});
  ingine::model::EmbeddingRequest er1;
  er1.inputs = {{10}};
  ingine::model::EmbeddingRequest er2;
  er2.inputs = {{20}};
  auto e1 = std::make_shared<ingine::engine::EmbeddingJob>(er1);
  auto e2 = std::make_shared<ingine::engine::EmbeddingJob>(er2);
  CHECK(s.submit(c1));
  CHECK(s.submit(c2));
  CHECK(s.submit_embedding(e1));
  CHECK(s.submit_embedding(e2));
  CHECK(drain(*c1).size() == 1);
  CHECK(drain(*c2).size() == 1);
  CHECK(e1->wait(std::chrono::seconds(5)) && e1->result().has_value());
  CHECK(e2->wait(std::chrono::seconds(5)) && e2->result().has_value());
  CHECK(e1->result()->stats.kv_slots_allocated == 0);
  CHECK(e1->result()->stats.lm_head_skipped);
  CHECK(gen.events.size() == 4);
  s.close();
  auto e3 = std::make_shared<ingine::engine::EmbeddingJob>(er1);
  CHECK(!s.submit_embedding(e3));
}

}  // namespace

int main() {
  test_cuts();
  test_cancel_fifo_bound();
  test_failure_and_close();
  test_prefill_stats_before_first_token();
  test_embedding_fair_interleaving_and_close();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "ok\n", g_failures);
  return g_failures ? 1 : 0;
}
