// Request scheduler (mirror of engine/ingine/server.py Scheduler/Job): one worker thread runs jobs strictly one at
// a time in FIFO order from a bounded waiting list. The worker only pulls token ids from Generator::generate and
// pushes them to the job's output; detokenisation, parsing and HTTP writes happen on the request thread, so the
// decode loop never waits on the client. Cancellation (client gone, stop string, shutdown) is checked between
// the tokens the generator emits.
#pragma once

#include "ingine/generator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace ingine::engine {

using Clock = std::chrono::steady_clock;

// Output drained from a job: token ids in order, then a failure message and/or the end of generation.
struct JobBatch {
  std::vector<int32_t> tokens;
  std::optional<std::string> failure;
  bool end = false;
  bool empty() const { return tokens.empty() && !failure && !end; }
};

class Job {
 public:
  Job(std::vector<int32_t> prompt_ids, int max_new_tokens, model::Sampling sampling,
      std::vector<model::PromptImage> images = {}, int root_prefix_tokens = 0, std::string cache_namespace = {},
      bool bypass_prefix_cache = false);

  const std::string id;  // "chatcmpl-" + 32 hex digits
  const std::vector<int32_t> prompt_ids;
  const std::vector<model::PromptImage> images;  // the prompt's images, by position (see model::Prompt)
  const int root_prefix_tokens;
  const std::string cache_namespace;
  const bool bypass_prefix_cache;
  const int max_new_tokens;
  const model::Sampling sampling;
  const Clock::time_point t_submit;

  void cancel() { cancel_.store(true, std::memory_order_release); }
  bool cancelled() const { return cancel_.load(std::memory_order_acquire); }
  std::optional<Clock::time_point> t_start() const;
  // Completed prefill/cache accounting, published before the first generated token is pushed (modern generators)
  // or upon generate() return (legacy 5-arg Generator subclasses).
  std::optional<model::GenerateStats> prefill_stats() const;
  // Returns prefill_stats() immediately if already published; otherwise waits until stats are published or the
  // job finishes/fails (synchronized compatibility path for legacy 5-arg generators on early stop-string cancel).
  std::optional<model::GenerateStats> wait_prefill_stats() const;

  // Waits up to `timeout` for output, then returns everything available (empty on timeout).
  JobBatch take(Clock::duration timeout);

 private:
  friend class Scheduler;
  void started();
  void set_prefill_stats(model::GenerateStats stats);
  void push(const std::vector<int32_t>& tokens);
  void fail(std::string message);
  void finish();

  std::atomic<bool> cancel_{false};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::deque<int32_t> tokens_;
  std::optional<std::string> failure_;
  bool end_ = false;
  bool finished_ = false;  // monotonic: remains true even after take() consumes end_
  std::optional<Clock::time_point> t_start_;
  std::optional<model::GenerateStats> prefill_stats_;
};

// One queued embedding batch request (zero-KV, no-lm_head prefill path).
class EmbeddingJob {
 public:
  explicit EmbeddingJob(model::EmbeddingRequest request);

  const std::string id;  // "embd-" + 32 hex digits
  const model::EmbeddingRequest request;
  const Clock::time_point t_submit;

  void cancel() { cancel_.store(true, std::memory_order_release); }
  bool cancelled() const { return cancel_.load(std::memory_order_acquire); }
  std::optional<Clock::time_point> t_start() const;

  // Waits up to `timeout` for completion. Returns true once finished (result or failure populated).
  bool wait(Clock::duration timeout) const;
  bool done() const;
  const std::optional<model::EmbeddingResult>& result() const { return result_; }
  const std::optional<std::string>& failure() const { return failure_; }

 private:
  friend class Scheduler;
  void started();
  void complete(model::EmbeddingResult result);
  void fail(std::string message);

  std::atomic<bool> cancel_{false};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  bool done_ = false;
  std::optional<Clock::time_point> t_start_;
  std::optional<model::EmbeddingResult> result_;
  std::optional<std::string> failure_;
};

struct SchedulerStats {
  int waiting = 0;
  bool running = false;
  int active = 0;
  int max_batch = 1;
  int max_queue = 0;
  int embed_waiting = 0;
  bool embed_running = false;
};

class Scheduler {
 public:
  Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue);
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // False when closed or when max_queue jobs already wait (plus active slots).
  bool submit(std::shared_ptr<Job> job);
  // Cancels the job and drops it from the waiting list if it has not started.
  void withdraw(const std::shared_ptr<Job>& job);
  // Submits a zero-KV embedding batch job sharing the GPU fairly with chat traffic.
  bool submit_embedding(std::shared_ptr<EmbeddingJob> job);
  // Cancels the embedding job and drops it from the waiting queue if not yet started.
  void withdraw_embedding(const std::shared_ptr<EmbeddingJob>& job);
  SchedulerStats stats() const;
  // Fails waiting jobs ("server shutting down"), cancels the running ones and joins the worker.
  void close();

 private:
  void loop();
  void loop_batched();
  void run(Job& job);
  void run_embedding(EmbeddingJob& job);

  model::Generator& generator_;
  const std::vector<int32_t> stop_list_;
  const std::unordered_set<int32_t> stop_ids_;
  const int max_queue_;
  const int max_batch_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> waiting_;
  std::vector<std::shared_ptr<Job>> active_;
  std::deque<std::shared_ptr<EmbeddingJob>> embed_waiting_;
  bool embed_running_ = false;
  bool closed_ = false;
  std::thread thread_;
};

}  // namespace ingine::engine
