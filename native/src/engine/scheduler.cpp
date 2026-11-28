#include "engine/scheduler.h"

#include "engine/log.h"

#include <cxxabi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <span>
#include <typeinfo>
#include <utility>

namespace ingine::engine {
namespace {

// uuid.uuid4().hex: 32 random hex digits.
std::string random_hex32() {
  thread_local std::mt19937_64 rng{std::random_device{}() ^ (uint64_t(std::random_device{}()) << 32)};
  char buf[33];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return buf;
}

std::string type_name(const std::exception& e) {
  int status = 0;
  char* name = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
  std::string out = status == 0 && name ? name : typeid(e).name();
  std::free(name);
  return out;
}

}  // namespace

Job::Job(std::vector<int32_t> prompt_ids_, int max_new_tokens_, model::Sampling sampling_,
         std::vector<model::PromptImage> images_, int root_prefix_tokens_, std::string cache_namespace_,
         bool bypass_prefix_cache_)
    : id("chatcmpl-" + random_hex32()),
      prompt_ids(std::move(prompt_ids_)),
      images(std::move(images_)),
      root_prefix_tokens(root_prefix_tokens_),
      cache_namespace(std::move(cache_namespace_)),
      bypass_prefix_cache(bypass_prefix_cache_),
      max_new_tokens(max_new_tokens_),
      sampling(sampling_),
      t_submit(Clock::now()) {}

std::optional<Clock::time_point> Job::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

std::optional<model::GenerateStats> Job::prefill_stats() const {
  std::lock_guard lock(mutex_);
  return prefill_stats_;
}

std::optional<model::GenerateStats> Job::wait_prefill_stats() const {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return prefill_stats_.has_value() || finished_ || failure_.has_value(); });
  return prefill_stats_;
}

JobBatch Job::take(Clock::duration timeout) {
  std::unique_lock lock(mutex_);
  cv_.wait_for(lock, timeout, [&] { return !tokens_.empty() || failure_ || end_; });
  JobBatch batch;
  batch.tokens.assign(tokens_.begin(), tokens_.end());
  tokens_.clear();
  // The failure and the end are reported once, after every token pushed before them.
  batch.failure = std::exchange(failure_, std::nullopt);
  batch.end = std::exchange(end_, false);
  return batch;
}

void Job::started() {
  std::lock_guard lock(mutex_);
  t_start_ = Clock::now();
}

void Job::set_prefill_stats(model::GenerateStats stats) {
  {
    std::lock_guard lock(mutex_);
    prefill_stats_ = std::move(stats);
  }
  cv_.notify_all();
}

void Job::push(const std::vector<int32_t>& tokens) {
  if (tokens.empty()) return;
  {
    std::lock_guard lock(mutex_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  }
  cv_.notify_all();
}

void Job::fail(std::string message) {
  {
    std::lock_guard lock(mutex_);
    failure_ = std::move(message);
  }
  cv_.notify_all();
}

void Job::finish() {
  {
    std::lock_guard lock(mutex_);
    end_ = true;
    finished_ = true;
  }
  cv_.notify_all();
}

EmbeddingJob::EmbeddingJob(model::EmbeddingRequest request_)
    : id("embd-" + random_hex32()), request(std::move(request_)), t_submit(Clock::now()) {}

std::optional<Clock::time_point> EmbeddingJob::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

bool EmbeddingJob::wait(Clock::duration timeout) const {
  std::unique_lock lock(mutex_);
  return cv_.wait_for(lock, timeout, [&] { return done_; });
}

bool EmbeddingJob::done() const {
  std::lock_guard lock(mutex_);
  return done_;
}

void EmbeddingJob::started() {
  std::lock_guard lock(mutex_);
  t_start_ = Clock::now();
}

void EmbeddingJob::complete(model::EmbeddingResult result) {
  {
    std::lock_guard lock(mutex_);
    result_ = std::move(result);
    done_ = true;
  }
  cv_.notify_all();
}

void EmbeddingJob::fail(std::string message) {
  {
    std::lock_guard lock(mutex_);
    failure_ = std::move(message);
    done_ = true;
  }
  cv_.notify_all();
}

Scheduler::Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue)
    : generator_(generator),
      stop_list_(std::move(stop_ids)),
      stop_ids_(stop_list_.begin(), stop_list_.end()),
      max_queue_(max_queue),
      max_batch_(std::max(1, generator.max_batch_size())),
      thread_([this] {
        if (max_batch_ > 1) loop_batched();
        else loop();
      }) {}

Scheduler::~Scheduler() { close(); }

bool Scheduler::submit(std::shared_ptr<Job> job) {
  {
    std::lock_guard lock(mutex_);
    const int cap = max_batch_ + max_queue_;
    if (closed_ || int(waiting_.size()) + int(active_.size()) >= cap) return false;
    waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void Scheduler::withdraw(const std::shared_ptr<Job>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(waiting_.begin(), waiting_.end(), job);
  if (it != waiting_.end()) waiting_.erase(it);
}

bool Scheduler::submit_embedding(std::shared_ptr<EmbeddingJob> job) {
  {
    std::lock_guard lock(mutex_);
    const int cap = std::max(1, max_queue_ + 1);
    if (closed_ || int(embed_waiting_.size()) + (embed_running_ ? 1 : 0) >= cap) return false;
    embed_waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void Scheduler::withdraw_embedding(const std::shared_ptr<EmbeddingJob>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(embed_waiting_.begin(), embed_waiting_.end(), job);
  if (it != embed_waiting_.end()) embed_waiting_.erase(it);
}

SchedulerStats Scheduler::stats() const {
  std::lock_guard lock(mutex_);
  const int act = int(active_.size());
  return {int(waiting_.size()), act > 0 || embed_running_, act, max_batch_, max_queue_,
          int(embed_waiting_.size()), embed_running_};
}

void Scheduler::close() {
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& job : waiting_) {
      job->cancel();
      job->fail("server shutting down");
      job->finish();
    }
    waiting_.clear();
    for (auto& job : active_) job->cancel();
    for (auto& ejob : embed_waiting_) {
      ejob->cancel();
      ejob->fail("server shutting down");
    }
    embed_waiting_.clear();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Scheduler::loop() {
  bool prefer_embed = false;
  for (;;) {
    std::shared_ptr<Job> chat_job;
    std::shared_ptr<EmbeddingJob> embed_job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || !embed_waiting_.empty() || closed_; });
      if (closed_) return;
      // Fair alternation whenever both chat and embedding queues have waiting work so neither starves.
      const bool take_embed = !embed_waiting_.empty() && (waiting_.empty() || prefer_embed);
      if (take_embed) {
        embed_job = embed_waiting_.front();
        embed_waiting_.pop_front();
        embed_running_ = true;
        prefer_embed = false;
      } else {
        chat_job = waiting_.front();
        waiting_.pop_front();
        active_.push_back(chat_job);
        prefer_embed = true;
      }
    }
    if (embed_job) {
      run_embedding(*embed_job);
      std::lock_guard lock(mutex_);
      embed_running_ = false;
    } else {
      run(*chat_job);
      chat_job->finish();
      std::lock_guard lock(mutex_);
      active_.clear();
    }
  }
}

void Scheduler::loop_batched() {
  struct SlotEntry {
    std::shared_ptr<Job> job;
    int slot = -1;
    int32_t last_token = 0;
    int generated = 0;
    int max_new_tokens = 0;
  };
  std::vector<SlotEntry> live;
  live.reserve(std::size_t(max_batch_));
  {
    std::lock_guard lock(mutex_);
    active_.reserve(std::size_t(max_batch_));
  }

  auto remove_active_job = [&](const std::shared_ptr<Job>& job) {
    std::lock_guard lock(mutex_);
    auto it = std::find(active_.begin(), active_.end(), job);
    if (it != active_.end()) active_.erase(it);
  };

  auto retire = [&](SlotEntry& e) {
    if (e.slot >= 0) {
      generator_.release_slot(e.slot);
      e.slot = -1;
    }
    e.job->finish();
    remove_active_job(e.job);
  };

  for (;;) {
    // 1. Check shutdown & retire any cancelled active slots immediately.
    bool shutting_down = false;
    {
      std::lock_guard lock(mutex_);
      shutting_down = closed_;
    }
    if (shutting_down) {
      for (auto& e : live) retire(e);
      live.clear();
      return;
    }
    for (auto it = live.begin(); it != live.end();) {
      if (it->job->cancelled()) {
        retire(*it);
        it = live.erase(it);
      } else {
        ++it;
      }
    }

    // 2. Admit newly arrived waiting jobs into free slots (new requests join mid-generation).
    while (int(live.size()) < max_batch_) {
      std::shared_ptr<Job> candidate;
      bool need_admit_check = false;
      {
        std::lock_guard lock(mutex_);
        if (closed_ || waiting_.empty()) break;
        const auto& front = waiting_.front();
        if (front->cancelled()) {
          candidate = front;
          waiting_.pop_front();
        } else if (!live.empty()) {
          candidate = front;
          need_admit_check = true;
        } else {
          candidate = front;
          active_.push_back(candidate);
          waiting_.pop_front();
        }
      }
      if (!need_admit_check && candidate->cancelled()) {
        candidate->finish();
        continue;
      }
      if (need_admit_check) {
        // Perform prompt-aware admission check OUTSIDE mutex_ and inside a per-job try/catch so any
        // exception fails only `candidate` while active live slots and subsequent waiting jobs continue.
        const int cn = int(candidate->prompt_ids.size());
        int c_eff = candidate->max_new_tokens;
        if (!candidate->sampling.forced.empty()) c_eff = std::min(c_eff, int(candidate->sampling.forced.size()));
        const int max_len = generator_.max_model_len();
        if (cn + c_eff > max_len) c_eff = std::max(0, max_len - cn);
        bool admitted = false;
        std::optional<std::string> admit_err;
        try {
          admitted = generator_.can_admit_slot(
              model::Prompt{candidate->prompt_ids, candidate->images, candidate->root_prefix_tokens,
                            candidate->cache_namespace, candidate->bypass_prefix_cache},
              c_eff, candidate->sampling);
        } catch (const std::exception& e) {
          server_log().error("admission check failed: {}: {}", type_name(e), e.what());
          admit_err = "generation failed: " + type_name(e) + ": " + e.what();
        } catch (...) {
          server_log().error("admission check failed: unknown exception");
          admit_err = "generation failed: unknown exception";
        }
        if (admit_err) {
          bool owned = false;
          {
            std::lock_guard lock(mutex_);
            auto it = std::find(waiting_.begin(), waiting_.end(), candidate);
            if (it != waiting_.end()) {
              waiting_.erase(it);
              owned = true;
            }
          }
          // Only fail/finish `candidate` if we still owned it in waiting_; close() or withdraw()
          // may already have finalized it while can_admit_slot() was running.
          if (owned) {
            candidate->fail(*admit_err);
            candidate->finish();
          }
          continue;
        }
        if (!admitted) break;  // wait until an active slot frees KV pages
        {
          std::lock_guard lock(mutex_);
          if (closed_ || waiting_.empty() || waiting_.front() != candidate) continue;
          if (candidate->cancelled()) {
            waiting_.pop_front();
            candidate->finish();
            continue;
          }
          active_.push_back(candidate);
          waiting_.pop_front();
        }
      }
      std::shared_ptr<Job> job = std::move(candidate);
      job->started();
      const int n = int(job->prompt_ids.size());
      int eff_max = job->max_new_tokens;
      if (!job->sampling.forced.empty()) eff_max = std::min(eff_max, int(job->sampling.forced.size()));
      const int max_len = generator_.max_model_len();
      if (n + eff_max > max_len) eff_max = max_len - n;
      try {
        if (eff_max <= 0) {
          throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                      std::to_string(max_len));
        }
        int32_t first_tok = 0;
        model::GenerateStats st;
        const int slot = generator_.start_slot(
            model::Prompt{job->prompt_ids, job->images, job->root_prefix_tokens, job->cache_namespace,
                          job->bypass_prefix_cache},
            eff_max, job->sampling, first_tok, st);
        if (slot < 0) {
          // Put back at front of queue; retry after next step frees resources.
          std::lock_guard lock(mutex_);
          auto it = std::find(active_.begin(), active_.end(), job);
          if (it != active_.end()) active_.erase(it);
          if (!closed_) waiting_.push_front(job);
          else {
            job->fail("server shutting down");
            job->finish();
          }
          break;
        }
        struct SlotGuard {
          model::Generator& gen;
          int slot;
          void release() { slot = -1; }
          ~SlotGuard() {
            if (slot >= 0) gen.release_slot(slot);
          }
        } slot_guard{generator_, slot};

        job->set_prefill_stats(st);
        const int recomputed = std::max(0, st.prompt_tokens - std::clamp(st.reused_tokens, 0, st.prompt_tokens));
        const std::string restored_str =
            st.restored_bytes ? std::to_string(*st.restored_bytes) : std::string("n/a");
        server_log().info("{}: slot={} reuse={} reused={} recomputed={} restored_bytes={} prefill={:.3f}s", job->id,
                          slot, model::to_string(st.reuse), st.reused_tokens, recomputed, restored_str,
                          st.prefill_seconds);
        if (!job->images.empty()) {
          server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", job->id, job->images.size(),
                            st.image_tokens, st.images_encoded, st.vision_seconds);
        }
        if (job->cancelled()) {
          job->finish();
          remove_active_job(job);
          continue;
        }
        job->push({first_tok});
        if (stop_ids_.contains(first_tok) || eff_max <= 1 || job->cancelled()) {
          job->finish();
          remove_active_job(job);
        } else {
          live.push_back(SlotEntry{std::move(job), slot, first_tok, 1, eff_max});
          slot_guard.release();
        }
      } catch (const std::exception& e) {
        server_log().error("generator failed: {}: {}", type_name(e), e.what());
        job->fail("generation failed: " + type_name(e) + ": " + e.what());
        job->finish();
        remove_active_job(job);
      } catch (...) {
        server_log().error("generator failed: unknown exception");
        job->fail("generation failed: unknown exception");
        job->finish();
        remove_active_job(job);
      }
    }

    // 3. Interleave at most ONE queued zero-KV embedding batch per iteration so active chat slots stay
    // parked untouched in paged KV and always advance at least one batched decode step before another
    // embedding batch runs (prevents starving either chat or embedding traffic).
    {
      std::shared_ptr<EmbeddingJob> embed_job;
      {
        std::lock_guard lock(mutex_);
        if (!closed_ && !embed_waiting_.empty()) {
          embed_job = embed_waiting_.front();
          embed_waiting_.pop_front();
          embed_running_ = true;
        }
      }
      if (embed_job) {
        run_embedding(*embed_job);
        std::lock_guard lock(mutex_);
        embed_running_ = false;
      }
    }

    // 4. Sleep if no active sequences and nothing waiting in either queue.
    if (live.empty()) {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || !embed_waiting_.empty() || closed_; });
      continue;
    }

    // 5. Execute ONE batched decode (or batched speculative verify) step across all live slots and stream immediately.
    std::vector<model::BatchSlotStep> inputs(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) inputs[i] = {live[i].slot, live[i].last_token};
    const bool use_multi = generator_.supports_batched_spec();
    std::vector<model::BatchSlotResult> multi_out;
    std::vector<int32_t> next_tokens;

    try {
      if (use_multi) {
        multi_out.resize(live.size());
        generator_.step_slots_multi(inputs, multi_out);
      } else {
        next_tokens.assign(live.size(), 0);
        generator_.step_slots(inputs, next_tokens);
      }
    } catch (const std::exception& e) {
      server_log().error("batched step failed: {}: {}", type_name(e), e.what());
      const std::string msg = "generation failed: " + type_name(e) + ": " + e.what();
      for (auto& entry : live) {
        entry.job->fail(msg);
        retire(entry);
      }
      live.clear();
      continue;
    } catch (...) {
      server_log().error("batched step failed: unknown exception");
      for (auto& entry : live) {
        entry.job->fail("generation failed: unknown exception");
        retire(entry);
      }
      live.clear();
      continue;
    }

    std::vector<SlotEntry> kept;
    kept.reserve(live.size());
    std::vector<int32_t> emit_buf;
    emit_buf.reserve(model::BatchSlotResult::kMaxTokens);
    for (std::size_t i = 0; i < live.size(); ++i) {
      SlotEntry& e = live[i];
      if (e.job->cancelled()) {
        retire(e);
        continue;
      }
      if (use_multi) {
        const model::BatchSlotResult& res = multi_out[i];
        emit_buf.clear();
        bool done = false;
        for (int k = 0; k < res.count; ++k) {
          if (e.job->cancelled()) {
            done = true;
            break;
          }
          const int32_t tok = res.tokens[k];
          emit_buf.push_back(tok);
          ++e.generated;
          if (stop_ids_.contains(tok) || e.generated >= e.max_new_tokens || e.job->cancelled()) {
            done = true;
            break;
          }
        }
        if (!emit_buf.empty()) {
          e.job->push(emit_buf);
          e.last_token = emit_buf.back();
        }
        if (done || emit_buf.empty() || e.job->cancelled()) {
          retire(e);
        } else {
          kept.push_back(std::move(e));
        }
      } else {
        const int32_t tok = next_tokens[i];
        e.job->push({tok});
        e.last_token = tok;
        ++e.generated;
        if (stop_ids_.contains(tok) || e.generated >= e.max_new_tokens || e.job->cancelled()) {
          retire(e);
        } else {
          kept.push_back(std::move(e));
        }
      }
    }
    live = std::move(kept);
  }
}

void Scheduler::run(Job& job) {
  if (job.cancelled()) return;
  job.started();
  int n = 0;
  std::vector<int32_t> out;
  const auto on_tokens = [&](std::span<const int32_t> tokens) -> bool {
    out.clear();
    bool more = true;
    for (int32_t tok : tokens) {
      if (job.cancelled()) {
        more = false;
        break;
      }
      out.push_back(tok);
      ++n;
      if (stop_ids_.contains(tok) || n >= job.max_new_tokens || job.cancelled()) {
        more = false;
        break;
      }
    }
    job.push(out);
    return more;
  };
  try {
    const auto on_prefill = [&](const model::GenerateStats& pst) { job.set_prefill_stats(pst); };
    const model::GenerateStats st = generator_.generate(
        model::Prompt{job.prompt_ids, job.images, job.root_prefix_tokens, job.cache_namespace,
                      job.bypass_prefix_cache},
        job.max_new_tokens, job.sampling, stop_list_, on_tokens, on_prefill);
    job.set_prefill_stats(st);
    // prefix reuse of the prompt: the path, the tokens reused and recomputed, restored bytes, prefill time
    const int recomputed = std::max(0, st.prompt_tokens - std::clamp(st.reused_tokens, 0, st.prompt_tokens));
    const std::string restored_str =
        st.restored_bytes ? std::to_string(*st.restored_bytes) : std::string("n/a");
    server_log().info("{}: reuse={} reused={} recomputed={} restored_bytes={} prefill={:.3f}s", job.id,
                      model::to_string(st.reuse), st.reused_tokens, recomputed, restored_str,
                      st.prefill_seconds);
    if (!job.images.empty()) {  // images: all of the prompt, those the vision tower encoded now, its time
      server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", job.id, job.images.size(),
                        st.image_tokens, st.images_encoded, st.vision_seconds);
    }
  } catch (const std::exception& e) {  // report to the client instead of killing the worker
    server_log().error("generator failed: {}: {}", type_name(e), e.what());
    job.fail("generation failed: " + type_name(e) + ": " + e.what());
  } catch (...) {
    server_log().error("generator failed: unknown exception");
    job.fail("generation failed: unknown exception");
  }
}

void Scheduler::run_embedding(EmbeddingJob& job) {
  if (job.cancelled()) {
    job.fail("embedding request cancelled");
    return;
  }
  job.started();
  try {
    model::EmbeddingResult res = generator_.embed(job.request);
    job.complete(std::move(res));
  } catch (const std::exception& e) {
    server_log().error("embedding failed: {}: {}", type_name(e), e.what());
    job.fail("embedding failed: " + type_name(e) + ": " + e.what());
  } catch (...) {
    server_log().error("embedding failed: unknown exception");
    job.fail("embedding failed: unknown exception");
  }
}

}  // namespace ingine::engine
