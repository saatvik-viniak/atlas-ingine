// Continuous-batching runtime and generator (B = 1..64+ concurrent sequences) with paged full-attention KV
// cache (page size 64), per-slot sliding rings + 4-tap conv states, bucketed CUDA-graph decode capture, and
// zero-copy Prefiller slot integration.
#include "model/batch_engine.h"

#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/ops.h"
#include "core/safetensors.h"
#include "ingine/batch_spec_plan.h"
#include "ingine/platos_key.h"
#include "kernels/batch/batch.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/batch_draft.h"
#include "kernels/draft/draft.h"
#include "kernels/head/head.h"
#include "kernels/prefill/prefill.h"
#include "model/batch_drafter.h"
#include "model/batch_prefix_cache.h"
#include "model/batch_spec_trace.h"
#include "model/kv_checkpoint.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/vision.h"
#include "model/weights_gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine::model {
namespace {

// Test-only seam allowing matched-chunk-boundary verification (e.g. cold 63+1+1 vs cached 63->64->65)
// without linker symbol wrapping. Empty in production (zero overhead).
std::mutex g_test_splits_mu;
std::vector<int> g_test_prefill_splits;

std::vector<int> snapshot_test_prefill_splits() {
  std::lock_guard<std::mutex> lock(g_test_splits_mu);
  return g_test_prefill_splits;
}

std::mutex g_spec_trace_mu;
BatchSpecTraceConfig g_spec_trace_cfg;

BatchSpecTraceConfig snapshot_spec_trace_cfg() {
  std::lock_guard<std::mutex> lock(g_spec_trace_mu);
  return g_spec_trace_cfg;
}

namespace bk = kernels::batch;
namespace dense = kernels::dense;
namespace kd = kernels::draft;
namespace head = kernels::head;
namespace pk = kernels::prefill;

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

std::vector<int> build_batch_buckets(int max_batch) {
  static constexpr int kCanonical[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
  std::vector<int> out;
  for (int b : kCanonical) {
    if (b < max_batch) out.push_back(b);
  }
  for (int b = 128; b < max_batch; b <<= 1) out.push_back(b);
  out.push_back(max_batch);
  return out;
}

int bucket_for_batch(int active, int max_batch) {
  static constexpr int kCanonical[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
  for (int b : kCanonical) {
    if (b >= active && b <= max_batch) return b;
  }
  return max_batch;
}

constexpr int H = HIDDEN;
constexpr int kHeadChunk = 16384;  // tiled unscaled FP8/BF16 unembed chunk for M > 32 (16384 * 2304 * 2 = 75.5 MB)

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const ActT* bf16p(const Tensor& t) { return static_cast<const ActT*>(t.data); }
const ActT* bf16p(const void* p) { return static_cast<const ActT*>(p); }

dense::Matrix matrix(const Dense& d) {
  return dense::Matrix{d.w.data, static_cast<const float*>(d.scale.data), d.rows(), d.q5_k(),
                       static_cast<const std::int64_t*>(d.blk_off.data),
                       static_cast<const std::int32_t*>(d.blk_soff.data),
                       static_cast<const std::uint16_t*>(d.q8_scale.data)};
}

std::vector<int> proj_layout(bool full, bool single_input) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(SEG_BEGIN[s]);
    t.push_back(SEG_BEGIN[s + 1]);
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

head::SamplingParams to_head_params(const Sampling& sampling) {
  head::SamplingParams p;
  p.temperature = sampling.temperature;
  p.top_k = sampling.top_k;
  p.top_p = sampling.top_p;
  for (std::int32_t id : sampling.banned_tokens) {
    if (id >= 0 && p.banned.count < head::BannedTokens::kMax) p.banned.ids[p.banned.count++] = id;
  }
  return p;
}

bool same_banned(const head::BannedTokens& a, const head::BannedTokens& b) {
  if (a.count != b.count) return false;
  for (int i = 0; i < a.count; ++i)
    if (a.ids[i] != b.ids[i]) return false;
  return true;
}

// ----------------------------------------------------------------------------- Per-slot state + paged full KV pool

struct BatchStatePool {
  int max_batch = 1;
  int max_ctx = 0;
  bool kv_fp8 = false;
  int max_pages_per_slot = 0;
  int num_pages = 0;

  // Per-slot contiguous slices: slot s starts at `base + s * *_slot_stride`.
  std::size_t ring_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW * KV_DIM;
  std::size_t ring_pos_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW;
  std::size_t conv_kv_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * KV_DIM;
  std::size_t conv_h_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * HIDDEN;

  DeviceBuffer k_ring, v_ring, ring_pos;
  DeviceBuffer conv_k, conv_v, conv_attn, conv_mlp;
  DeviceBuffer slot_pos;  // int64 [max_batch]

  // Shared paged full-attention KV pool: [num_pages, NUM_FULL_VISITS, kPageSize, KV_HEADS, HEAD_DIM]
  DeviceBuffer k_full_pool, v_full_pool;
  DeviceBuffer page_table;  // int32 [max_batch, max_pages_per_slot]

  // Single scratch full-KV buffer for Prefiller::run (reused across slots, scattered immediately after prefill)
  DeviceBuffer scratch_k_full, scratch_v_full;

  // Refcounted host-side allocator for physical full-KV pages (shared committed pages + private tails)
  PagedKvPagePool pages;
  std::vector<std::vector<int>> slot_pages;
  std::vector<int> slot_pos_host;
  std::vector<bool> slot_in_use;

  void init(int max_b, int ctx, bool fp8, int max_kv_tokens, std::size_t extra_reserve_bytes = 0,
            std::optional<std::size_t> vram_budget_bytes = std::nullopt) {
    max_batch = max_b;
    max_ctx = ctx;
    kv_fp8 = fp8;
    max_pages_per_slot = (ctx + bk::kPageSize - 1) / bk::kPageSize;

    // Pre-compute ALL fixed pool buffer sizes BEFORE allocating a single byte so lease caps fail closed!
    const std::size_t es = fp8 ? 1 : 2;
    const std::size_t scratch_bytes = std::size_t(NUM_FULL_VISITS) * ctx * KV_DIM * es;
    const std::size_t fixed_bytes =
        2 * std::size_t(max_b) * ring_slot_stride * sizeof(ActT) +
        std::size_t(max_b) * ring_pos_slot_stride * sizeof(std::int64_t) +
        2 * std::size_t(max_b) * conv_kv_slot_stride * sizeof(float) +
        2 * std::size_t(max_b) * conv_h_slot_stride * sizeof(float) +
        std::size_t(max_b) * sizeof(std::int64_t) +
        2 * scratch_bytes +
        std::size_t(max_b) * max_pages_per_slot * sizeof(std::int32_t);

    // Paged full-KV pool sizing (explicitly reserves extra_reserve_bytes for decoder/logits/rollback/graphs
    // and enforces vram_budget_bytes on BOTH explicit max_kv_tokens and auto-sized branches!).
    const std::size_t page_kv_bytes = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM * es;
    const std::size_t bytes_per_page = 2 * page_kv_bytes;  // K + V
    const int max_possible_pages = max_b * max_pages_per_slot;
    if (max_kv_tokens > 0) {
      num_pages =
          std::min(max_possible_pages, std::max(max_pages_per_slot, (max_kv_tokens + bk::kPageSize - 1) / bk::kPageSize));
      const std::size_t total_planned = fixed_bytes + std::size_t(num_pages) * bytes_per_page + extra_reserve_bytes;
      const VramLeaseCheck chk = check_vram_lease_fit(vram_budget_bytes, total_planned);
      if (!chk.ok) {
        throw std::runtime_error("BatchStatePool: VRAM lease cap exceeded for explicit max_kv_tokens=" +
                                 std::to_string(max_kv_tokens) + " (need " + std::to_string(chk.planned_bytes >> 20) +
                                 " MiB including decoder reserve, remaining " +
                                 std::to_string(chk.remaining_bytes >> 20) + " MiB)");
      }
    } else {
      std::size_t free_bytes = 0, total_bytes = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      if (vram_budget_bytes.has_value()) {
        const std::size_t min_needed = fixed_bytes + extra_reserve_bytes;
        if (*vram_budget_bytes <= min_needed) {
          throw std::runtime_error("BatchStatePool: VRAM lease cap exhausted before paged KV pool allocation (need >= " +
                                   std::to_string(min_needed >> 20) + " MiB, remaining " +
                                   std::to_string(*vram_budget_bytes >> 20) + " MiB)");
        }
        const std::size_t cap_avail = std::min(free_bytes, *vram_budget_bytes) > min_needed
                                          ? (std::min(free_bytes, *vram_budget_bytes) - min_needed)
                                          : 0;
        const int fit_pages = int(cap_avail / bytes_per_page);
        if (fit_pages < max_pages_per_slot) {
          throw std::runtime_error("BatchStatePool: VRAM lease cap cannot fit minimum 1-slot full-KV pool (" +
                                   std::to_string(max_pages_per_slot) + " pages needed, only " +
                                   std::to_string(fit_pages) + " fit in remaining " +
                                   std::to_string(*vram_budget_bytes >> 20) + " MiB)");
        }
        num_pages = std::min(max_possible_pages, fit_pages);
      } else {
        const std::size_t base_reserve = std::min<std::size_t>(std::size_t(2560) << 20, free_bytes / 4);
        const std::size_t reserve = base_reserve + extra_reserve_bytes + fixed_bytes;
        const std::size_t avail = free_bytes > reserve ? free_bytes - reserve : 0;
        const int fit_pages = int(avail / bytes_per_page);
        num_pages = std::min(max_possible_pages, std::max(std::min(max_possible_pages, max_pages_per_slot), fit_pages));
      }
      if (num_pages < 1) throw std::runtime_error("BatchStatePool: insufficient VRAM for paged full-KV pool");
    }

    // Allocate per-slot sliding rings & conv states
    k_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    v_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    ring_pos = DeviceBuffer(std::size_t(max_b) * ring_pos_slot_stride * sizeof(std::int64_t));
    conv_k = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_v = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_attn = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    conv_mlp = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    slot_pos = DeviceBuffer(std::size_t(max_b) * sizeof(std::int64_t));

    k_ring.fill(0);
    v_ring.fill(0);
    ring_pos.fill(0xff);
    conv_k.fill(0);
    conv_v.fill(0);
    conv_attn.fill(0);
    conv_mlp.fill(0);
    slot_pos.fill(0);

    // Scratch full-KV buffer for single-sequence Prefiller
    scratch_k_full = DeviceBuffer(scratch_bytes);
    scratch_v_full = DeviceBuffer(scratch_bytes);
    scratch_k_full.fill(0);
    scratch_v_full.fill(0);

    k_full_pool = DeviceBuffer(std::size_t(num_pages) * page_kv_bytes);
    v_full_pool = DeviceBuffer(std::size_t(num_pages) * page_kv_bytes);
    k_full_pool.fill(0);
    v_full_pool.fill(0);

    page_table = DeviceBuffer(std::size_t(max_b) * max_pages_per_slot * sizeof(std::int32_t));
    page_table.fill(0xff);  // -1

    pages.init(num_pages);
    slot_pages.assign(max_b, {});
    for (auto& sp : slot_pages) sp.reserve(std::size_t(max_pages_per_slot));
    slot_pos_host.assign(max_b, 0);
    slot_in_use.assign(max_b, false);
  }

  // Zero-copy State view pointing directly at slot `s`'s sliding rings & conv states, and shared scratch full KV.
  State prefill_view(int s) const {
    State st;
    st.max_ctx = max_ctx;
    st.kv_fp8 = kv_fp8;
    const DType full_dt = kv_fp8 ? DType::FP8_E4M3FN : kActDType;
    st.k_ring = rm(static_cast<ActT*>(k_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.v_ring = rm(static_cast<ActT*>(v_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.ring_pos = rm(static_cast<std::int64_t*>(ring_pos.p) + std::size_t(s) * ring_pos_slot_stride, DType::I64,
                     {NUM_SLIDING_VISITS, WINDOW});
    st.k_full = rm(scratch_k_full.p, full_dt, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
    st.v_full = rm(scratch_v_full.p, full_dt, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
    st.conv_k = rm(static_cast<float*>(conv_k.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_v = rm(static_cast<float*>(conv_v.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_attn = rm(static_cast<float*>(conv_attn.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                      {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.conv_mlp = rm(static_cast<float*>(conv_mlp.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                     {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.length = 0;
    return st;
  }

  int free_slot_count() const {
    int n = 0;
    for (bool used : slot_in_use)
      if (!used) ++n;
    return n;
  }

  int active_slot_count() const { return max_batch - free_slot_count(); }

  int alloc_slot() {
    for (int s = 0; s < max_batch; ++s) {
      if (!slot_in_use[s]) {
        slot_in_use[s] = true;
        slot_pages[s].clear();
        slot_pos_host[s] = 0;
        return s;
      }
    }
    return -1;
  }

  bool attach_committed_pages(int s, std::span<const int> committed, cudaStream_t stream) {
    if (committed.empty()) return true;
    auto& sp = slot_pages[s];
    const int cur = int(sp.size());
    if (cur + int(committed.size()) > max_pages_per_slot) return false;
    // Reserve capacity BEFORE retaining any page ref so a host std::bad_alloc never leaks active_refs.
    sp.reserve(std::size_t(cur) + committed.size());
    for (int p : committed) {
      pages.retain_committed_page(p);
      try {
        sp.push_back(p);
      } catch (...) {
        pages.release_active_page(p);
        throw;
      }
    }
    std::int32_t* dev_row = static_cast<std::int32_t*>(page_table.p) + std::size_t(s) * max_pages_per_slot;
    CUDA_CHECK(cudaMemcpyAsync(dev_row + cur, sp.data() + cur, committed.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    return true;
  }

  int additional_pages_needed(int s, int total_tokens) const {
    const int needed = (total_tokens + bk::kPageSize - 1) / bk::kPageSize;
    const int cur = int(slot_pages[s].size());
    return needed > cur ? (needed - cur) : 0;
  }

  bool ensure_pages(int s, int total_tokens, cudaStream_t stream) {
    const int needed = (total_tokens + bk::kPageSize - 1) / bk::kPageSize;
    if (needed > max_pages_per_slot) return false;
    auto& sp = slot_pages[s];
    const int cur = int(sp.size());
    if (needed <= cur) return true;
    const int add = needed - cur;
    if (pages.free_count() < add) return false;
    // Reserve capacity BEFORE allocating any page so a host std::bad_alloc never leaks active_refs.
    sp.reserve(std::size_t(needed));
    for (int i = 0; i < add; ++i) {
      const int p = pages.alloc_private_page();
      if (p < 0) return false;
      try {
        sp.push_back(p);
      } catch (...) {
        pages.release_active_page(p);
        throw;
      }
    }
    std::int32_t* dev_row = static_cast<std::int32_t*>(page_table.p) + std::size_t(s) * max_pages_per_slot;
    CUDA_CHECK(cudaMemcpyAsync(dev_row + cur, sp.data() + cur, std::size_t(add) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    return true;
  }

  void free_slot(int s) {
    if (s < 0 || s >= max_batch || !slot_in_use[s]) return;
    for (int p : slot_pages[s]) pages.release_active_page(p);
    slot_pages[s].clear();
    slot_pos_host[s] = 0;
    slot_in_use[s] = false;
  }

  void set_slot_pos(int s, std::int64_t pos, cudaStream_t stream) {
    slot_pos_host[s] = int(pos);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int64_t*>(slot_pos.p) + s, &pos, sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream));
  }
};

// ----------------------------------------------------------------------------- BatchDecoder (B = 1..max_batch, W = 1..8)

class BatchDecoder {
 public:
  struct VerifyOutput {
    const float* logits = nullptr;  // [B * W, UNPADDED_VOCAB]
    const ActT* taps = nullptr;     // [B * W, NUM_TAPS, H]
  };

  static std::size_t estimate_bytes(int max_batch, int max_verify_width, bool dense_fp8, bool spec_enabled) {
    const std::size_t B = std::size_t(max_batch);
    const std::size_t W = std::size_t(spec_enabled ? std::max(1, max_verify_width) : 1);
    const std::size_t M = B * W;
    std::size_t bytes = 0;
    // Per-row activations + logits + taps + MoE + unified gate/proj/wo FP32 scratch
    bytes += M * (20 * H * sizeof(float) + 5 * H * sizeof(ActT));
    bytes += M * std::size_t(UNPADDED_VOCAB) * sizeof(float);
    bytes += M * (HEADS * HEAD_DIM + TOP_K * (INTER + H) + H) * sizeof(ActT);
    bytes += M * (PROJ_ROWS_FULL + H + 3 * 16) * sizeof(float);
    bytes += M * (3 * 2 * H + 3 * GATE_RANK) * sizeof(ActT);
    if (M >= 2) {
      bytes += std::size_t(3) * 8 * H * sizeof(ActT);
    }
    if (M > 8) {
      bytes += M * std::size_t(kHeadChunk) * sizeof(float);
    }
    if (dense_fp8 && M > 32) bytes += std::size_t(kHeadChunk) * H * sizeof(ActT);
    if (spec_enabled) {
      bytes += M * std::size_t(NUM_TAPS) * H * sizeof(ActT);
      bytes += bk::attn_verify_workspace_bytes(max_batch, int(W));
      if (W > 1) {
        // Backup conv (2 rows) + backup sliding rings (W rows) + 48-visit conv histories (M rows)
        bytes += B * NUM_VISITS * 2 * (2 * KV_DIM + 2 * H) * sizeof(float);
        bytes += B * NUM_SLIDING_VISITS * W * (2 * KV_DIM * sizeof(ActT) + sizeof(std::int64_t));
        bytes += NUM_VISITS * M * (2 * KV_DIM + 2 * H) * sizeof(float);
      }
    } else {
      bytes += bk::attn_workspace_bytes(max_batch);
    }
    // Headroom for CUDA graph instantiation
    bytes += std::size_t(spec_enabled ? 384 : 128) << 20;
    return bytes;
  }

  BatchDecoder(const Weights& w, const Ablation& abl, BatchStatePool& pool, int max_batch, int max_verify_width = 1,
               bool spec_enabled = false, std::optional<std::size_t> vram_limit_bytes = std::nullopt,
               std::size_t free_at_entry = 0)
      : w_(w),
        abl_(abl),
        pool_(pool),
        max_batch_(max_batch),
        max_verify_width_(spec_enabled ? std::max(1, max_verify_width) : 1),
        max_rows_(max_batch * max_verify_width_),
        spec_enabled_(spec_enabled),
        vram_limit_bytes_(vram_limit_bytes),
        free_at_entry_(free_at_entry),
        gemm_(std::make_unique<pk::Gemm>()),
        tokens_host_(std::size_t(max_rows_) * sizeof(std::int64_t)),
        slots_host_(std::size_t(max_batch_) * sizeof(std::int32_t)),
        wlim_host_(std::size_t(max_batch_) * sizeof(std::int32_t)) {
    auto alloc = [&](std::size_t bytes) {
      storage_.emplace_back(bytes);
      storage_.back().fill(0);
      return storage_.back().p;
    };
    const std::size_t B = std::size_t(max_batch_);
    const std::size_t M = std::size_t(max_rows_);
    auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); };
    auto bf16 = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * sizeof(ActT))); };
    auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * sizeof(int))); };

    tokens_dev_ = static_cast<std::int64_t*>(alloc(M * sizeof(std::int64_t)));
    slot_ids_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    slot_w_limit_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    B0_ = f32(M * H), B1_ = f32(M * H);
    for (int p = 0; p < PASSES; ++p) B2_[p] = f32(M * H), E_[p] = f32(M * H);
    hid_ = f32(M * H), part_ = f32(M * H), y_ = f32(M * H), y2_ = f32(M * H);
    xin_ = bf16(3 * M * H), xm_ = bf16(M * H), xf_ = bf16(M * H);
    xin_chunk_ = M >= 2 ? bf16(3 * 8 * H) : nullptr;
    mixf_ = f32(3 * M * H), msgs_ = f32(4 * M * H), mixm_ = f32(M * H);
    proj_fp32_ = f32(M * PROJ_ROWS_FULL);
    attn_out_ = bf16(M * HEADS * HEAD_DIM);
    wo_fp32_ = f32(M * H);
    gin_ = bf16(3 * M * 2 * H);
    g16_ = bf16(3 * M * GATE_RANK);
    gate_ss_ = f32(3 * M * 16);
    router_logits_ = f32(M * EXPERTS);
    idx_ = i32(M * TOP_K);
    wts_ = f32(M * TOP_K);
    counts_ = i32(EXPERTS);
    tiles_n_ = i32(1);
    pair_token_ = i32(M * TOP_K);
    pair_dst_ = i32(M * TOP_K);
    tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_rows_)) * sizeof(int4)));
    act_ = bf16(M * TOP_K * INTER);
    down_ = bf16(M * TOP_K * H);
    moe_sum_ = bf16(M * H);
    logits_ = f32(M * UNPADDED_VOCAB);
    head_fp32_ = M > 8 ? f32(M * kHeadChunk) : nullptr;
    wscratch_ = (w.dense_fp8 && M > 32) ? bf16(std::size_t(kHeadChunk) * H) : nullptr;
    attn_ws_ = DeviceBuffer(spec_enabled_ ? std::max(bk::attn_workspace_bytes(max_batch_),
                                                     bk::attn_verify_workspace_bytes(max_batch_, max_verify_width_))
                                          : bk::attn_workspace_bytes(max_batch_));
    attn_ws_.fill(0);

    if (spec_enabled_) {
      taps_ = bf16(M * NUM_TAPS * H);
      if (max_verify_width_ > 1) {
        const std::size_t W = std::size_t(max_verify_width_);
        bk_buf_.max_W = max_verify_width_;
        bk_buf_.conv_k = f32(B * NUM_VISITS * 2 * KV_DIM);
        bk_buf_.conv_v = f32(B * NUM_VISITS * 2 * KV_DIM);
        bk_buf_.conv_attn = f32(B * NUM_VISITS * 2 * H);
        bk_buf_.conv_mlp = f32(B * NUM_VISITS * 2 * H);
        bk_buf_.ring_k = bf16(B * NUM_SLIDING_VISITS * W * KV_DIM);
        bk_buf_.ring_v = bf16(B * NUM_SLIDING_VISITS * W * KV_DIM);
        bk_buf_.ring_pos = static_cast<std::int64_t*>(alloc(B * NUM_SLIDING_VISITS * W * sizeof(std::int64_t)));

        hist_kv_visit_stride_ = M * KV_DIM;
        hist_h_visit_stride_ = M * H;
        hist_k_ = f32(std::size_t(NUM_VISITS) * hist_kv_visit_stride_);
        hist_v_ = f32(std::size_t(NUM_VISITS) * hist_kv_visit_stride_);
        hist_attn_ = f32(std::size_t(NUM_VISITS) * hist_h_visit_stride_);
        hist_mlp_ = f32(std::size_t(NUM_VISITS) * hist_h_visit_stride_);
      }
    }

    segments_[0] = upload_ints(proj_layout(false, true));
    segments_[1] = upload_ints(proj_layout(false, false));
    full_segments_ = upload_ints(proj_layout(true, false));
  }

  ~BatchDecoder() {
    for (auto& [b, exec] : graphs_)
      if (exec) cudaGraphExecDestroy(exec);
    for (auto& [k, exec] : verify_graphs_)
      if (exec) cudaGraphExecDestroy(exec);
  }

  void capture_buckets(cudaStream_t stream) {
    // Capture power-of-2 + intermediate buckets (1,2,3,4,6,8,12,16,24,32,48,64) up to max_batch_
    // to eliminate non-power-of-two padding cliffs at c=12, c=24, c=48.
    const std::vector<int> buckets = build_batch_buckets(max_batch_);

    // Warmup & capture with all slot_ids = -1 so ZERO state is modified.
    std::vector<std::int64_t> ztok(std::size_t(max_rows_), 0);
    std::vector<std::int32_t> nslot(std::size_t(max_batch_), -1);
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ztok.data(), ztok.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, nslot.data(), nslot.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int B : buckets) {
      for (int w = 0; w < 2; ++w) decode_eager(B, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      cudaGraph_t graph = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      try {
        decode_eager(B, stream);
      } catch (...) {
        cudaStreamEndCapture(stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
      }
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      cudaGraphExec_t exec = nullptr;
      const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
      cudaGraphDestroy(graph);
      CUDA_CHECK(err);
      CUDA_CHECK(cudaGraphUpload(exec, stream));
      graphs_[B] = exec;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Runs one plain 1-token batched decode step for `inputs` (1..max_batch_). Returns logits [B_active, UNPADDED_VOCAB].
  const float* run_step(std::span<const BatchSlotStep> inputs, cudaStream_t stream) {
    const int B_active = int(inputs.size());
    if (B_active < 1 || B_active > max_batch_) throw std::invalid_argument("BatchDecoder::run_step: bad batch size");

    auto it = graphs_.lower_bound(B_active);
    const int B_cap = (it != graphs_.end()) ? it->first : B_active;

    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    for (int i = 0; i < B_active; ++i) {
      ht[i] = inputs[i].token;
      hs[i] = inputs[i].slot;
    }
    const int tok_copy = (B_cap < 8 && max_rows_ >= 8) ? 8 : B_cap;
    for (int i = B_active; i < B_cap; ++i) {
      ht[i] = 0;
      hs[i] = -1;
    }
    for (int i = B_cap; i < tok_copy; ++i) ht[i] = 0;
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(tok_copy) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));

    if (it != graphs_.end()) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_eager(B_active, stream);
    }
    return logits_;
  }

  const float* last_logits() const { return logits_; }

  // Runs one batched speculative verify pass for B_active slots x W trial rows (W in {1, 2, 4, 8}).
  // Captures target taps [B_cap * W, NUM_TAPS, H] and pre-verify backup + conv histories (when W > 1).
  // Does NOT advance slot_pos; caller MUST call rollback_verify() with commit_counts [B_cap].
  VerifyOutput run_verify(int B_active, int W, const std::int64_t* tokens_host_bw, const std::int32_t* slot_ids_host,
                          const std::int32_t* slot_w_limit_host, int& B_cap_out, cudaStream_t stream) {
    if (!spec_enabled_) throw std::logic_error("BatchDecoder::run_verify called with spec disabled");
    if (B_active < 1 || B_active > max_batch_ || W < 1 || W > max_verify_width_)
      throw std::invalid_argument("BatchDecoder::run_verify: bad B or W");

    const int B_cap = bucket_for_batch(B_active, max_batch_);
    B_cap_out = B_cap;

    // Lazily capture CUDA graph for (B_cap, W) using padded slot_ids=-1 (ZERO mutation of live slots).
    // If a process VRAM lease cap is active, require >= 64 MiB remaining headroom before capturing a lazy graph;
    // otherwise fall back safely to eager verify without ever crossing the lease.
    const std::pair<int, int> key{B_cap, W};
    auto it = verify_graphs_.find(key);
    if (it == verify_graphs_.end() && verify_graphs_.size() < 24 && lazy_graph_within_lease()) {
      ensure_verify_graph(B_cap, W, stream);
      it = verify_graphs_.find(key);
    }

    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    auto* hw = static_cast<std::int32_t*>(wlim_host_.data());
    for (int b = 0; b < B_active; ++b) {
      hs[b] = slot_ids_host[b];
      hw[b] = slot_w_limit_host[b];
      for (int w = 0; w < W; ++w) ht[b * W + w] = tokens_host_bw[b * W + w];
    }
    const int n_rows = B_cap * W;
    const int tok_copy = (n_rows < 8 && max_rows_ >= 8) ? 8 : n_rows;
    for (int b = B_active; b < B_cap; ++b) {
      hs[b] = -1;
      hw[b] = 1;
      for (int w = 0; w < W; ++w) ht[b * W + w] = 0;
    }
    for (int i = n_rows; i < tok_copy; ++i) ht[i] = 0;
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(tok_copy) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_w_limit_dev_, hw, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));

    if (it != verify_graphs_.end() && it->second != nullptr) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_verify_eager(B_cap, W, stream);
    }
    return {logits_, taps_};
  }

  void rollback_verify(int B_cap, int W, const std::int32_t* commit_counts_dev, cudaStream_t stream) {
    bk::BatchedVerifyStatePtrs st = state_ptrs();
    bk::BatchedVerifyHistory hist{};
    hist.hist_k = hist_k_;
    hist.hist_v = hist_v_;
    hist.hist_attn = hist_attn_;
    hist.hist_mlp = hist_mlp_;
    hist.kv_visit_stride = hist_kv_visit_stride_;
    hist.h_visit_stride = hist_h_visit_stride_;
    bk::batched_verify_rollback(st, bk_buf_, hist, slot_ids_dev_, slot_w_limit_dev_, commit_counts_dev, B_cap, W, stream);
  }

  const std::int32_t* slot_w_limit_dev() const { return slot_w_limit_dev_; }

 private:
  bool lazy_graph_within_lease() const {
    if (!vram_limit_bytes_.has_value()) return true;
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    const std::size_t used = free_at_entry_ > free_b ? (free_at_entry_ - free_b) : 0;
    const std::size_t rem = *vram_limit_bytes_ > used ? (*vram_limit_bytes_ - used) : 0;
    return rem >= (std::size_t(64) << 20);
  }

  bk::BatchedVerifyStatePtrs state_ptrs() const {
    bk::BatchedVerifyStatePtrs st{};
    st.conv_k = static_cast<float*>(pool_.conv_k.p);
    st.conv_v = static_cast<float*>(pool_.conv_v.p);
    st.conv_attn = static_cast<float*>(pool_.conv_attn.p);
    st.conv_mlp = static_cast<float*>(pool_.conv_mlp.p);
    st.conv_kv_slot_stride = pool_.conv_kv_slot_stride;
    st.conv_h_slot_stride = pool_.conv_h_slot_stride;
    st.k_ring = static_cast<ActT*>(pool_.k_ring.p);
    st.v_ring = static_cast<ActT*>(pool_.v_ring.p);
    st.ring_pos = static_cast<std::int64_t*>(pool_.ring_pos.p);
    st.ring_slot_stride = pool_.ring_slot_stride;
    st.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
    st.slot_pos = static_cast<std::int64_t*>(pool_.slot_pos.p);
    return st;
  }

  void ensure_verify_graph(int B_cap, int W, cudaStream_t stream) {
    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    auto* hw = static_cast<std::int32_t*>(wlim_host_.data());
    const int n_tok = (B_cap * W < 8 && max_rows_ >= 8) ? 8 : (B_cap * W);
    std::memset(ht, 0, std::size_t(n_tok) * sizeof(std::int64_t));
    for (int b = 0; b < B_cap; ++b) {
      hs[b] = -1;
      hw[b] = W;
    }
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(n_tok) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_w_limit_dev_, hw, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    decode_verify_eager(B_cap, W, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      decode_verify_eager(B_cap, W, stream);
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph) cudaGraphDestroy(graph);
      verify_graphs_[{B_cap, W}] = nullptr;
      return;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
      verify_graphs_[{B_cap, W}] = nullptr;
      return;
    }
    CUDA_CHECK(cudaGraphUpload(exec, stream));
    verify_graphs_[{B_cap, W}] = exec;
  }

  const float* const* block_table(const std::vector<const float*>& blocks) {
    auto it = tables_.find(blocks);
    if (it == tables_.end()) {
      DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
      if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
      it = tables_.emplace(blocks, std::move(b)).first;
    }
    return static_cast<const float* const*>(it->second.p);
  }

  void decode_core_eager(int B, int W, bool verify_mode, cudaStream_t stream) {
    const int M = B * W;
    const Weights& w = w_;
    const Ablation& abl = abl_;
    const std::size_t slab = std::size_t(M) * H;
    const bool use_gemv = (M <= 8);
    const bool save_hist = (verify_mode && W > 1);

    if (save_hist) {
      bk::batched_verify_save(state_ptrs(), bk_buf_, slot_ids_dev_, slot_w_limit_dev_, B, W, stream);
    }

    const float* attn_dirs = f32p(abl.attn_dirs);
    const float* attn_w = f32p(abl.attn_w);
    const float* mlp_dirs = f32p(abl.mlp_dirs);
    const float* mlp_w = f32p(abl.mlp_w);

    auto run_mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                       const float* abl_r, const float* abl_w, float* partial_out, const float* queries, int Q,
                       const float* norm_w, ActT* out_bf16, float* out_f32, ActT* tap_new_bf16 = nullptr,
                       ActT* tap_out_copy = nullptr) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : block_table(blocks);
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries;
      p.Q = Q;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      if (tap_new_bf16) {
        p.new_bf16 = tap_new_bf16;
        p.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
      }
      if (tap_out_copy) {
        p.out_copy = tap_out_copy;
        p.out_copy_stride = std::int64_t(NUM_TAPS) * H;
      }
      p.M = M;
      dense::mix(p, false, stream);
    };

    auto next_gemv_chunk = [](int rem) -> int {
      if (rem <= 8) return rem;
      if (rem == 9) return 5;  // keep every chunk in {2..8} so M >= 2 stays on gemv_mma_kernel
      return 8;
    };

    // Keep embed RMSNorm on fixed BW=64 (M >= 8) whenever buffer capacity permits so M=1..7 matches M>=8 bit-for-bit.
    const int M_embed = (M < 8 && max_rows_ >= 8) ? 8 : M;
    if (w.embed_q8()) {
      dense::embed_q8(tokens_dev_, M_embed, w.embed.data, f32p(w.embed_norm), B0_, false, stream);
    } else if (w.embed_q4()) {
      auto desc = ops::make_inkling_embed_q4_desc(M_embed, VOCAB, /*allow_pdl=*/false);
      desc.token_bounds = ops::TokenBoundsMode::PreValidated;
      const ops::EmbedQ4RmsNormArgs args{tokens_dev_, w.embed.data, f32p(w.embed_norm), B0_};
      ops::run_embed_q4_rmsnorm(desc, args, stream, /*allow_fallback=*/false, /*force_fallback=*/false);
    } else {
      dense::embed(tokens_dev_, M_embed, bf16p(w.embed), f32p(w.embed_norm), B0_, false, stream);
    }

    int prev_v = -1;
    int tap = 0;
    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool msgs = is_loop_layer(l) && p > 0;
      const bool boundary = (l % BLOCK == 0);
      const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = B0_, base = B0_, write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = B1_, base = B1_, write = nullptr;
      } else {
        hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
        base = part_, write = y2_;
      }

      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? part_ : E_[p - 1];
        const float* src_write = i == 0 ? y2_ : nullptr;
        run_mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
                f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
        if (i == 0) write = nullptr;
      }

      // ---- Attention read + message gates + optional target tap capture (visits 5, 23, 41)
      ActT* tap_dst = nullptr;
      if (verify_mode && taps_ && tap < 3 && v == TAP_VISITS[tap]) {
        tap_dst = taps_ + std::size_t(tap) * H;
        ++tap;
      }
      const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
      const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
      float* partial_out = hid != base ? hid : nullptr;
      if (msgs) {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_, tap_dst);
        bk::batched_message_gates(mixf_, !layer.full, msgs_, 3, M, bf16p(layer.qkv_gate_down),
                                  bf16p(layer.qkv_gate_forget16), bf16p(layer.qkv_gate_gain16), f32p(layer.attn_norm3),
                                  gin_, g16_, gate_ss_, xin_, stream);
      } else {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
                f32p(layer.attn_norm_mix), xin_, nullptr, tap_dst);
      }

      // ---- Attention projections over M = B * W rows
      // Always use chunked dense::proj on gemv_mma_kernel (chunks in {2..8}, padding M=1 -> 2 when max_rows_ >= 2)
      // so wproj is 0-ULP M-invariant across all M = 1..96 and cannot trigger cuBLASLt or M=1/M>=2 MoE router flips.
      const int R = proj_rows(layer.full);
      const bool single = !layer.full && !msgs;
      const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
      if (M == 1 && max_rows_ >= 2) {
        const ActT* xptr = xin_;
        if (!single) {
          for (int g = 0; g < 3; ++g) {
            CUDA_CHECK(cudaMemcpyAsync(xin_chunk_ + std::size_t(g) * 2 * H, xin_ + std::size_t(g) * slab,
                                       std::size_t(H) * sizeof(ActT), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(xin_chunk_ + std::size_t(g) * 2 * H + H, 0,
                                       std::size_t(H) * sizeof(ActT), stream));
          }
          xptr = xin_chunk_;
        } else {
          CUDA_CHECK(cudaMemsetAsync(xin_ + H, 0, std::size_t(H) * sizeof(ActT), stream));
        }
        dense::proj(xptr, 2, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full), proj_fp32_,
                    false, stream);
      } else if (M <= 8) {
        dense::proj(xin_, M, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full), proj_fp32_,
                    false, stream);
      } else {
        for (int m0 = 0; m0 < M;) {
          const int cm = next_gemv_chunk(M - m0);
          const ActT* xptr = xin_ + std::size_t(m0) * H;
          if (!single) {
            for (int g = 0; g < 3; ++g) {
              CUDA_CHECK(cudaMemcpyAsync(xin_chunk_ + std::size_t(g) * cm * H,
                                         xin_ + std::size_t(g) * slab + std::size_t(m0) * H,
                                         std::size_t(cm) * H * sizeof(ActT), cudaMemcpyDeviceToDevice, stream));
            }
            xptr = xin_chunk_;
          }
          dense::proj(xptr, cm, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                      proj_fp32_ + std::size_t(m0) * R, false, stream);
          m0 += cm;
        }
      }

      // ---- Batched per-slot attention (1-tok fast path when !verify_mode or W==1 vs multi-token verify path)
      const bool use_verify_kernels = (verify_mode && W > 1);
      if (!use_verify_kernels) {
        bk::BatchedAttnArgs a;
        a.proj = static_cast<const void*>(proj_fp32_);
        a.proj_fp32 = true;
        a.B = B;
        a.slot_ids = slot_ids_dev_;
        a.slot_pos = static_cast<const std::int64_t*>(pool_.slot_pos.p);
        a.q_scale = f32p(layer.q_scale);
        a.r_scale = f32p(layer.r_scale);
        a.rel_proj = f32p(layer.rel_proj);
        a.conv_k_visit = static_cast<float*>(pool_.conv_k.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
        a.conv_v_visit = static_cast<float*>(pool_.conv_v.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
        a.conv_slot_stride = pool_.conv_kv_slot_stride;
        a.workspace = attn_ws_.p;
        a.workspace_size = attn_ws_.bytes;
        a.out = attn_out_;
        if (layer.full) {
          a.k_conv = f32p(layer.k_static);
          a.v_conv = f32p(layer.v_static);
          a.k_full_pool = pool_.k_full_pool.p;
          a.v_full_pool = pool_.v_full_pool.p;
          a.page_table = static_cast<const std::int32_t*>(pool_.page_table.p);
          a.max_pages_per_slot = pool_.max_pages_per_slot;
          a.full_visit_idx = FULL_INDEX[v];
          a.max_ctx = pool_.max_ctx;
          a.cache_fp8 = pool_.kv_fp8;
          a.window = 0;
          a.log_scaling = true;
        } else {
          const int ri = RING_INDEX[v];
          a.k_conv = f32p(layer.k_conv);
          a.v_conv = f32p(layer.v_conv);
          a.k_ring_visit = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
          a.v_ring_visit = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
          a.ring_pos_visit = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(ri) * WINDOW;
          a.ring_slot_stride = pool_.ring_slot_stride;
          a.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
          a.cache_fp8 = false;
          a.window = WINDOW;
          a.log_scaling = false;
        }
        bk::batched_attn_decode(a, stream);
      } else {
        bk::BatchedVerifyAttnArgs va;
        va.proj = static_cast<const void*>(proj_fp32_);
        va.proj_fp32 = true;
        va.B = B;
        va.W = W;
        va.slot_ids = slot_ids_dev_;
        va.slot_w_limit = slot_w_limit_dev_;
        va.slot_pos = static_cast<const std::int64_t*>(pool_.slot_pos.p);
        va.q_scale = f32p(layer.q_scale);
        va.r_scale = f32p(layer.r_scale);
        va.rel_proj = f32p(layer.rel_proj);
        va.conv_k_visit = static_cast<float*>(pool_.conv_k.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
        va.conv_v_visit = static_cast<float*>(pool_.conv_v.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
        va.conv_slot_stride = pool_.conv_kv_slot_stride;
        va.hist_k_visit = save_hist ? (hist_k_ + std::size_t(v) * hist_kv_visit_stride_) : nullptr;
        va.hist_v_visit = save_hist ? (hist_v_ + std::size_t(v) * hist_kv_visit_stride_) : nullptr;
        va.workspace = attn_ws_.p;
        va.workspace_size = attn_ws_.bytes;
        va.out = attn_out_;
        if (layer.full) {
          va.k_conv = f32p(layer.k_static);
          va.v_conv = f32p(layer.v_static);
          va.k_full_pool = pool_.k_full_pool.p;
          va.v_full_pool = pool_.v_full_pool.p;
          va.page_table = static_cast<const std::int32_t*>(pool_.page_table.p);
          va.max_pages_per_slot = pool_.max_pages_per_slot;
          va.full_visit_idx = FULL_INDEX[v];
          va.max_ctx = pool_.max_ctx;
          va.cache_fp8 = pool_.kv_fp8;
          va.window = 0;
          va.log_scaling = true;
        } else {
          const int ri = RING_INDEX[v];
          va.k_conv = f32p(layer.k_conv);
          va.v_conv = f32p(layer.v_conv);
          va.k_ring_visit = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
          va.v_ring_visit = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
          va.ring_pos_visit = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(ri) * WINDOW;
          va.ring_slot_stride = pool_.ring_slot_stride;
          va.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
          va.cache_fp8 = false;
          va.window = WINDOW;
          va.log_scaling = false;
        }
        bk::batched_attn_verify(va, stream);
      }

      // ---- Output projection + per-slot post-attention 4-tap conv
      // Always use chunked dense::gemv on gemv_mma_kernel (padding M=1 -> 2 when max_rows_ >= 2) so wo is 0-ULP M-invariant.
      float* conv_attn_v = static_cast<float*>(pool_.conv_attn.p) + std::size_t(v) * (CONV_K - 1) * H;
      float* hist_attn_v = save_hist ? (hist_attn_ + std::size_t(v) * hist_h_visit_stride_) : nullptr;
      if (M == 1 && max_rows_ >= 2) {
        CUDA_CHECK(cudaMemsetAsync(attn_out_ + HEADS * HEAD_DIM, 0, std::size_t(HEADS * HEAD_DIM) * sizeof(ActT), stream));
        dense::gemv(attn_out_, 2, HEADS * HEAD_DIM, matrix(layer.wo), wo_fp32_, false, stream);
      } else {
        for (int m0 = 0; m0 < M;) {
          const int cm = next_gemv_chunk(M - m0);
          dense::gemv(attn_out_ + std::size_t(m0) * HEADS * HEAD_DIM, cm, HEADS * HEAD_DIM, matrix(layer.wo),
                      wo_fp32_ + std::size_t(m0) * H, false, stream);
          m0 += cm;
        }
      }
      if (!use_verify_kernels) {
        bk::batched_static_conv_step(wo_fp32_, true, H, B, H, slot_ids_dev_, f32p(layer.attn_conv), conv_attn_v,
                                     pool_.conv_h_slot_stride, y_, stream);
      } else {
        bk::batched_static_conv_verify(wo_fp32_, true, H, B, W, H, slot_ids_dev_, slot_w_limit_dev_,
                                       f32p(layer.attn_conv), conv_attn_v, pool_.conv_h_slot_stride, y_, hist_attn_v,
                                       stream);
      }

      // ---- MLP read + message gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      if (msgs) {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
        bk::batched_message_gates(mixm_, false, msgs_ + 3 * slab, 1, M, bf16p(layer.mlp1_gate_down),
                                  bf16p(layer.mlp1_gate_forget16), bf16p(layer.mlp1_gate_gain16), f32p(layer.mlp_norm1),
                                  gin_, g16_, gate_ss_, xm_, stream);
      } else {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
      }

      // ---- Router + grouped tensor-core MoE + per-slot post-MLP 4-tap conv
      bk::batched_router_logits(xm_, bf16p(layer.router_bf16), M, router_logits_, stream);
      CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), stream));
      pk::moe_topk(router_logits_, f32p(layer.router_bias), M, idx_, wts_, counts_, stream);
      const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
      pk::moe_route(idx_, M, work, stream);
      const Experts& ex = layer.experts;
      pk::MoeExperts me;
      me.fmt = static_cast<int>(ex.w13_fmt);
      me.w13 = ex.w13.data;
      me.w13_scale = ex.w13_scale.data;
      me.w13_factor = f32p(ex.w13_factor);
      me.w2 = ex.w2.data;
      me.w2_scale = ex.w2_scale.data;
      me.w2_factor = f32p(ex.w2_factor);
      me.w2_fmt = static_cast<int>(ex.w2_fmt);
      pk::moe_gate_up(me, xm_, M, work, act_, stream);
      pk::moe_down(me, act_, M, work, down_, stream);
      pk::moe_combine(down_, wts_, M, moe_sum_, stream);
      float* conv_mlp_v = static_cast<float*>(pool_.conv_mlp.p) + std::size_t(v) * (CONV_K - 1) * H;
      float* hist_mlp_v = save_hist ? (hist_mlp_ + std::size_t(v) * hist_h_visit_stride_) : nullptr;
      if (!use_verify_kernels) {
        bk::batched_static_conv_step(moe_sum_, false, H, B, H, slot_ids_dev_, f32p(layer.mlp_conv), conv_mlp_v,
                                     pool_.conv_h_slot_stride, y2_, stream);
      } else {
        bk::batched_static_conv_verify(moe_sum_, false, H, B, W, H, slot_ids_dev_, slot_w_limit_dev_,
                                       f32p(layer.mlp_conv), conv_mlp_v, pool_.conv_h_slot_stride, y2_, hist_mlp_v,
                                       stream);
      }
      prev_v = v;
    }

    // ---- Final read (capturing tap 3 = xf_ in verify mode) + LM head
    ActT* final_tap = (verify_mode && taps_) ? (taps_ + std::size_t(NUM_TAPS - 1) * H) : nullptr;
    run_mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H,
            mlp_w + 4 * std::size_t(prev_v), nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr,
            nullptr, final_tap);
    const bool head_gemv = use_gemv || (w.unembed.fp8() && M <= 32) || w.unembed.blocked();
    if (head_gemv) {
      if (M == 1 && max_rows_ >= 2) {
        CUDA_CHECK(cudaMemsetAsync(xf_ + H, 0, std::size_t(H) * sizeof(ActT), stream));
        dense::head(xf_, 2, matrix(w.unembed), logits_, false, stream);
      } else {
        for (int m0 = 0; m0 < M;) {
          const int cm = next_gemv_chunk(M - m0);
          dense::head(xf_ + std::size_t(m0) * H, cm, matrix(w.unembed), logits_ + std::size_t(m0) * UNPADDED_VOCAB,
                      false, stream);
          m0 += cm;
        }
      }
    } else {
      const float* hscale = w.unembed.fp8() ? f32p(w.unembed.scale) : nullptr;
      for (int v0 = 0; v0 < UNPADDED_VOCAB; v0 += kHeadChunk) {
        const int cv = std::min(kHeadChunk, UNPADDED_VOCAB - v0);
        const ActT* wchunk;
        if (w.unembed.fp8()) {
          const auto* q = static_cast<const std::uint8_t*>(w.unembed.w.data) + std::size_t(v0) * H;
          bk::dequant_e4m3_unscaled(q, cv, H, wscratch_, stream);
          wchunk = wscratch_;
        } else {
          wchunk = bf16p(w.unembed.w) + std::size_t(v0) * H;
        }
        gemm_->run(xf_, H, wchunk, H, head_fp32_, cv, true, M, cv, H, stream);
        bk::batched_head_scale_softcap(head_fp32_, hscale, M, cv, v0, UNPADDED_VOCAB, logits_, stream);
      }
    }

    // Plain 1-token path advances slot_pos directly; verify path advances ONCE per slot inside rollback_verify!
    if (!verify_mode) {
      bk::batched_advance_pos(slot_ids_dev_, B, static_cast<std::int64_t*>(pool_.slot_pos.p), stream);
    }
  }

  void decode_eager(int B, cudaStream_t stream) { decode_core_eager(B, 1, false, stream); }
  void decode_verify_eager(int B, int W, cudaStream_t stream) { decode_core_eager(B, W, true, stream); }

  const Weights& w_;
  const Ablation& abl_;
  BatchStatePool& pool_;
  int max_batch_;
  int max_verify_width_;
  int max_rows_;
  bool spec_enabled_;
  std::optional<std::size_t> vram_limit_bytes_;
  std::size_t free_at_entry_ = 0;
  std::unique_ptr<pk::Gemm> gemm_;

  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  std::array<DeviceBuffer, 2> segments_;
  DeviceBuffer full_segments_;
  DeviceBuffer attn_ws_;
  std::map<int, cudaGraphExec_t> graphs_;
  std::map<std::pair<int, int>, cudaGraphExec_t> verify_graphs_;
  PinnedHostBuffer tokens_host_, slots_host_, wlim_host_;

  std::int64_t* tokens_dev_ = nullptr;
  std::int32_t* slot_ids_dev_ = nullptr;
  std::int32_t* slot_w_limit_dev_ = nullptr;
  float *B0_ = nullptr, *B1_ = nullptr, *B2_[PASSES] = {}, *E_[PASSES] = {};
  float *hid_ = nullptr, *part_ = nullptr, *y_ = nullptr, *y2_ = nullptr;
  ActT *xin_ = nullptr, *xin_chunk_ = nullptr, *xm_ = nullptr, *xf_ = nullptr;
  float *mixf_ = nullptr, *msgs_ = nullptr, *mixm_ = nullptr;
  float* proj_fp32_ = nullptr;
  ActT* attn_out_ = nullptr;
  float* wo_fp32_ = nullptr;
  ActT *gin_ = nullptr, *g16_ = nullptr;
  float* gate_ss_ = nullptr;
  float* router_logits_ = nullptr;
  int *idx_ = nullptr, *counts_ = nullptr, *tiles_n_ = nullptr, *pair_token_ = nullptr, *pair_dst_ = nullptr;
  int4* tiles_ = nullptr;
  float* wts_ = nullptr;
  ActT *act_ = nullptr, *down_ = nullptr, *moe_sum_ = nullptr;
  float* logits_ = nullptr;
  float* head_fp32_ = nullptr;
  ActT* wscratch_ = nullptr;
  ActT* taps_ = nullptr;
  bk::BatchedVerifyBackup bk_buf_{};
  float *hist_k_ = nullptr, *hist_v_ = nullptr, *hist_attn_ = nullptr, *hist_mlp_ = nullptr;
  std::size_t hist_kv_visit_stride_ = 0, hist_h_visit_stride_ = 0;
};

// ----------------------------------------------------------------------------- BatchGenerator (implements Generator)

struct PrefixCheckpointGpu {
  int ring_rows = 0;  // min(prefix_len, WINDOW) valid rows saved per sliding visit
  DeviceBuffer k_ring, v_ring, ring_pos;
  DeviceBuffer conv_k, conv_v, conv_attn, conv_mlp;
  DeviceBuffer tail_k, tail_v;  // [NUM_FULL_VISITS, tail_tokens, KV_DIM] when tail_tokens > 0
  DeviceBuffer logits;          // [UNPADDED_VOCAB] fp32
  bool has_draft = false;
  DraftCheckpointGpu draft;     // populated when RuntimeOptions::spec is enabled
};

class BatchGenerator final : public Generator {
 public:
  explicit BatchGenerator(const RuntimeOptions& options)
      : options_(options), sampled_host_(std::size_t(std::max(1, options.max_batch)) * sizeof(std::int32_t)) {
    if (options.max_batch < 1) throw std::invalid_argument("max_batch must be >= 1");
    if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be >= 16");
    if (options.kv_cache_mode != KvCacheMode::PerPass)
      throw std::invalid_argument("--kv-cache-mode last_full requires --max-batch 1 (B=1 qualification control)");
    if (options.loop_passes != PASSES)
      throw std::invalid_argument("--loop-passes != 3 currently requires --max-batch 1");
    if (options.kv_q4 || options.kv_paged || options.kv_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) supports BF16 or --kv-fp8 KV cache (not --kv-q4/--kv-paged)");
    if (options.offload_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) requires resident weights (not --offload-vram-mb/--atlas-6g)");
    if (options.vram_limit_mb > 0 && is_gguf_model(options.model_dir))
      throw std::invalid_argument(
          "BatchGenerator: --vram-limit-mb is not supported for GGUF models (leave --vram-limit-mb 0)");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    std::size_t total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_at_entry_, &total_b));
    if (options.vram_limit_mb > 0) {
      // Preflight weights peak BEFORE allocating a single byte on GPU (header-only safetensors scan).
      SafetensorsModel hdr(options.model_dir);
      std::size_t non_vision_bytes = 0;
      for (const auto& f : hdr.files()) {
        for (const auto& t : f->tensors()) {
          const bool is_vis =
              (t.name.rfind("vision", 0) == 0 || t.name.rfind("visual", 0) == 0 || t.name.rfind("model.visual", 0) == 0);
          if (!is_vis || options.vision) non_vision_bytes += std::size_t(t.bytes);
        }
      }
      const std::size_t cap = std::size_t(options.vram_limit_mb) << 20;
      const std::size_t est_weights_peak = estimate_weights_peak_bytes(non_vision_bytes, options.dense_fp8,
                                                                       UNPADDED_VOCAB, HIDDEN);
      const VramLeaseCheck chk = check_vram_lease_fit(std::min(free_at_entry_, cap), est_weights_peak);
      if (!chk.ok) {
        throw std::runtime_error("BatchGenerator: VRAM lease cap exceeded before load_weights (peak_need=" +
                                 std::to_string(chk.planned_bytes >> 20) + " MiB, budget=" +
                                 std::to_string(chk.remaining_bytes >> 20) + " MiB)");
      }
    }
    weights_ = load_weights(options.model_dir, options.dense_fp8, stream_);
    ablation_ = load_ablation(options.ablation);
    init();
  }

  BatchGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : options_(options),
        weights_(std::move(weights)),
        ablation_(std::move(ablation)),
        sampled_host_(std::size_t(std::max(1, options.max_batch)) * sizeof(std::int32_t)) {
    if (options.max_batch < 1) throw std::invalid_argument("max_batch must be >= 1");
    if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be >= 16");
    if (options.kv_cache_mode != KvCacheMode::PerPass)
      throw std::invalid_argument("--kv-cache-mode last_full requires --max-batch 1 (B=1 qualification control)");
    if (options.loop_passes != PASSES)
      throw std::invalid_argument("--loop-passes != 3 currently requires --max-batch 1");
    if (options.kv_q4 || options.kv_paged || options.kv_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) supports BF16 or --kv-fp8 KV cache (not --kv-q4/--kv-paged)");
    if (options.offload_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) requires resident weights (not --offload-vram-mb/--atlas-6g)");
    if (options.vram_limit_mb > 0) {
      bool is_gguf = is_gguf_model(options.model_dir);
      for (const Layer& layer : weights_.layers) {
        if (static_cast<int>(layer.experts.w13_fmt) > 5 || static_cast<int>(layer.experts.w2_fmt) > 5) {
          is_gguf = true;
          break;
        }
      }
      if (is_gguf)
        throw std::invalid_argument(
            "BatchGenerator: --vram-limit-mb is not supported for GGUF models (leave --vram-limit-mb 0)");
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    std::size_t total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_at_entry_, &total_b));
    init();
  }

  ~BatchGenerator() override {
    if (stream_) cudaStreamSynchronize(stream_);
    prefix_cache_.clear(pool_.pages);
    decoder_.reset();
    drafter_.reset();
    prefiller_.reset();
    vision_.reset();
    if (restore_ev_stop_) cudaEventDestroy(restore_ev_stop_);
    if (restore_ev_start_) cudaEventDestroy(restore_ev_start_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }
  bool supports_images() const override { return vision_ != nullptr; }
  int max_batch_size() const override { return options_.max_batch; }
  bool supports_batched_spec() const override { return drafter_ != nullptr; }
  PrefixCacheStats cache_stats() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return prefix_cache_.stats();
  }

  bool can_admit_slot(int prompt_tokens, int max_new_tokens) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.free_slot_count() <= 0) return false;
    const int eff_max = std::max(0, std::min(max_new_tokens, options_.max_ctx - prompt_tokens));
    const int worst_tokens = std::min(options_.max_ctx, prompt_tokens + std::max(1, eff_max - 1));
    const int lifetime_pages = (worst_tokens + bk::kPageSize - 1) / bk::kPageSize;
    const int reserved = active_reserved_growth_pages();
    return pool_.pages.available_for_admission() >= lifetime_pages + reserved;
  }

  bool can_admit_slot(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.free_slot_count() <= 0) return false;
    const int n = int(full_prompt.ids.size());
    const int eff_max = effective_max_new(n, max_new_tokens, sampling);
    const int worst_tokens = std::min(options_.max_ctx, n + std::max(1, eff_max - 1));
    const int lifetime_pages = (worst_tokens + bk::kPageSize - 1) / bk::kPageSize;
    const bool cache_eligible = options_.prefix_cache && !full_prompt.bypass_prefix_cache &&
                                !options_.enable_persona_hooks && sampling.persona == nullptr &&
                                full_prompt.images.empty() && prefix_cache_.vram_budget_bytes() > 0;
    int F = 0;
    std::uint64_t hit_id = 0;
    if (cache_eligible && n > 0) {
      const StateIdentity identity = build_identity(full_prompt);
      if (const auto* hit = prefix_cache_.find_longest_prefix_const(identity, full_prompt.ids)) {
        F = int(hit->tokens.size()) / bk::kPageSize;
        hit_id = hit->id;
      }
    }
    const int fresh_lifetime = std::max(0, lifetime_pages - F);
    const int reserved = active_reserved_growth_pages();
    const int avail = pool_.pages.free_count() + prefix_cache_.reclaimable_pages_excluding(pool_.pages, hit_id);
    return avail >= fresh_lifetime + reserved;
  }

  GenerateStats warm_start(const Prompt& full_prompt, const Sampling& sampling) override {
    if (!options_.prefix_cache || prefix_cache_.vram_budget_bytes() == 0) {
      throw std::invalid_argument("warm_start requires batched prefix caching to be enabled");
    }
    if (options_.enable_persona_hooks || sampling.persona != nullptr) {
      throw std::invalid_argument("warm_start is not supported with persona hooks in continuous-batch mode");
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
    if (int(full_prompt.ids.size()) >= options_.max_ctx) {
      throw std::invalid_argument("warm_start: prompt of " + std::to_string(full_prompt.ids.size()) +
                                  " tokens leaves no generation room within max_ctx " +
                                  std::to_string(options_.max_ctx));
    }
    std::int32_t first_tok = 0;
    GenerateStats stats;
    const int s = start_slot(full_prompt, 1, sampling, first_tok, stats);
    if (s < 0) throw std::runtime_error("warm_start: no free batch slot");
    release_slot(s);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const StateIdentity expected_id = build_identity(full_prompt);
      const auto* hit = prefix_cache_.find_longest_prefix_const(expected_id, full_prompt.ids);
      const bool exact_retained = hit != nullptr && hit->tokens.size() == full_prompt.ids.size() &&
                                  std::equal(full_prompt.ids.begin(), full_prompt.ids.end(), hit->tokens.begin()) &&
                                  (!drafter_ || hit->payload.has_draft);
      if (!exact_retained) {
        throw std::runtime_error(
            "warm_start: batched prefix cache could not retain exact warmed prefix and identity");
      }
    }
    return stats;
  }

  int start_slot(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                 std::int32_t& first_token, GenerateStats& stats) override {
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
    }
    if (n >= options_.max_ctx) {
      throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                  std::to_string(options_.max_ctx));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.free_slot_count() <= 0) return -1;

    // Prefix-cache eligibility: text + supported resident KV modes (with or without batched spec), unhooked, not bypassed.
    const bool cache_eligible = options_.prefix_cache && !full_prompt.bypass_prefix_cache &&
                                !options_.enable_persona_hooks && sampling.persona == nullptr &&
                                full_prompt.images.empty() && prefix_cache_.vram_budget_bytes() > 0;
    const StateIdentity identity = build_identity(full_prompt);
    const auto* hit = cache_eligible ? prefix_cache_.find_longest_prefix(identity, prompt) : nullptr;
    if (hit && drafter_ && !hit->payload.has_draft) hit = nullptr;

    const int L = hit ? int(hit->tokens.size()) : 0;
    const int F = L / bk::kPageSize;
    const int tail = L % bk::kPageSize;
    const std::uint64_t hit_id = hit ? hit->id : 0;

    const int eff_max = effective_max_new(n, max_new_tokens, sampling);
    const int worst_tokens = std::min(options_.max_ctx, n + std::max(1, eff_max - 1));
    const int lifetime_pages = (worst_tokens + bk::kPageSize - 1) / bk::kPageSize;
    const int fresh_lifetime = std::max(0, lifetime_pages - F);
    const int initial_pages = (n + 1 + bk::kPageSize - 1) / bk::kPageSize;
    const int fresh_needed = std::max(0, initial_pages - F);

    if (pool_.active_slot_count() > 0) {
      const int reserved = active_reserved_growth_pages();
      const int avail = pool_.pages.free_count() + prefix_cache_.reclaimable_pages_excluding(pool_.pages, hit_id);
      if (avail < fresh_lifetime + reserved) return -1;
    }

    int evictions = 0;
    if (pool_.pages.free_count() < fresh_needed) {
      evictions += prefix_cache_.ensure_free_pages(pool_.pages, fresh_needed, hit_id);
    }
    if (pool_.pages.free_count() < fresh_needed) return -1;

    const int s = pool_.alloc_slot();
    if (s < 0) return -1;
    slot_meta_[s].max_pos = worst_tokens;

    bool slot_committed = false;
    struct SlotGuard {
      BatchStatePool& pool;
      cudaStream_t stream;
      int slot;
      bool& committed;
      ~SlotGuard() {
        if (!committed && slot >= 0) {
          if (stream) cudaStreamSynchronize(stream);
          pool.free_slot(slot);
        }
      }
    } slot_guard{pool_, stream_, s, slot_committed};

    if (F > 0 && !pool_.attach_committed_pages(s, hit->committed_pages, stream_)) return -1;
    if (!pool_.ensure_pages(s, n + 1, stream_)) return -1;

    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t restored_bytes = 0;
    double restore_seconds = 0.0;
    bool recorded_restore_events = false;
    const int root_boundary =
        cache_eligible ? validate_root_prefix_tokens(prompt, full_prompt.images, full_prompt.root_prefix_tokens) : 0;

    if (L == n && hit != nullptr) {
      // Exact prompt hit: zero prefill compute and zero full-KV scratch gather.
      CUDA_CHECK(cudaEventRecord(restore_ev_start_, stream_));
      restored_bytes += restore_recurrent_state(s, hit->payload);
      if (tail > 0) {
        bk::copy_tail_kv_to_pool_page(hit->payload.tail_k.p, hit->payload.tail_v.p, tail, pool_.kv_fp8,
                                      pool_.slot_pages[s][std::size_t(F)], pool_.k_full_pool.p, pool_.v_full_pool.p,
                                      stream_);
        restored_bytes += hit->payload.tail_k.bytes + hit->payload.tail_v.bytes;
      }
      CUDA_CHECK(cudaMemcpyAsync(prefill_logits_.p, hit->payload.logits.p, prefill_logits_.bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      restored_bytes += prefill_logits_.bytes;
      if (drafter_) {
        restored_bytes += drafter_->restore_slot_checkpoint(s, hit->payload.draft);
      }
      CUDA_CHECK(cudaEventRecord(restore_ev_stop_, stream_));
      recorded_restore_events = true;
      pool_.set_slot_pos(s, n, stream_);
      prefix_cache_.record_d2d_restore(restored_bytes);
    } else {
      State st = pool_.prefill_view(s);
      int scatter_from = 0;
      if (L == 0) {
        reset_state(st, stream_);
        if (drafter_) drafter_->reset_slot(s);
      } else {
        CUDA_CHECK(cudaEventRecord(restore_ev_start_, stream_));
        restored_bytes += restore_recurrent_state(s, hit->payload);
        const std::size_t es = pool_.kv_fp8 ? 1 : 2;
        if (F > 0) {
          bk::gather_full_kv(pool_.k_full_pool.p, pool_.v_full_pool.p, s,
                             static_cast<const std::int32_t*>(pool_.page_table.p), pool_.max_pages_per_slot,
                             F * bk::kPageSize, pool_.kv_fp8, pool_.scratch_k_full.p, pool_.scratch_v_full.p,
                             pool_.max_ctx, stream_);
          restored_bytes += std::size_t(2) * NUM_FULL_VISITS * std::size_t(F * bk::kPageSize) * KV_DIM * es;
        }
        if (tail > 0) {
          bk::copy_tail_kv_to_scratch(hit->payload.tail_k.p, hit->payload.tail_v.p, tail, pool_.kv_fp8,
                                      pool_.scratch_k_full.p, pool_.scratch_v_full.p, pool_.max_ctx, F * bk::kPageSize,
                                      stream_);
          restored_bytes += hit->payload.tail_k.bytes + hit->payload.tail_v.bytes;
        }
        if (drafter_) {
          restored_bytes += drafter_->restore_slot_checkpoint(s, hit->payload.draft);
        }
        CUDA_CHECK(cudaEventRecord(restore_ev_stop_, stream_));
        recorded_restore_events = true;
        st.length = L;
        scatter_from = F * bk::kPageSize;
        prefix_cache_.record_d2d_restore(restored_bytes);
      }

      const std::size_t chunk = std::size_t(prefill_chunk_);
      struct Encoded {
        const PromptImage* image;
        DeviceBuffer rows;
      };
      std::deque<Encoded> encoded;
      std::size_t next_image = 0;
      std::vector<EmbedRows> rows;

      const std::vector<int> forced_splits = (L == 0) ? snapshot_test_prefill_splits() : std::vector<int>{};
      std::size_t split_idx = 0;
      std::size_t start = std::size_t(L);
      while (start < std::size_t(n)) {
        std::size_t target_end = std::min(start + chunk, std::size_t(n));
        if (split_idx < forced_splits.size() && forced_splits[split_idx] > 0) {
          target_end = std::min(target_end, start + std::size_t(forced_splits[split_idx]));
          ++split_idx;
        }
        if (root_boundary > L && root_boundary < n && start < std::size_t(root_boundary) &&
            target_end > std::size_t(root_boundary)) {
          target_end = std::size_t(root_boundary);
        }
        const int T = int(target_end - start);
        for (; next_image < full_prompt.images.size() &&
               std::size_t(full_prompt.images[next_image].pos) < start + std::size_t(T);
             ++next_image) {
          const PromptImage& image = full_prompt.images[next_image];
          const auto tv = std::chrono::steady_clock::now();
          DeviceBuffer out = vision_->encode(image.rgb->data(), image.height, image.width);
          while (!encoded.empty() &&
                 std::size_t(encoded.front().image->pos + encoded.front().image->tokens()) <= start) {
            encoded.pop_front();
          }
          encoded.push_back(Encoded{&image, std::move(out)});
          stats.vision_seconds += seconds_since(tv);
          ++stats.images_encoded;
        }
        rows.clear();
        for (const Encoded& e : encoded) {
          const std::int64_t first = std::max<std::int64_t>(e.image->pos, std::int64_t(start));
          const std::int64_t last =
              std::min<std::int64_t>(std::int64_t(e.image->pos) + e.image->tokens(), std::int64_t(start) + T);
          if (first < last) {
            rows.push_back(EmbedRows{int(first - std::int64_t(start)), int(last - first),
                                     static_cast<const ActT*>(e.rows.p) + std::size_t(first - e.image->pos) * H});
          }
        }
        CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, prompt.data() + start, std::size_t(T) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream_));
        Tensor taps_view;
        Tensor* taps_ptr = nullptr;
        if (drafter_) {
          taps_view = rm(prefill_taps_.p, kActDType, {T, NUM_TAPS, H});
          taps_ptr = &taps_view;
        }
        prefiller_->run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, int(start),
                        static_cast<float*>(prefill_logits_.p), taps_ptr, rows);
        if (drafter_) {
          drafter_->commit_slot_prefill(s, static_cast<const ActT*>(prefill_taps_.p), T, int(start));
          drafter_->note_tokens_slot(s, prompt.subspan(start, std::size_t(T)));
        }
        start = target_end;

        if (cache_eligible && root_boundary > L && root_boundary < n && start == std::size_t(root_boundary)) {
          if (root_boundary > scatter_from) {
            bk::scatter_full_kv(pool_.scratch_k_full.p, pool_.scratch_v_full.p, pool_.max_ctx, scatter_from,
                                root_boundary - scatter_from, pool_.kv_fp8, s,
                                static_cast<const std::int32_t*>(pool_.page_table.p), pool_.max_pages_per_slot,
                                pool_.k_full_pool.p, pool_.v_full_pool.p, stream_);
            scatter_from = root_boundary;
          }
          CUDA_CHECK(cudaStreamSynchronize(stream_));
          evictions += snapshot_and_publish(s, identity, prompt.subspan(0, std::size_t(root_boundary)), true);
        }
      }
      if (!encoded.empty()) CUDA_CHECK(cudaStreamSynchronize(stream_));

      if (n > scatter_from) {
        bk::scatter_full_kv(pool_.scratch_k_full.p, pool_.scratch_v_full.p, pool_.max_ctx, scatter_from,
                            n - scatter_from, pool_.kv_fp8, s,
                            static_cast<const std::int32_t*>(pool_.page_table.p), pool_.max_pages_per_slot,
                            pool_.k_full_pool.p, pool_.v_full_pool.p, stream_);
      }
      pool_.set_slot_pos(s, n, stream_);
    }

    // Initialize per-slot sampling state and sample first_token.
    SlotMeta& meta = slot_meta_[s];
    meta.params = to_head_params(sampling);
    meta.rng.seed =
        sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    meta.rng.counter = 0;
    meta.forced.assign(sampling.forced.begin(), sampling.forced.end());
    meta.forced_idx = 0;
    meta.max_pos = worst_tokens;
    meta.remaining_emissions = std::max(0, eff_max - 1);
    meta.logit_masker = sampling.logit_masker;

    auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
    const float* first_logits = apply_slot_mask(static_cast<const float*>(prefill_logits_.p), meta.logit_masker);
    head::sample(first_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params, meta.rng, sampled, stream_);
    CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (recorded_restore_events) {
      float restore_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&restore_ms, restore_ev_start_, restore_ev_stop_));
      restore_seconds = double(restore_ms) * 1e-3;
    }
    first_token = static_cast<const std::int32_t*>(sampled_host_.data())[0];
    if (!meta.forced.empty()) {
      first_token = meta.forced[0];
      meta.forced_idx = 1;
    }
    if (meta.logit_masker) meta.logit_masker->advance(first_token);

    // Stream is synchronized: publish end-of-prefill checkpoint BEFORE any decode/speculation occurs!
    if (cache_eligible && L < n) {
      evictions += snapshot_and_publish(s, identity, prompt, root_boundary == n);
    }

    if (drafter_) {
      drafter_->set_slot_seed(s, splitmix64(meta.rng.seed ^ splitmix64(meta.rng.counter++)));
      drafter_->note_tokens_slot(s, std::span<const std::int32_t>(&first_token, 1));
    }

    stats.prompt_tokens = n;
    stats.reused_tokens = L;
    stats.reuse = (L > 0) ? Reuse::PagedPrefix : Reuse::None;
    stats.restored_bytes = restored_bytes;
    stats.restore_seconds = restore_seconds;
    stats.cache_evictions = evictions;
    stats.image_tokens = image_tokens;
    stats.prefill_seconds = seconds_since(t0);
    last_prefill_logits_valid_ = true;
    slot_committed = true;
    return s;
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<std::int32_t> out_tokens) override {
    const int B = int(inputs.size());
    if (B == 0) return;
    if (B > options_.max_batch) throw std::invalid_argument("step_slots: batch size exceeds max_batch");
    if (int(out_tokens.size()) < B) throw std::invalid_argument("step_slots: out_tokens too small");

    std::lock_guard<std::mutex> lock(mutex_);
    // Pass 1 (read-only): validate every slot, token id, and KV page capacity before mutating any state.
    std::vector<bool> seen(static_cast<std::size_t>(options_.max_batch), false);
    int total_add_pages = 0;
    for (int i = 0; i < B; ++i) {
      const int s = inputs[i].slot;
      if (s < 0 || s >= options_.max_batch || !pool_.slot_in_use[s])
        throw std::invalid_argument("step_slots: invalid or inactive slot");
      if (seen[static_cast<std::size_t>(s)])
        throw std::invalid_argument("step_slots: duplicate slot in batch");
      seen[static_cast<std::size_t>(s)] = true;
      ops::validate_token_ids_i32(&inputs[i].token, 1, VOCAB);
      const int pos = pool_.slot_pos_host[s];
      if (pos >= options_.max_ctx) throw std::runtime_error("step_slots: slot reached max_ctx");
      const int needed = (pos + 1 + bk::kPageSize - 1) / bk::kPageSize;
      if (needed > pool_.max_pages_per_slot) throw std::runtime_error("step_slots: paged KV pool exhausted");
      total_add_pages += pool_.additional_pages_needed(s, pos + 1);
    }
    if (pool_.pages.free_count() + pool_.pages.evictable_count() < total_add_pages) {
      throw std::runtime_error("step_slots: paged KV pool exhausted");
    }
    // Pass 2: evict cached pages if needed and allocate KV pages only after all inputs and capacity are verified.
    if (total_add_pages > 0 && pool_.pages.free_count() < total_add_pages) {
      prefix_cache_.ensure_free_pages(pool_.pages, total_add_pages);
    }
    for (int i = 0; i < B; ++i) {
      const int s = inputs[i].slot;
      const int pos = pool_.slot_pos_host[s];
      if (!pool_.ensure_pages(s, pos + 1, stream_))
        throw std::runtime_error("step_slots: paged KV pool exhausted");
    }

    const float* logits = nullptr;
    const ActT* taps = nullptr;
    int B_cap = B;
    std::vector<int> old_pos(static_cast<std::size_t>(B), 0);
    for (int i = 0; i < B; ++i) old_pos[std::size_t(i)] = pool_.slot_pos_host[inputs[i].slot];
    if (drafter_) {
      // Keep drafter taps & position synchronized even when legacy 1-token step_slots() is called!
      std::vector<std::int64_t> toks(static_cast<std::size_t>(B), 0);
      std::vector<std::int32_t> sids(static_cast<std::size_t>(B), -1), wlims(static_cast<std::size_t>(B), 1);
      for (int i = 0; i < B; ++i) {
        toks[std::size_t(i)] = inputs[i].token;
        sids[std::size_t(i)] = inputs[i].slot;
      }
      auto vout = decoder_->run_verify(B, 1, toks.data(), sids.data(), wlims.data(), B_cap, stream_);
      logits = vout.logits;
      taps = vout.taps;
      std::vector<std::int32_t> ones(std::size_t(B_cap), 1);
      CUDA_CHECK(cudaMemcpyAsync(drafter_->accept_scratch().commit, ones.data(), std::size_t(B_cap) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream_));
      decoder_->rollback_verify(B_cap, 1, drafter_->accept_scratch().commit, stream_);
    } else {
      logits = decoder_->run_step(inputs, stream_);
    }
    last_step_batch_ = B;
    for (int i = 0; i < B; ++i) pool_.slot_pos_host[inputs[i].slot] += 1;

    // Fast path: all greedy with identical banned tokens and no active logit masker -> single batched argmax launch.
    bool all_greedy = true;
    const head::BannedTokens& b0 = slot_meta_[inputs[0].slot].params.banned;
    for (int i = 0; i < B; ++i) {
      const SlotMeta& sm = slot_meta_[inputs[i].slot];
      const head::SamplingParams& sp = sm.params;
      if (sp.temperature > 0.0f || !same_banned(sp.banned, b0) || (sm.logit_masker && sm.logit_masker->is_active())) {
        all_greedy = false;
        break;
      }
    }

    auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
    if (all_greedy) {
      head::argmax(logits, B, UNPADDED_VOCAB, UNPADDED_VOCAB, sampled, stream_, b0);
    } else {
      for (int i = 0; i < B; ++i) {
        SlotMeta& meta = slot_meta_[inputs[i].slot];
        const float* row_logits = apply_slot_mask(logits + std::size_t(i) * UNPADDED_VOCAB, meta.logit_masker);
        head::sample(row_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params, meta.rng, sampled + i, stream_);
      }
    }

    if (drafter_) {
      std::vector<kd::BatchSlotCommitMeta> cmetas(static_cast<std::size_t>(B));
      for (int i = 0; i < B; ++i) {
        cmetas[std::size_t(i)] = {inputs[i].slot, old_pos[std::size_t(i)], 1, 0};
      }
      drafter_->commit_batch(cmetas, taps, 1);
    }

    CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, std::size_t(B) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto* host_out = static_cast<const std::int32_t*>(sampled_host_.data());

    const BatchSpecTraceConfig trace_cfg = snapshot_spec_trace_cfg();
    std::vector<BatchSpecSlotTrace> traces;
    if (trace_cfg.on_step) traces.resize(static_cast<std::size_t>(B));

    for (int i = 0; i < B; ++i) {
      SlotMeta& meta = slot_meta_[inputs[i].slot];
      std::int32_t tok = host_out[i];
      if (meta.forced_idx < meta.forced.size()) tok = meta.forced[meta.forced_idx++];
      if (meta.logit_masker) meta.logit_masker->advance(tok);
      if (meta.remaining_emissions > 0) --meta.remaining_emissions;
      out_tokens[i] = tok;
      if (drafter_) drafter_->note_tokens_slot(inputs[i].slot, std::span<const std::int32_t>(&tok, 1));
      if (trace_cfg.on_step) {
        auto& tr = traces[std::size_t(i)];
        tr.slot = inputs[i].slot;
        tr.pos = old_pos[std::size_t(i)];
        tr.anchor = inputs[i].token;
        tr.temperature = meta.params.temperature;
        tr.top_k = meta.params.top_k;
        tr.top_p = meta.params.top_p;
        for (int bi = 0; bi < meta.params.banned.count; ++bi) tr.banned.push_back(meta.params.banned.ids[bi]);
        tr.w_cap = 1;
        tr.W_batch = 1;
        tr.accepted = 0;
        tr.next_token = tok;
        tr.commit = 1;
        if (trace_cfg.capture_target_logits && logits) {
          tr.target_logits.resize(std::size_t(UNPADDED_VOCAB));
          CUDA_CHECK(cudaMemcpy(tr.target_logits.data(), logits + std::size_t(i) * UNPADDED_VOCAB,
                                std::size_t(UNPADDED_VOCAB) * sizeof(float), cudaMemcpyDeviceToHost));
        }
        if (trace_cfg.capture_target_taps && taps) {
          std::vector<ActT> raw(std::size_t(NUM_TAPS) * H);
          CUDA_CHECK(cudaMemcpy(raw.data(), taps + std::size_t(i) * NUM_TAPS * H, raw.size() * sizeof(ActT),
                                cudaMemcpyDeviceToHost));
          tr.target_taps.resize(raw.size());
          for (std::size_t k = 0; k < raw.size(); ++k) tr.target_taps[k] = act::host_to_float(raw[k]);
        }
      }
    }
    if (trace_cfg.on_step) trace_cfg.on_step(traces);
  }

  void step_slots_multi(std::span<const BatchSlotStep> inputs, std::span<BatchSlotResult> out) override {
    const int B = int(inputs.size());
    if (B == 0) return;
    if (B > options_.max_batch) throw std::invalid_argument("step_slots_multi: batch size exceeds max_batch");
    if (int(out.size()) < B) throw std::invalid_argument("step_slots_multi: out too small");
    if (!drafter_) {
      Generator::step_slots_multi(inputs, out);
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const BatchSpecTraceConfig trace_cfg = snapshot_spec_trace_cfg();

    // 1. Validate slots, token ids, and guarantee mandatory 1-token progress page for EVERY active slot first.
    std::vector<bool> in_batch(std::size_t(options_.max_batch), false);
    for (int b = 0; b < B; ++b) {
      const int s = inputs[b].slot;
      if (s < 0 || s >= options_.max_batch || !pool_.slot_in_use[s])
        throw std::invalid_argument("step_slots_multi: invalid or inactive slot");
      if (in_batch[std::size_t(s)]) throw std::invalid_argument("step_slots_multi: duplicate slot in batch");
      in_batch[std::size_t(s)] = true;
      ops::validate_token_ids_i32(&inputs[b].token, 1, VOCAB);
      const int pos = pool_.slot_pos_host[s];
      if (pos >= options_.max_ctx) throw std::runtime_error("step_slots_multi: slot reached max_ctx");
      const int add = pool_.additional_pages_needed(s, pos + 1);
      if (add > 0 && pool_.pages.free_count() < add) {
        prefix_cache_.ensure_free_pages(pool_.pages, add);
      }
      if (!pool_.ensure_pages(s, pos + 1, stream_))
        throw std::runtime_error("step_slots_multi: paged KV pool exhausted");
    }

    // 2. Run batched drafter across all B active slots simultaneously.
    std::vector<BatchDrafter::DraftRequest> dreqs(static_cast<std::size_t>(B));
    std::vector<BatchSpecSlotTrace> traces;
    if (trace_cfg.on_step) traces.resize(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) {
      const int s = inputs[b].slot;
      dreqs[std::size_t(b)] = {s, inputs[b].token, slot_meta_[s].params.temperature};
      if (trace_cfg.on_step) {
        auto& tr = traces[std::size_t(b)];
        tr.slot = s;
        tr.pos = pool_.slot_pos_host[s];
        tr.anchor = inputs[b].token;
        tr.temperature = slot_meta_[s].params.temperature;
        tr.top_k = slot_meta_[s].params.top_k;
        tr.top_p = slot_meta_[s].params.top_p;
        for (int bi = 0; bi < slot_meta_[s].params.banned.count; ++bi)
          tr.banned.push_back(slot_meta_[s].params.banned.ids[bi]);
        tr.has_draft = true;
        tr.draft_seed = drafter_->slot_seed(s);
        tr.draft_draws = drafter_->slot_draws(s);
        tr.slot_extra_ids = drafter_->slot_extra_ids(s);
      }
    }
    const BatchDrafter::BatchDraftOutput dout = drafter_->draft_batch(dreqs);
    std::vector<std::int32_t> h_drafts(std::size_t(B) * kd::GAMMA);
    std::vector<float> h_conf(std::size_t(B) * kd::GAMMA);
    CUDA_CHECK(cudaMemcpyAsync(h_drafts.data(), dout.tokens, h_drafts.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_conf.data(), dout.conf, h_conf.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (trace_cfg.on_step) {
      std::vector<float> h_qstats(std::size_t(B) * kd::GAMMA * 2);
      CUDA_CHECK(cudaMemcpy(h_qstats.data(), dout.qstats, h_qstats.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int b = 0; b < B; ++b) {
        auto& tr = traces[std::size_t(b)];
        for (int g = 0; g < kd::GAMMA; ++g) {
          tr.draft_tokens[g] = h_drafts[std::size_t(b) * kd::GAMMA + g];
          tr.draft_conf[g] = h_conf[std::size_t(b) * kd::GAMMA + g];
          tr.draft_qstats[g][0] = h_qstats[(std::size_t(b) * kd::GAMMA + g) * 2 + 0];
          tr.draft_qstats[g][1] = h_qstats[(std::size_t(b) * kd::GAMMA + g) * 2 + 1];
        }
        if (trace_cfg.capture_candidates) {
          const std::size_t ncand = std::size_t(kd::GAMMA) * kd::CANDIDATES;
          tr.draft_cand_idx.resize(ncand);
          tr.draft_cand_logits.resize(ncand);
          CUDA_CHECK(cudaMemcpy(tr.draft_cand_idx.data(), dout.cand_idx + std::size_t(b) * ncand,
                                ncand * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(tr.draft_cand_logits.data(), dout.cand_logits + std::size_t(b) * ncand,
                                ncand * sizeof(float), cudaMemcpyDeviceToHost));
        }
      }
    }

    // 3-5. Plan exact per-slot verify widths w_cap[b] <= lead + 1 (respecting confidence, remaining emissions,
    //      max_ctx, reserved lifetime page ceiling, and aggregate spec_page_budget) and pad ONLY the shared launch
    //      shape W_batch in {1, 2, 4, 8}.
    std::vector<std::int32_t> slot_ids(std::size_t(B), -1);
    std::vector<int> old_pos(std::size_t(B), 0);
    std::vector<VerifySlotInput> slot_plan_inputs(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) {
      const int s = inputs[b].slot;
      slot_ids[std::size_t(b)] = s;
      const int pos = pool_.slot_pos_host[s];
      old_pos[std::size_t(b)] = pos;
      const SlotMeta& meta = slot_meta_[s];

      int lead = 0;
      if (meta.logit_masker != nullptr) {
        lead = 0;  // Constrained decoding (#830 JSON/JSON-Schema FSM): 1-token-at-a-time masked sampling.
      } else if (meta.forced_idx < meta.forced.size() && options_.spec_threshold <= 0.f) {
        lead = kd::GAMMA;
      } else {
        while (lead < kd::GAMMA && h_conf[std::size_t(b) * kd::GAMMA + lead] > options_.spec_threshold) ++lead;
      }
      if (trace_cfg.on_step) traces[std::size_t(b)].lead = lead;

      const int max_pages_res = (meta.max_pos + bk::kPageSize - 1) / bk::kPageSize;
      const int max_pos_res = max_pages_res * bk::kPageSize;
      slot_plan_inputs[std::size_t(b)] = VerifySlotInput{lead,
                                                         meta.remaining_emissions,
                                                         pos,
                                                         options_.max_ctx,
                                                         max_pos_res,
                                                         int(pool_.slot_pages[s].size()),
                                                         inputs[b].max_tokens};
    }

    int other_reserved = 0;
    for (int s = 0; s < options_.max_batch; ++s) {
      if (!pool_.slot_in_use[s] || in_batch[std::size_t(s)]) continue;
      const int mp = (slot_meta_[s].max_pos + bk::kPageSize - 1) / bk::kPageSize;
      const int cp = int(pool_.slot_pages[s].size());
      if (mp > cp) other_reserved += (mp - cp);
    }
    const int avail_pages = pool_.pages.free_count() + prefix_cache_.reclaimable_pages_excluding(pool_.pages, 0);
    const int spec_page_budget = std::max(0, avail_pages - other_reserved);

    const VerifyWidthPlan width_plan =
        plan_batch_verify_widths(slot_plan_inputs, max_verify_width_, bk::kPageSize, spec_page_budget);
    std::vector<std::int32_t> w_cap(std::size_t(B), 1);
    for (int b = 0; b < B; ++b) w_cap[std::size_t(b)] = width_plan.w_cap[std::size_t(b)];
    const int W_batch = width_plan.W_batch;
    const int total_extra = width_plan.total_extra_pages;

    if (total_extra > 0) {
      if (pool_.pages.free_count() < total_extra) {
        prefix_cache_.ensure_free_pages(pool_.pages, total_extra);
      }
      for (int b = 0; b < B; ++b) {
        if (!pool_.ensure_pages(slot_ids[std::size_t(b)], old_pos[std::size_t(b)] + w_cap[std::size_t(b)], stream_)) {
          throw std::runtime_error("step_slots_multi: planned page allocation failed");
        }
      }
    }

    // 6. Populate verify input tokens [B * W_batch] and execute batched target verification.
    std::vector<std::int64_t> v_tokens(std::size_t(B) * W_batch, 0);
    for (int b = 0; b < B; ++b) {
      v_tokens[std::size_t(b) * W_batch + 0] = inputs[b].token;
      for (int k = 0; k < w_cap[std::size_t(b)] - 1; ++k) {
        v_tokens[std::size_t(b) * W_batch + 1 + k] = h_drafts[std::size_t(b) * kd::GAMMA + k];
      }
    }

    int B_cap = B;
    const BatchDecoder::VerifyOutput vout =
        decoder_->run_verify(B, W_batch, v_tokens.data(), slot_ids.data(), w_cap.data(), B_cap, stream_);

    // 7. Heterogeneous acceptance (fast batched greedy path, or general mixed greedy / forced / exact sampled path).
    const BatchDrafter::AcceptScratch& ascr = drafter_->accept_scratch();
    bool all_greedy_uniform = true;
    const head::BannedTokens& b0 = slot_meta_[slot_ids[0]].params.banned;
    for (int b = 0; b < B; ++b) {
      const SlotMeta& meta = slot_meta_[slot_ids[std::size_t(b)]];
      if (meta.params.temperature > 0.f || !same_banned(meta.params.banned, b0) || meta.forced_idx < meta.forced.size() ||
          meta.logit_masker != nullptr) {
        all_greedy_uniform = false;
        break;
      }
    }

    std::vector<std::int32_t> h_acc(std::size_t(B), 0);
    std::vector<std::int32_t> h_next(std::size_t(B), 0);
    std::vector<std::int32_t> h_commit(std::size_t(B_cap), 1);

    if (all_greedy_uniform) {
      head::argmax(vout.logits, B * W_batch, UNPADDED_VOCAB, UNPADDED_VOCAB, ascr.target_argmax, stream_, b0);
      kd::batch_greedy_accept(ascr.target_argmax, dout.tokens, decoder_->slot_w_limit_dev(), B, W_batch, ascr.accepted,
                              ascr.next, ascr.commit, stream_);
      // Pad commit_counts for [B, B_cap) with 1 (ignored since slot_ids == -1) and run rollback on device.
      decoder_->rollback_verify(B_cap, W_batch, ascr.commit, stream_);
      CUDA_CHECK(cudaMemcpyAsync(h_acc.data(), ascr.accepted, std::size_t(B) * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaMemcpyAsync(h_next.data(), ascr.next, std::size_t(B) * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      for (int b = 0; b < B; ++b) h_commit[std::size_t(b)] = h_acc[std::size_t(b)] + 1;
    } else {
      for (int b = 0; b < B; ++b) {
        const int s = slot_ids[std::size_t(b)];
        SlotMeta& meta = slot_meta_[s];
        const int wl = w_cap[std::size_t(b)];
        const float* slot_logits = vout.logits + std::size_t(b) * W_batch * UNPADDED_VOCAB;

        if (meta.forced_idx < meta.forced.size()) {
          // Teacher-forced replay: accept drafts matching forced sequence, next = forced[forced_idx + acc].
          int acc = 0;
          while (acc < wl - 1 && meta.forced_idx + std::size_t(acc) < meta.forced.size() &&
                 h_drafts[std::size_t(b) * kd::GAMMA + acc] == meta.forced[meta.forced_idx + std::size_t(acc)]) {
            ++acc;
          }
          std::int32_t next_tok = 0;
          if (meta.forced_idx + std::size_t(acc) < meta.forced.size()) {
            next_tok = meta.forced[meta.forced_idx + std::size_t(acc)];
          } else {
            const float* row_logits = apply_slot_mask(slot_logits + std::size_t(acc) * UNPADDED_VOCAB, meta.logit_masker);
            head::argmax(row_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, ascr.target_argmax, stream_, meta.params.banned);
            CUDA_CHECK(cudaMemcpyAsync(&next_tok, ascr.target_argmax, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                       stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
          }
          h_acc[std::size_t(b)] = acc;
          h_next[std::size_t(b)] = next_tok;
          h_commit[std::size_t(b)] = acc + 1;
          meta.forced_idx = std::min(meta.forced.size(), meta.forced_idx + std::size_t(acc + 1));
        } else if (meta.logit_masker != nullptr) {
          // Constrained decoding (#830): wl == 1; apply bitmask before greedy/sampled selection.
          std::int32_t next_tok = 0;
          const float* row_logits = apply_slot_mask(slot_logits, meta.logit_masker);
          head::sample(row_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params, meta.rng, ascr.target_argmax,
                       stream_);
          CUDA_CHECK(cudaMemcpyAsync(&next_tok, ascr.target_argmax, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                     stream_));
          CUDA_CHECK(cudaStreamSynchronize(stream_));
          h_acc[std::size_t(b)] = 0;
          h_next[std::size_t(b)] = next_tok;
          h_commit[std::size_t(b)] = 1;
        } else if (meta.params.temperature <= 0.f) {
          head::argmax(slot_logits, wl, UNPADDED_VOCAB, UNPADDED_VOCAB, ascr.target_argmax, stream_, meta.params.banned);
          std::int32_t h_targ[kd::BLOCK] = {};
          CUDA_CHECK(cudaMemcpyAsync(h_targ, ascr.target_argmax, std::size_t(wl) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToHost, stream_));
          CUDA_CHECK(cudaStreamSynchronize(stream_));
          int acc = 0;
          while (acc < wl - 1 && h_drafts[std::size_t(b) * kd::GAMMA + acc] == h_targ[acc]) ++acc;
          h_acc[std::size_t(b)] = acc;
          h_next[std::size_t(b)] = h_targ[acc];
          h_commit[std::size_t(b)] = acc + 1;
        } else {
          // Exact speculative rejection sampling against slot's own temperature, top-k, top-p, bans, RNG, and proposal q!
          if (wl == 1) {
            std::int32_t next_tok = 0;
            head::sample(slot_logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params, meta.rng, ascr.target_argmax,
                         stream_);
            CUDA_CHECK(cudaMemcpyAsync(&next_tok, ascr.target_argmax, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                       stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            h_acc[std::size_t(b)] = 0;
            h_next[std::size_t(b)] = next_tok;
            h_commit[std::size_t(b)] = 1;
          } else {
            const std::int32_t* slot_cidx = dout.cand_idx + std::size_t(b) * kd::GAMMA * kd::CANDIDATES;
            const float* slot_clog = dout.cand_logits + std::size_t(b) * kd::GAMMA * kd::CANDIDATES;
            kd::scatter_slot_draft_logits(slot_cidx, slot_clog, UNPADDED_VOCAB, ascr.draft_logits, stream_);
            head::sampling_probs(slot_logits, wl, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params, ascr.probs, stream_);
            const std::uint64_t rng_seed = meta.rng.seed;
            const std::uint64_t rng_cnt_u = meta.rng.counter;
            head::uniform(wl - 1, meta.rng, ascr.uniforms, stream_);
            kd::sampled_accept(ascr.probs, ascr.draft_logits, UNPADDED_VOCAB, dout.tokens + std::size_t(b) * kd::GAMMA,
                               wl - 1, meta.params.temperature, dout.qstats + std::size_t(b) * kd::GAMMA * 2,
                               ascr.uniforms, ascr.ws, ascr.resid, ascr.one_res, stream_);
            const std::uint64_t rng_cnt_resid = meta.rng.counter;
            head::sample_from_probs(ascr.resid, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.rng, &ascr.one_res->next,
                                    stream_);
            kd::clear_slot_draft_logits(slot_cidx, UNPADDED_VOCAB, ascr.draft_logits, stream_);
            kd::AcceptResult res_h{};
            CUDA_CHECK(cudaMemcpyAsync(&res_h, ascr.one_res, sizeof(res_h), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (trace_cfg.on_step && trace_cfg.capture_sampled_details) {
              auto& tr = traces[std::size_t(b)];
              tr.has_sampled_details = true;
              tr.accept_rng_seed = rng_seed;
              tr.accept_rng_counter_u = rng_cnt_u;
              tr.accept_rng_counter_resid = rng_cnt_resid;
              tr.uniforms.resize(std::size_t(wl - 1));
              tr.target_probs.resize(std::size_t(wl) * UNPADDED_VOCAB);
              tr.residual_probs.resize(std::size_t(UNPADDED_VOCAB));
              CUDA_CHECK(cudaMemcpy(tr.uniforms.data(), ascr.uniforms, std::size_t(wl - 1) * sizeof(float),
                                    cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(tr.target_probs.data(), ascr.probs,
                                    std::size_t(wl) * UNPADDED_VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(tr.residual_probs.data(), ascr.resid, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                                    cudaMemcpyDeviceToHost));
            }
            h_acc[std::size_t(b)] = res_h.accepted;
            h_next[std::size_t(b)] = res_h.next;
            h_commit[std::size_t(b)] = res_h.accepted + 1;
          }
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(ascr.commit, h_commit.data(), std::size_t(B_cap) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream_));
      decoder_->rollback_verify(B_cap, W_batch, ascr.commit, stream_);
    }

    // 8. Commit exact accepted tap rows (including W=1 single-row commits!) into BatchDrafter.
    std::vector<kd::BatchSlotCommitMeta> cmetas(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) {
      cmetas[std::size_t(b)] = {slot_ids[std::size_t(b)], old_pos[std::size_t(b)], h_commit[std::size_t(b)], 0};
    }
    drafter_->commit_batch(cmetas, vout.taps, W_batch);

    if (trace_cfg.on_step && (trace_cfg.capture_target_logits || trace_cfg.capture_target_taps)) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      for (int b = 0; b < B; ++b) {
        auto& tr = traces[std::size_t(b)];
        const int wl = w_cap[std::size_t(b)];
        const int cm = h_commit[std::size_t(b)];
        if (trace_cfg.capture_target_logits && vout.logits) {
          tr.target_logits.resize(std::size_t(wl) * UNPADDED_VOCAB);
          CUDA_CHECK(cudaMemcpy(tr.target_logits.data(), vout.logits + std::size_t(b) * W_batch * UNPADDED_VOCAB,
                                std::size_t(wl) * UNPADDED_VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
        }
        if (trace_cfg.capture_target_taps && vout.taps) {
          std::vector<ActT> raw(std::size_t(cm) * NUM_TAPS * H);
          CUDA_CHECK(cudaMemcpy(raw.data(), vout.taps + std::size_t(b) * W_batch * NUM_TAPS * H,
                                raw.size() * sizeof(ActT), cudaMemcpyDeviceToHost));
          tr.target_taps.resize(raw.size());
          for (std::size_t k = 0; k < raw.size(); ++k) tr.target_taps[k] = act::host_to_float(raw[k]);
        }
      }
    }

    // 9. Update host state, slot-local dynamic shortlists, and output results.
    for (int b = 0; b < B; ++b) {
      const int s = slot_ids[std::size_t(b)];
      const int acc = h_acc[std::size_t(b)];
      const int commit = h_commit[std::size_t(b)];
      pool_.slot_pos_host[s] += commit;
      SlotMeta& meta = slot_meta_[s];
      meta.remaining_emissions = std::max(0, meta.remaining_emissions - commit);

      BatchSlotResult res{};
      for (int k = 0; k < acc; ++k) res.tokens[k] = h_drafts[std::size_t(b) * kd::GAMMA + k];
      res.tokens[acc] = h_next[std::size_t(b)];
      res.count = commit;
      res.spec_rounds = (w_cap[std::size_t(b)] > 1) ? 1 : 0;
      res.spec_accepted = acc;
      out[b] = res;

      if (trace_cfg.on_step) {
        auto& tr = traces[std::size_t(b)];
        tr.w_cap = w_cap[std::size_t(b)];
        tr.W_batch = W_batch;
        tr.accepted = acc;
        tr.next_token = h_next[std::size_t(b)];
        tr.commit = commit;
      }

      if (meta.logit_masker) {
        for (int k = 0; k < res.count; ++k) meta.logit_masker->advance(res.tokens[k]);
      }
      drafter_->note_tokens_slot(s, std::span<const std::int32_t>(res.tokens, std::size_t(res.count)));
    }
    if (trace_cfg.on_step) trace_cfg.on_step(traces);
  }

  void release_slot(int slot) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_) CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (slot >= 0 && slot < int(slot_meta_.size())) slot_meta_[slot].logit_masker.reset();
    pool_.free_slot(slot);
    if (drafter_ && slot >= 0 && slot < options_.max_batch) drafter_->reset_slot(slot);
  }

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

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    return generate_from_slots(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
  }

  bool supports_embeddings() const override { return true; }
  int embedding_dim() const override { return HIDDEN; }
  int num_tap_layers() const override { return prefiller_ ? prefiller_->num_visits() : NUM_VISITS; }
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
    // Stateless zero-KV embedding prefill: never allocates a batch slot or touches pool_/prefix_cache_.
    return prefiller_->embed(req, options_.max_ctx, platos_key_gpu_.get());
  }

 private:
  struct SlotMeta {
    head::SamplingParams params{};
    head::SamplerRng rng{};
    std::vector<std::int32_t> forced;
    std::size_t forced_idx = 0;
    int max_pos = 0;
    std::shared_ptr<TokenMasker> logit_masker;
    int remaining_emissions = 0;
  };

  const float* apply_slot_mask(const float* row_logits, const std::shared_ptr<TokenMasker>& masker) {
    if (!masker || !masker->is_active()) return row_logits;
    const std::size_t words = (std::size_t(UNPADDED_VOCAB) + 31u) / 32u;
    if (mask_host_.size() < words) mask_host_.resize(words);
    std::fill(mask_host_.begin(), mask_host_.begin() + std::ptrdiff_t(words), 0u);
    if (!masker->fill_mask(mask_host_.data(), words)) return row_logits;
    if (!mask_dev_.p || mask_dev_.bytes < words * sizeof(std::uint32_t)) {
      mask_dev_ = DeviceBuffer(words * sizeof(std::uint32_t));
    }
    if (!masked_logits_.p || masked_logits_.bytes < prefill_logits_.bytes) {
      masked_logits_ = DeviceBuffer(prefill_logits_.bytes);
    }
    CUDA_CHECK(cudaMemcpyAsync(mask_dev_.p, mask_host_.data(), words * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, stream_));
    head::apply_bitmask(row_logits, static_cast<const std::uint32_t*>(mask_dev_.p), UNPADDED_VOCAB,
                        static_cast<float*>(masked_logits_.p), stream_);
    return static_cast<const float*>(masked_logits_.p);
  }

  int effective_max_new(int prompt_tokens, int max_new_tokens, const Sampling& sampling) const {
    int eff = std::max(0, max_new_tokens);
    if (!sampling.forced.empty()) eff = std::min(eff, int(sampling.forced.size()));
    const int room = std::max(0, options_.max_ctx - prompt_tokens);
    return std::min(eff, room);
  }

  int active_reserved_growth_pages() const {
    int reserved = 0;
    for (int s = 0; s < options_.max_batch; ++s) {
      if (!pool_.slot_in_use[s]) continue;
      const int max_pages = (slot_meta_[s].max_pos + bk::kPageSize - 1) / bk::kPageSize;
      const int cur_pages = int(pool_.slot_pages[s].size());
      if (max_pages > cur_pages) reserved += (max_pages - cur_pages);
    }
    return reserved;
  }

  StateIdentity build_identity(const Prompt& prompt) const {
    StateIdentity id{};
    id.model_sha256 = model_sha256_;
    id.act_dtype = INGINE_ACT_FP16 ? 1u : 0u;
    id.dense_fp8 = options_.dense_fp8;
    id.kv_format = options_.kv_fp8 ? 1u : 0u;
    id.kv_cache_mode = static_cast<std::uint32_t>(options_.kv_cache_mode);
    id.loop_passes = options_.loop_passes;
    id.max_ctx = options_.max_ctx;
    id.prefill_chunk = prefill_chunk_;
    id.spec_enabled = options_.spec;
    id.cache_namespace = prompt.cache_namespace;
    return id;
  }

  std::uint64_t restore_recurrent_state(int s, const PrefixCheckpointGpu& gpu) {
    std::uint64_t bytes = 0;
    const int r = gpu.ring_rows;
    const std::size_t ring_visit_pitch = std::size_t(WINDOW) * KV_DIM * sizeof(ActT);
    const std::size_t ring_row_width = std::size_t(r) * KV_DIM * sizeof(ActT);
    ActT* dst_kr = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(s) * pool_.ring_slot_stride;
    ActT* dst_vr = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(s) * pool_.ring_slot_stride;
    std::int64_t* dst_rp = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(s) * pool_.ring_pos_slot_stride;

    if (r < WINDOW) {
      CUDA_CHECK(cudaMemsetAsync(dst_rp, 0xff, pool_.ring_pos_slot_stride * sizeof(std::int64_t), stream_));
    }
    if (r > 0) {
      CUDA_CHECK(cudaMemcpy2DAsync(dst_kr, ring_visit_pitch, gpu.k_ring.p, ring_row_width, ring_row_width,
                                   NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(dst_vr, ring_visit_pitch, gpu.v_ring.p, ring_row_width, ring_row_width,
                                   NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
      const std::size_t rp_pitch = std::size_t(WINDOW) * sizeof(std::int64_t);
      const std::size_t rp_width = std::size_t(r) * sizeof(std::int64_t);
      CUDA_CHECK(cudaMemcpy2DAsync(dst_rp, rp_pitch, gpu.ring_pos.p, rp_width, rp_width, NUM_SLIDING_VISITS,
                                   cudaMemcpyDeviceToDevice, stream_));
      bytes += gpu.k_ring.bytes + gpu.v_ring.bytes + gpu.ring_pos.bytes;
    }

    const std::size_t conv_kv_bytes = pool_.conv_kv_slot_stride * sizeof(float);
    const std::size_t conv_h_bytes = pool_.conv_h_slot_stride * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(pool_.conv_k.p) + std::size_t(s) * pool_.conv_kv_slot_stride,
                               gpu.conv_k.p, conv_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(pool_.conv_v.p) + std::size_t(s) * pool_.conv_kv_slot_stride,
                               gpu.conv_v.p, conv_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(pool_.conv_attn.p) + std::size_t(s) * pool_.conv_h_slot_stride,
                               gpu.conv_attn.p, conv_h_bytes, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(pool_.conv_mlp.p) + std::size_t(s) * pool_.conv_h_slot_stride,
                               gpu.conv_mlp.p, conv_h_bytes, cudaMemcpyDeviceToDevice, stream_));
    bytes += 2 * conv_kv_bytes + 2 * conv_h_bytes;
    return bytes;
  }

  int snapshot_and_publish(int s, const StateIdentity& identity, std::span<const std::int32_t> prefix_tokens,
                           bool is_root) {
    const int L = int(prefix_tokens.size());
    if (L <= 0 || prefix_cache_.vram_budget_bytes() == 0) return 0;
    if (prefix_cache_.touch_exact(identity, prefix_tokens)) return 0;

    const int F = L / bk::kPageSize;
    const int tail = L % bk::kPageSize;
    const int ring_rows = std::min(L, WINDOW);
    const std::size_t es = pool_.kv_fp8 ? 1 : 2;
    const std::size_t ring_bytes = std::size_t(NUM_SLIDING_VISITS) * std::size_t(ring_rows) * KV_DIM * sizeof(ActT);
    const std::size_t rp_bytes = std::size_t(NUM_SLIDING_VISITS) * std::size_t(ring_rows) * sizeof(std::int64_t);
    const std::size_t conv_kv_bytes = pool_.conv_kv_slot_stride * sizeof(float);
    const std::size_t conv_h_bytes = pool_.conv_h_slot_stride * sizeof(float);
    const std::size_t tail_bytes = tail > 0 ? (std::size_t(NUM_FULL_VISITS) * std::size_t(tail) * KV_DIM * es) : 0;
    const std::size_t logits_bytes = prefill_logits_.bytes;
    const std::size_t draft_bytes = drafter_ ? drafter_->estimate_checkpoint_bytes(L) : 0;
    const std::size_t total_private_bytes =
        2 * ring_bytes + rp_bytes + 2 * conv_kv_bytes + 2 * conv_h_bytes + 2 * tail_bytes + logits_bytes + draft_bytes;
    if (total_private_bytes > prefix_cache_.vram_budget_bytes()) return 0;

    int evicted = prefix_cache_.ensure_private_vram(pool_.pages, total_private_bytes);
    if (prefix_cache_.stats().vram_used_bytes + total_private_bytes > prefix_cache_.vram_budget_bytes()) {
      return evicted;
    }

    try {
      BatchPrefixEntry<PrefixCheckpointGpu> entry;
      entry.identity = identity;
      entry.tokens.assign(prefix_tokens.begin(), prefix_tokens.end());
      entry.committed_pages.assign(pool_.slot_pages[s].begin(), pool_.slot_pages[s].begin() + F);
      entry.tail_tokens = tail;
      entry.private_vram_bytes = total_private_bytes;
      entry.is_root = is_root;

      PrefixCheckpointGpu& gpu = entry.payload;
      gpu.ring_rows = ring_rows;
      gpu.k_ring = DeviceBuffer(ring_bytes);
      gpu.v_ring = DeviceBuffer(ring_bytes);
      gpu.ring_pos = DeviceBuffer(rp_bytes);
      gpu.conv_k = DeviceBuffer(conv_kv_bytes);
      gpu.conv_v = DeviceBuffer(conv_kv_bytes);
      gpu.conv_attn = DeviceBuffer(conv_h_bytes);
      gpu.conv_mlp = DeviceBuffer(conv_h_bytes);
      if (tail > 0) {
        gpu.tail_k = DeviceBuffer(tail_bytes);
        gpu.tail_v = DeviceBuffer(tail_bytes);
      }
      gpu.logits = DeviceBuffer(logits_bytes);

      const std::size_t ring_visit_pitch = std::size_t(WINDOW) * KV_DIM * sizeof(ActT);
      const std::size_t ring_row_width = std::size_t(ring_rows) * KV_DIM * sizeof(ActT);
      const ActT* src_kr = static_cast<const ActT*>(pool_.k_ring.p) + std::size_t(s) * pool_.ring_slot_stride;
      const ActT* src_vr = static_cast<const ActT*>(pool_.v_ring.p) + std::size_t(s) * pool_.ring_slot_stride;
      const std::int64_t* src_rp =
          static_cast<const std::int64_t*>(pool_.ring_pos.p) + std::size_t(s) * pool_.ring_pos_slot_stride;
      CUDA_CHECK(cudaMemcpy2DAsync(gpu.k_ring.p, ring_row_width, src_kr, ring_visit_pitch, ring_row_width,
                                   NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(gpu.v_ring.p, ring_row_width, src_vr, ring_visit_pitch, ring_row_width,
                                   NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
      const std::size_t rp_pitch = std::size_t(WINDOW) * sizeof(std::int64_t);
      const std::size_t rp_width = std::size_t(ring_rows) * sizeof(std::int64_t);
      CUDA_CHECK(cudaMemcpy2DAsync(gpu.ring_pos.p, rp_width, src_rp, rp_pitch, rp_width, NUM_SLIDING_VISITS,
                                   cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(gpu.conv_k.p,
                                 static_cast<const float*>(pool_.conv_k.p) + std::size_t(s) * pool_.conv_kv_slot_stride,
                                 conv_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(gpu.conv_v.p,
                                 static_cast<const float*>(pool_.conv_v.p) + std::size_t(s) * pool_.conv_kv_slot_stride,
                                 conv_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          gpu.conv_attn.p, static_cast<const float*>(pool_.conv_attn.p) + std::size_t(s) * pool_.conv_h_slot_stride,
          conv_h_bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(gpu.conv_mlp.p,
                                 static_cast<const float*>(pool_.conv_mlp.p) + std::size_t(s) * pool_.conv_h_slot_stride,
                                 conv_h_bytes, cudaMemcpyDeviceToDevice, stream_));
      if (tail > 0) {
        bk::copy_tail_kv_from_scratch(pool_.scratch_k_full.p, pool_.scratch_v_full.p, pool_.max_ctx, F * bk::kPageSize,
                                      tail, pool_.kv_fp8, gpu.tail_k.p, gpu.tail_v.p, stream_);
      }
      CUDA_CHECK(cudaMemcpyAsync(gpu.logits.p, prefill_logits_.p, logits_bytes, cudaMemcpyDeviceToDevice, stream_));
      if (drafter_) {
        gpu.has_draft = true;
        (void)drafter_->save_slot_checkpoint(s, L, gpu.draft);
      }
      // Synchronize stream before publishing committed pages & snapshot buffers into shared cache.
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      prefix_cache_.record_d2d_park(total_private_bytes);
      auto [published, pub_evicted] = prefix_cache_.publish(pool_.pages, std::move(entry));
      (void)published;
      evicted += pub_evicted;
    } catch (...) {
      // Graceful fallback under VRAM pressure: release cached entries, never fail live request.
      cudaGetLastError();
      evicted += prefix_cache_.clear(pool_.pages, true);
    }
    return evicted;
  }

  void init() {
    prefill_chunk_ = std::min(options_.prefill_chunk, options_.max_ctx);
    if (prefill_chunk_ < 1) throw std::invalid_argument("prefill_chunk must be positive");

    const std::optional<std::size_t> vram_limit_bytes =
        options_.vram_limit_mb > 0 ? std::optional<std::size_t>(std::size_t(options_.vram_limit_mb) << 20)
                                   : std::nullopt;
    auto remaining_lease_opt = [&]() -> std::optional<std::size_t> {
      if (!vram_limit_bytes.has_value()) return std::nullopt;
      std::size_t free_b = 0, total_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      const std::size_t used_since_entry = free_at_entry_ > free_b ? (free_at_entry_ - free_b) : 0;
      const std::size_t rem = *vram_limit_bytes > used_since_entry ? (*vram_limit_bytes - used_since_entry) : 0;
      return std::min(free_b, rem);
    };
    auto effective_free_bytes = [&]() -> std::size_t {
      const auto rem = remaining_lease_opt();
      if (rem.has_value()) return *rem;
      std::size_t free_b = 0, total_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      return free_b;
    };
    auto enforce_lease_now = [&](const char* phase) {
      if (!vram_limit_bytes.has_value()) return;
      std::size_t free_b = 0, total_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      const std::size_t used_since_entry = free_at_entry_ > free_b ? (free_at_entry_ - free_b) : 0;
      if (used_since_entry > *vram_limit_bytes) {
        throw std::runtime_error(std::string("BatchGenerator: VRAM lease cap exceeded after ") + phase + " (used " +
                                 std::to_string(used_since_entry >> 20) + " MiB > cap " +
                                 std::to_string(*vram_limit_bytes >> 20) + " MiB)");
      }
    };
    enforce_lease_now("weights");

    {
      const std::size_t est_prefill = estimate_prefiller_peak_bytes(prefill_chunk_, UNPADDED_VOCAB, options_.max_batch);
      const VramLeaseCheck chk = check_vram_lease_fit(remaining_lease_opt(), est_prefill);
      if (!chk.ok) {
        throw std::runtime_error("BatchGenerator: VRAM lease cap exceeded before Prefiller allocation (peak_need=" +
                                 std::to_string(chk.planned_bytes >> 20) + " MiB, remaining=" +
                                 std::to_string(chk.remaining_bytes >> 20) + " MiB)");
      }
    }
    if (options_.vision) vision_ = std::make_unique<VisionEncoder>(options_.model_dir, stream_);
    prefiller_ = std::make_unique<Prefiller>(weights_, ablation_, prefill_chunk_, stream_);
    if (!options_.platos_key.empty()) {
      platos_key_gpu_ = PlatosKeyGpu::load(options_.platos_key, options_.platos_key_metadata, HIDDEN, stream_);
      platos_key_gpu_->key()->validate_binding(model_id(), model_family(), HIDDEN, prefiller_->num_visits());
    }
    chunk_tokens_ = DeviceBuffer(std::size_t(prefill_chunk_) * sizeof(std::int32_t));
    prefill_logits_ = DeviceBuffer(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    sampled_dev_ = DeviceBuffer(std::size_t(options_.max_batch) * sizeof(std::int32_t));

    // Pin model identity digest over config.json + optional ablation file (fail-closed across models/ablations).
    std::array<std::uint8_t, 64> id_buf{};
    const auto cfg_h = sha256_file_or_zero(options_.model_dir / "config.json");
    std::memcpy(id_buf.data(), cfg_h.data(), 32);
    if (!options_.ablation.empty()) {
      const auto abl_h = sha256_file_or_zero(options_.ablation);
      std::memcpy(id_buf.data() + 32, abl_h.data(), 32);
    }
    model_sha256_ = sha256_bytes(id_buf.data(), id_buf.size());
    enforce_lease_now("weights+prefiller");

    // Size and allocate BatchDrafter BEFORE BatchStatePool page reservation so VRAM headroom is exact.
    max_verify_width_ = 1;
    std::size_t decoder_reserve_bytes =
        BatchDecoder::estimate_bytes(options_.max_batch, 1, options_.dense_fp8, false);
    if (options_.spec) {
      const std::size_t free_b = effective_free_bytes();
      max_verify_width_ = 8;
      // Only step down from W=8 if dense verify activations + rollback + drafter would crowd out VRAM.
      const std::size_t min_kv_headroom = std::size_t(2048) << 20;
      for (int cand_w : {8, 4, 2}) {
        const std::size_t est = BatchDecoder::estimate_bytes(options_.max_batch, cand_w, options_.dense_fp8, true);
        if (free_b > est + min_kv_headroom) {
          max_verify_width_ = cand_w;
          break;
        }
        max_verify_width_ = cand_w;
      }
      drafter_ = std::make_unique<BatchDrafter>(options_.model_dir, weights_, options_.max_batch, options_.max_ctx,
                                                max_verify_width_, stream_, BatchDrafter::SHORTLIST_ROWS,
                                                remaining_lease_opt());
      prefill_taps_ = DeviceBuffer(std::size_t(prefill_chunk_) * NUM_TAPS * H * sizeof(ActT));
      decoder_reserve_bytes =
          BatchDecoder::estimate_bytes(options_.max_batch, max_verify_width_, options_.dense_fp8, true);
      enforce_lease_now("BatchDrafter");
    }

    pool_.init(options_.max_batch, options_.max_ctx, options_.kv_fp8, options_.max_kv_tokens, decoder_reserve_bytes,
               remaining_lease_opt());
    enforce_lease_now("BatchStatePool");
    slot_meta_.resize(options_.max_batch);

    {
      const VramLeaseCheck chk = check_vram_lease_fit(remaining_lease_opt(), decoder_reserve_bytes);
      if (!chk.ok) {
        throw std::runtime_error("BatchGenerator: VRAM lease cap exceeded before BatchDecoder allocation (need " +
                                 std::to_string(chk.planned_bytes >> 20) + " MiB, remaining " +
                                 std::to_string(chk.remaining_bytes >> 20) + " MiB)");
      }
    }
    decoder_ = std::make_unique<BatchDecoder>(weights_, ablation_, pool_, options_.max_batch, max_verify_width_,
                                              drafter_ != nullptr, vram_limit_bytes, free_at_entry_);
    decoder_->capture_buckets(stream_);
    CUDA_CHECK(cudaEventCreate(&restore_ev_start_));
    CUDA_CHECK(cudaEventCreate(&restore_ev_stop_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    enforce_lease_now("BatchDecoder+graphs");

    // Size private recurrent/tail/logits VRAM budget for prefix checkpoints while reserving headroom for lazy
    // verify CUDA graphs so neither runtime checkpoints nor lazy graph capture can ever exceed the lease.
    const std::size_t lazy_graph_reserve = std::size_t(drafter_ ? 256 : 64) << 20;
    if (!options_.prefix_cache || options_.park_mb == 0) {
      prefix_cache_.set_vram_budget(0);
    } else if (options_.park_mb > 0) {
      const std::size_t cap = std::size_t(options_.park_mb) << 20;
      const VramLeaseCheck chk = check_vram_lease_fit(remaining_lease_opt(), cap + lazy_graph_reserve);
      if (!chk.ok) {
        throw std::runtime_error("BatchGenerator: explicit --park-mb=" + std::to_string(options_.park_mb) +
                                 " + lazy graph reserve exceeds remaining VRAM lease (" +
                                 std::to_string(chk.remaining_bytes >> 20) + " MiB)");
      }
      prefix_cache_.set_vram_budget(cap);
    } else {
      const std::size_t free_b = effective_free_bytes();
      const std::size_t reserve = std::max(lazy_graph_reserve, std::min<std::size_t>(std::size_t(1024) << 20, free_b / 4));
      const std::size_t avail = free_b > reserve ? free_b - reserve : 0;
      const std::size_t auto_cap = std::min<std::size_t>(std::size_t(2048) << 20, avail / 2);
      prefix_cache_.set_vram_budget(auto_cap);
    }
  }

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
    const std::int64_t negative =
        std::count_if(prompt.ids.begin(), prompt.ids.end(), [](std::int32_t id) { return id < 0; });
    bool covered = negative == tokens;
    for (const PromptImage& image : prompt.images) {
      for (int k = 0; covered && k < image.tokens(); ++k) covered = prompt.ids[std::size_t(image.pos + k)] < 0;
    }
    if (!covered) throw std::invalid_argument("negative token ids of the prompt are not exactly its image tokens");
    return int(tokens);
  }

 public:
  bool copy_last_prefill_logits(std::vector<float>& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_prefill_logits_valid_ || !prefill_logits_.p) return false;
    out.resize(UNPADDED_VOCAB);
    CUDA_CHECK(cudaMemcpy(out.data(), prefill_logits_.p, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return true;
  }

  bool copy_last_step_logits(int lane_idx, std::vector<float>& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lane_idx < 0 || lane_idx >= last_step_batch_ || !decoder_ || !decoder_->last_logits()) return false;
    out.resize(UNPADDED_VOCAB);
    CUDA_CHECK(cudaMemcpy(out.data(), decoder_->last_logits() + std::size_t(lane_idx) * UNPADDED_VOCAB,
                          std::size_t(UNPADDED_VOCAB) * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
  }

 private:
  RuntimeOptions options_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t restore_ev_start_ = nullptr;
  cudaEvent_t restore_ev_stop_ = nullptr;
  Weights weights_;
  Ablation ablation_;
  std::array<std::uint8_t, 32> model_sha256_{};
  int prefill_chunk_ = 3072;
  int max_verify_width_ = 1;
  std::size_t free_at_entry_ = 0;
  std::unique_ptr<VisionEncoder> vision_;
  std::shared_ptr<const PlatosKeyGpu> platos_key_gpu_;
  std::unique_ptr<Prefiller> prefiller_;
  std::unique_ptr<BatchDrafter> drafter_;
  BatchStatePool pool_;
  BatchPrefixCache<PrefixCheckpointGpu> prefix_cache_;
  std::unique_ptr<BatchDecoder> decoder_;
  DeviceBuffer chunk_tokens_, prefill_logits_, prefill_taps_, sampled_dev_;
  DeviceBuffer masked_logits_, mask_dev_;
  std::vector<std::uint32_t> mask_host_;
  PinnedHostBuffer sampled_host_;
  std::vector<SlotMeta> slot_meta_;
  bool last_prefill_logits_valid_ = false;
  int last_step_batch_ = 0;
  mutable std::mutex mutex_;
};

}  // namespace

void set_batch_prefill_splits_for_test(std::vector<int> splits) {
  std::lock_guard<std::mutex> lock(g_test_splits_mu);
  g_test_prefill_splits = std::move(splits);
}

bool copy_batch_prefill_logits_for_test(const Generator& gen, std::vector<float>& out) {
  const auto* bg = dynamic_cast<const BatchGenerator*>(&gen);
  return bg ? bg->copy_last_prefill_logits(out) : false;
}

bool copy_batch_step_logits_for_test(const Generator& gen, int lane_idx, std::vector<float>& out) {
  const auto* bg = dynamic_cast<const BatchGenerator*>(&gen);
  return bg ? bg->copy_last_step_logits(lane_idx, out) : false;
}

void set_batch_spec_trace_hook_for_test(BatchSpecTraceConfig cfg) {
  std::lock_guard<std::mutex> lock(g_spec_trace_mu);
  g_spec_trace_cfg = std::move(cfg);
}

void clear_batch_spec_trace_hook_for_test() {
  std::lock_guard<std::mutex> lock(g_spec_trace_mu);
  g_spec_trace_cfg = BatchSpecTraceConfig{};
}

std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options) {
  return std::make_unique<BatchGenerator>(options);
}

std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  return std::make_unique<BatchGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model
