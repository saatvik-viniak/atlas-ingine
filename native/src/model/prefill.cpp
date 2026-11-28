// model.Model.prefill for one chunk of T tokens, wired as fast.FastDecoder.decode (block states B0, B1, B2[p],
// E[p], partial / write pairs, message mixes) with T rows instead of M:
//   embed -> per visit: [message mix] -> attention read (mix) [-> message gates] -> taps -> projections
//   (cuBLASLt over the BF16 rows, FP8 rows dequantized per visit as weights.dequant) -> attn_prep (q/rel post-
//   processing, key/value convs, cache or sliding rows) -> rel_logits -> attention (+ gate) -> wo -> attention
//   conv -> MLP read (mix) [-> gate] -> router GEMM + top-8 -> grouped experts -> combine -> MLP conv
//   -> final read (mix) -> taps -> head of the last row.
// The mixes and the embedding are ModelCore's decode kernels (per-row CTAs, any row count). Ablation of the
// attention / MLP writes happens inside the mixes that consume them, as in the decode step.
#include "model/prefill.h"

#include "core/device.h"
#include "core/ops.h"
#include "ingine/generator.h"
#include "kernels/dense/dense.h"
#include "kernels/prefill/prefill.h"
#include "model/expert_cache.h"
#include "model/paged_kv.h"
#include "model/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace pk = kernels::prefill;
namespace dense = kernels::dense;

namespace {

constexpr int H = HIDDEN;

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const ActT* bf16p(const Tensor& t) { return static_cast<const ActT*>(t.data); }
const ActT* bf16p(const void* p) { return static_cast<const ActT*>(p); }

// fast.Buffers.blocks
std::vector<const float*> blocks_of(int l, int p, int last_pass, const float* B0, const float* B1, float* const* B2,
                                    float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : last_pass]};
  return {B0, B1, B2[last_pass], E[last_pass]};
}

}  // namespace

Prefiller::Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream, int loop_passes,
                     LoopCdMode loopcd_mode, float loopcd_omega)
    : w_(w),
      abl_(abl),
      max_chunk_(max_chunk),
      loop_passes_(loop_passes),
      loopcd_mode_(loopcd_mode),
      loopcd_omega_(loopcd_omega),
      stream_(stream),
      gemm_(std::make_unique<pk::Gemm>()) {
  if (max_chunk < 1) throw std::invalid_argument("Prefiller: max_chunk must be positive");
  if (loop_passes_ < 1 || loop_passes_ > MAX_PASSES)
    throw std::invalid_argument("Prefiller: loop_passes must be in 1.." + std::to_string(MAX_PASSES));
  const int L = WINDOW + max_chunk;
  kpos_rows_ = (L + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile * pk::kAttnKeyTile;
  if (kpos_rows_ / pk::kAttnKeyTile > pk::kAttnMaxTiles)
    throw std::invalid_argument("Prefiller: max_chunk above " +
                                std::to_string(pk::kAttnMaxTiles * pk::kAttnKeyTile - WINDOW));
  const std::size_t C = std::size_t(max_chunk);
  auto alloc = [&](std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  };
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto bf16 = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };
  auto align256 = [](std::size_t n) { return (n + 255) / 256 * 256; };

  // Persistent across a visit (block states, residual accumulators, message buffer, tiny helpers).
  B0_ = f32(C * H), B1_ = f32(C * H);
  for (int p = 0; p < loop_passes_; ++p) B2_[p] = f32(C * H), E_[p] = f32(C * H);
  hid_ = f32(C * H), part_ = f32(C * H), y_ = f32(C * H), y2_ = f32(C * H);
  msgs_ = f32(4 * C * H);
  kpos_ = i32(kpos_rows_);
  bounds_ = static_cast<int4*>(alloc(std::size_t(kpos_rows_ / pk::kAttnKeyTile) * sizeof(int4)));
  g16_ = bf16(C * GATE_RANK);
  wscratch_ = (w.dense_fp8 || w.attn_q5_k || w.dense_q8)
                  ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM))
                  : nullptr;
  if (loopcd_mode_ == LoopCdMode::LogitsAdaptive) {
    logits_weak_ = f32(UNPADDED_VOCAB);
    loopcd_reduce_ = f32(dense::kLoopCdReduceFloatsPerRow);
  }
  zero_conv_ = f32(std::size_t(CONV_K - 1) * H);
  ids_ = static_cast<std::int64_t*>(alloc(C * 8));

  // Disjoint-lifetime phase arena: Attention phase (mixf/xin -> proj/q/rel/krows/vrows/attn_out/wo_out)
  // and MLP phase (mixm/xm/router/routing/act/down/exl3_scratch/moe_sum) never overlap in a visit,
  // and final read (xf_) runs after the visit loop. Message-gate scratch (gin_, fpre_, gpre_) starts
  // past both phases' live gate-call prefixes (mixf_+xin_ and mixm_+xm_+routing), saving ~113 KB/tok
  // (~30% of Prefiller scratch, allowing prefill_chunk=1024 inside 6 GB VRAM caps).
  bool any_exl3_w13 = false;
  for (int l = 0; l < LAYERS; ++l) any_exl3_w13 |= is_exl3(w.expert_formats[l]);
  const bool need_exl3_scratch = (w.expert_cache && any_exl3_w13);

  std::size_t off_a = 0;
  auto take_a = [&](std::size_t bytes) {
    const std::size_t r = off_a;
    off_a = align256(off_a + bytes);
    return r;
  };
  const std::size_t off_mixf = take_a(3 * C * H * 4);
  const std::size_t off_xin = take_a(3 * C * H * 2);
  const std::size_t a_prefix_end = off_a;

  std::size_t off_b = 0;
  auto take_b = [&](std::size_t bytes) {
    const std::size_t r = off_b;
    off_b = align256(off_b + bytes);
    return r;
  };
  const std::size_t off_mixm = take_b(C * H * 4);
  const std::size_t off_xm = take_b(C * H * 2);
  const std::size_t off_xf = take_b(C * H * 2);
  const std::size_t off_router = take_b(C * EXPERTS * 4);
  const std::size_t off_wts = take_b(C * TOP_K * 4);
  const std::size_t off_idx = take_b(C * TOP_K * 4);
  const std::size_t off_counts = take_b(EXPERTS * 4);
  const std::size_t off_tiles_n = take_b(4);
  const std::size_t off_pair_tok = take_b(C * TOP_K * 4);
  const std::size_t off_pair_dst = take_b(C * TOP_K * 4);
  const std::size_t off_tiles = take_b(std::size_t(pk::moe_tiles_capacity(max_chunk)) * sizeof(int4));
  const std::size_t b_prefix_end = off_b;

  // Phase A tail (dead during attention gates() and during entire Phase B)
  const std::size_t off_rel = take_a(C * HEADS * REL_EXTENT * 2);
  const std::size_t off_proj = take_a(C * PROJ_ROWS_FULL * 2);
  const std::size_t off_q = take_a(C * HEADS * HEAD_DIM * 2);
  const std::size_t off_rel16 = take_a(C * HEADS * D_REL * 2);
  const std::size_t off_attn_out = take_a(C * HEADS * HEAD_DIM * 2);
  const std::size_t off_wo_out = take_a(C * H * 2);
  const std::size_t off_krows = take_a(std::size_t(kpos_rows_) * KV_DIM * 2);
  const std::size_t off_vrows = take_a(std::size_t(kpos_rows_) * KV_DIM * 2);

  // Phase B tail (dead during MLP gates() and during entire Phase A)
  const std::size_t off_act = take_b(C * TOP_K * INTER * 2);
  const std::size_t off_down = take_b(C * TOP_K * H * 2);
  const std::size_t off_exl3 = need_exl3_scratch ? take_b(C * TOP_K * H * 2) : off_down;
  const std::size_t off_moe_sum = take_b(C * H * 2);

  // Message-gate scratch lives past BOTH live prefixes so neither attention gates() nor MLP gates() aliases live data.
  std::size_t off_g = std::max(a_prefix_end, b_prefix_end);
  auto take_g = [&](std::size_t bytes) {
    const std::size_t r = off_g;
    off_g = align256(off_g + bytes);
    return r;
  };
  const std::size_t off_gin = take_g(3 * C * 2 * H * 2);
  const std::size_t off_fpre = take_g(C * H * 4);
  const std::size_t off_gpre = take_g(C * H * 4);

  auto* arena = static_cast<char*>(alloc(std::max({off_a, off_b, off_g})));
  auto at_f32 = [&](std::size_t off) { return reinterpret_cast<float*>(arena + off); };
  auto at_bf16 = [&](std::size_t off) { return reinterpret_cast<ActT*>(arena + off); };
  auto at_i32 = [&](std::size_t off) { return reinterpret_cast<int*>(arena + off); };

  mixf_ = at_f32(off_mixf);
  xin_ = at_bf16(off_xin);
  rel_ = at_bf16(off_rel);
  proj_ = at_bf16(off_proj);
  q_ = at_bf16(off_q);
  rel16_ = at_bf16(off_rel16);
  attn_out_ = at_bf16(off_attn_out);
  wo_out_ = at_bf16(off_wo_out);
  krows_ = at_bf16(off_krows);
  vrows_ = at_bf16(off_vrows);

  mixm_ = at_f32(off_mixm);
  xm_ = at_bf16(off_xm);
  xf_ = at_bf16(off_xf);
  router_logits_ = at_f32(off_router);
  wts_ = at_f32(off_wts);
  idx_ = at_i32(off_idx);
  counts_ = at_i32(off_counts);
  tiles_n_ = at_i32(off_tiles_n);
  pair_token_ = at_i32(off_pair_tok);
  pair_dst_ = at_i32(off_pair_dst);
  tiles_ = reinterpret_cast<int4*>(arena + off_tiles);
  act_ = at_bf16(off_act);
  down_ = at_bf16(off_down);
  exl3_scratch_ = at_bf16(off_exl3);
  moe_sum_ = at_bf16(off_moe_sum);

  gin_ = at_bf16(off_gin);
  fpre_ = at_f32(off_fpre);
  gpre_ = at_f32(off_gpre);

  for (auto& b : storage_) b.fill(0);  // e.g. gathered sliding rows past ring + chunk: never keys, kept finite
}

Prefiller::~Prefiller() = default;

std::size_t Prefiller::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

const float* const* Prefiller::block_table(const std::vector<const float*>& blocks) {
  auto it = tables_.find(blocks);
  if (it == tables_.end()) {
    DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
    if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
    it = tables_.emplace(blocks, std::move(b)).first;
  }
  return static_cast<const float* const*>(it->second.p);
}

void Prefiller::run(State& st, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
                    std::span<const EmbedRows> rows, const kernels::persona::PersonaDeviceTable* hooks) {
  if (T < 1 || T > max_chunk_) throw std::invalid_argument("Prefiller::run: T outside 1..max_chunk");
  if (pos0 < 0 || pos0 + T > st.max_ctx)
    throw std::invalid_argument("context " + std::to_string(pos0 + T) + " exceeds max_ctx " +
                                std::to_string(st.max_ctx));
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;  // one [T, H] slab of a [G, T, H] buffer
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);
  ActT* tap_out = taps ? static_cast<ActT*>(taps->data) : nullptr;

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 ActT* out_bf16, float* out_f32, const kernels::persona::DepthHookSite* depth_hook = nullptr) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;  // the detector of the same visit
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    if (hooks && depth_hook) dense::mix_hooked(p, depth_hook, nullptr, pos0, false, s);
    else dense::mix(p, false, s);
  };
  // B rows of a Dense matrix as BF16: the matrix itself, or its FP8 rows dequantized (weights.dequant)
  auto bf16_rows = [&](const Dense& d) -> const ActT* {
    if (d.blocked()) {
      dense::dequant_blocked(
          dense::Matrix{d.w.data, nullptr, d.rows(), false, static_cast<const std::int64_t*>(d.blk_off.data),
                        static_cast<const std::int32_t*>(d.blk_soff.data),
                        static_cast<const std::uint16_t*>(d.q8_scale.data)},
          d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (d.q5_k()) {
      dense::dequant_q5_k_rows(d.w.data, d.rows(), d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  // message_gate for G gates: h_g, msg_g fp32 [T, H] -> out bf16 [G, T, H] = bf16(rms_norm(gated) * norm_w[g])
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, ActT* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  pk::widen_ids(tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  if (w.embed_q8()) {
    dense::embed_q8(ids_, T, w.embed.data, f32p(w.embed_norm), B0_, false, s);
  } else if (w.embed_q4()) {
    auto desc = ops::make_inkling_embed_q4_desc(T, VOCAB, false);
    desc.token_bounds = ops::TokenBoundsMode::PreValidated;
    const ops::EmbedQ4RmsNormArgs args{ids_, w.embed.data, f32p(w.embed_norm), B0_};
    ops::run_embed_q4_rmsnorm(desc, args, s);
  } else {
    dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  }
  // image tokens: the vision tower's rows take the place of the placeholder's embedding (model.forward's
  // masked_scatter into the normed embeddings); every later read of the embedding block sees them
  for (const EmbedRows& r : rows) {
    if (r.row < 0 || r.count < 0 || r.row + r.count > T) throw std::invalid_argument("Prefiller::run: embedding rows outside the chunk");
    pk::put_rows(r.src, r.count, B0_ + std::size_t(r.row) * H, s);
  }
  if (st.plan.passes > loop_passes_) {
    throw std::invalid_argument("Prefiller::run: state loop_passes exceeds Prefiller capacity");
  }
  const VisitPlan& plan = st.plan;
  const int last_pass = plan.last_pass();
  int prev_v = -1;
  int tap = 0;

  for (int v = 0; v < plan.num_visits; ++v) {
    const int l = plan.visits[std::size_t(v)].layer, p = plan.visits[std::size_t(v)].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, last_pass, B0_, B1_, B2_, E_);

    // ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[last_pass] : hid_;
      base = part_, write = y2_;
    }
    // LoopCD-Hidden (arXiv:2610.02185 Eq. 2): match Decoder::decode identically across all T prompt rows.
    if (l == LOOP_END && loopcd_mode_ == LoopCdMode::Hidden && last_pass > 0) {
      dense::loopcd_hidden_from_residual(E_[last_pass], base, write, E_[last_pass - 1], T, loopcd_omega_, false, s);
      base = E_[last_pass];
      write = nullptr;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = abl_.active ? (mlp_dirs + std::size_t(pv) * H) : mlp_dirs;
    const float* w_prev = abl_.active ? (mlp_w + 4 * std::size_t(pv)) : mlp_w;
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_,
          hooks ? &hooks->depth_msg[v] : nullptr);
      if (i == 0) write = nullptr;
    }
    // ---- attention read(s)
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_,
          hooks ? &hooks->depth_attn[v] : nullptr);
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr, hooks ? &hooks->depth_attn[v] : nullptr);
    }
    if (tap < 3 && v == TAP_VISITS[tap]) {
      if (tap_out) pk::store_tap(hid, T, tap_out + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
      ++tap;
    }

    // ---- projections (fast.proj_layout): one GEMM for a single input, else per input segment
    const int R = proj_rows(layer.full);
    const ActT* wp = bf16_rows(layer.wproj);
    const bool single = !layer.full && !msgs;
    if (single) {
      gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
    } else {
      struct Part {
        int begin, end, input;
      };
      const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                            {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                            {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                            {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                            {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
      for (const Part& part : parts) {
        if (part.begin >= R) break;
        gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                   R, false, T, part.end - part.begin, H, s);
      }
    }

    // ---- attention
    pk::AttnPrepParams ap;
    ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
    ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
    ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
    ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
    ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
    ap.q = q_, ap.rel = rel16_;
    pk::AttnParams at;
    at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
    if (layer.full) {
      const int fi = st.full_slot(v);
      if (st.paged_kv) {
        st.paged_kv->ensure_write_pages(fi, pos0, T, s);
        ap.k_out = nullptr, ap.v_out = nullptr;
        ap.k_pages = st.paged_kv->k_table(fi);
        ap.v_pages = st.paged_kv->v_table(fi);
        ap.page_shift = PagedKvCache::kPageShift;
        ap.out_fp8 = false, ap.out_q4 = true;
        pk::attn_prep(ap, s);
        at.k = nullptr, at.v = nullptr, at.kv_fp8 = false, at.kv_q4 = true, at.kpos = nullptr, at.bounds = nullptr,
        at.L = pos0 + T;
        at.window = 0;
      } else {
        const std::size_t off = std::size_t(pos0) * st.full_row_bytes();
        void* kc = st.k_full_at(fi).data;
        void* vc = st.v_full_at(fi).data;
        ap.k_out = static_cast<char*>(kc) + off, ap.v_out = static_cast<char*>(vc) + off;
        ap.out_fp8 = st.kv_fp8, ap.out_q4 = st.kv_q4;
        pk::attn_prep(ap, s);
        at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.kv_q4 = st.kv_q4, at.kpos = nullptr, at.bounds = nullptr,
        at.L = pos0 + T;
        at.window = 0;
      }
    } else {
      const int ri = plan.ring_index[std::size_t(v)];
      auto* k_ring = static_cast<ActT*>(st.k_ring_at(ri).data);
      auto* v_ring = static_cast<ActT*>(st.v_ring_at(ri).data);
      std::int64_t* ring_pos = st.ring_pos_at(ri);
      // keys = cat(ring, chunk) as model.prefill reads them, before the ring takes the chunk's last rows
      pk::ring_gather(k_ring, v_ring, ring_pos, pos0, T, krows_, vrows_, kpos_, kpos_rows_, s);
      ap.k_out = krows_ + std::size_t(WINDOW) * KV_DIM, ap.v_out = vrows_ + std::size_t(WINDOW) * KV_DIM;
      ap.out_fp8 = false, ap.out_q4 = false;
      pk::attn_prep(ap, s);
      pk::ring_update(krows_, vrows_, pos0, T, k_ring, v_ring, ring_pos, s);
      pk::key_tile_bounds(kpos_, WINDOW + T, bounds_, s);
      at.k = krows_, at.v = vrows_, at.kv_fp8 = false, at.kv_q4 = false, at.kpos = kpos_, at.bounds = bounds_,
      at.L = WINDOW + T;
      at.window = WINDOW;
    }
    pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
    pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
    pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
    if (layer.full && st.paged_kv) {
      st.paged_kv->prefill_attention(st.full_slot(v), at, s);
    } else {
      pk::attention(at, s);
    }

    // ---- output projection + post-attention conv
    const ActT* wo = bf16_rows(layer.wo);
    gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    pk::static_conv(wo_out_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
    pk::conv_state_update(st.conv_attn_at(v), wo_out_, H, T, H, s);
    if (hooks) dense::residual_steer(y_, T, &hooks->steer_attn[v], s);

    // ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    const float* r_attn = abl_.active ? (attn_dirs + std::size_t(v) * H) : attn_dirs;
    const float* w_attn = abl_.active ? (attn_w + 4 * std::size_t(v)) : attn_w;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1, nullptr,
          nullptr, mixm_, hooks ? &hooks->depth_ffn[v] : nullptr);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1,
          f32p(layer.mlp_norm1), xm_, nullptr, hooks ? &hooks->depth_ffn[v] : nullptr);
    }
    if (hooks) dense::tap_ffn_inp(xm_, T, &hooks->ffn_inp_tap[v], nullptr, pos0, s);

    // ---- routed experts (model._moe_grouped) + post-MLP conv
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    if (hooks) pk::moe_topk_hooked(router_logits_, f32p(layer.router_bias), T, pos0, idx_, wts_, counts_, &hooks->router[v], s);
    else pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const Experts& ex = layer.experts;
    const bool exl13 = is_exl3(ex.w13_fmt), exl2 = is_exl3(ex.w2_fmt);
    // EXL3 TM=32 (3 CTAs/SM, zero register spills) beats TM=128 by ~89% on 64K prefill.
    const int moe_tr = (exl13 || exl2) ? 32 : pk::moe_tile_rows(T);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s, moe_tr);
    pk::MoeExperts me{static_cast<int>(ex.w13_fmt),
                      ex.w13.data,
                      exl13 ? ex.w13_rin.data : ex.w13_scale.data,
                      exl13 ? nullptr : f32p(ex.w13_factor),
                      ex.w2.data,
                      exl2 ? ex.w2_rin.data : ex.w2_scale.data,
                      exl2 ? nullptr : f32p(ex.w2_factor),
                      static_cast<int>(ex.w2_fmt),
                      ex.w13_rout.data,
                      ex.w2_rout.data,
                      exl3_scratch_};
    me.tile_rows = moe_tr;
    if (w.expert_cache) {
      const auto eptrs = w.expert_cache->layer_ptrs(l);
      me.w13_ptrs = eptrs.w13;
      me.w13_scale_ptrs = eptrs.w13_scale;
      me.w13_rout_ptrs = eptrs.w13_rout;
      me.w2_ptrs = eptrs.w2;
      me.w2_scale_ptrs = eptrs.w2_scale;
      me.w2_rout_ptrs = eptrs.w2_rout;
      w.expert_cache->run_prefill_layer(l, counts_, s, [&](const std::uint8_t* mask) {
        me.active_mask = mask;
        pk::moe_gate_up(me, xm_, T, work, act_, s);
        if (hooks) pk::apply_expert_hooks_prefill(act_, work, T, pos0, &hooks->expert[v], s, mask);
        pk::moe_down(me, act_, T, work, down_, s);
      });
    } else {
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      if (hooks) pk::apply_expert_hooks_prefill(act_, work, T, pos0, &hooks->expert[v], s);
      pk::moe_down(me, act_, T, work, down_, s);
    }
    pk::moe_combine(down_, wts_, T, moe_sum_, s);
    pk::static_conv(moe_sum_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
    pk::conv_state_update(st.conv_mlp_at(v), moe_sum_, H, T, H, s);
    if (hooks) dense::residual_steer(y2_, T, &hooks->steer_mlp[v], s);
    prev_v = v;
  }

  // ---- final read, taps, head of the last row
  const float* r_last = abl_.active ? (mlp_dirs + std::size_t(prev_v) * H) : mlp_dirs;
  const float* w_last = abl_.active ? (mlp_w + 4 * std::size_t(prev_v)) : mlp_w;
  mix({B0_, B1_, B2_[last_pass], E_[last_pass]}, part_, y2_, r_last, w_last, nullptr,
      f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr, hooks ? &hooks->depth_output : nullptr);
  if (tap_out)
    CUDA_CHECK(cudaMemcpy2DAsync(tap_out + std::size_t(NUM_TAPS - 1) * H, std::size_t(NUM_TAPS) * H * 2, xf_,
                                 std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, s));
  auto blocked_head_mat = [&]() {
    const Dense& u = w.unembed;
    return dense::Matrix{u.w.data, nullptr, UNPADDED_VOCAB, false, static_cast<const std::int64_t*>(u.blk_off.data),
                         static_cast<const std::int32_t*>(u.blk_soff.data),
                         static_cast<const std::uint16_t*>(u.q8_scale.data)};
  };
  if (loopcd_mode_ == LoopCdMode::LogitsAdaptive && last_pass > 0) {
    const int ref_pass = last_pass - 1;
    // Shared-coda approximation (reuses final coda residual part+y2 while substituting earlier loop blocks
    // B2[ref_pass], E[ref_pass] for B2[last_pass], E[last_pass]; unlike arXiv:2610.02185 Eq. 3, avoids running
    // coda layers 18..23 twice). Uses pk::head_logits_dual so prefill FP8 rounding matches pk::head_logits bit-for-bit.
    mix({B0_, B1_, B2_[ref_pass], E_[ref_pass]}, part_, y2_, r_last, w_last, nullptr, f32p(w.final_query), 1,
        f32p(w.final_norm), xm_, nullptr, nullptr);
    if (w.unembed.blocked()) {
      const dense::Matrix m = blocked_head_mat();
      dense::head(xf_ + std::size_t(T - 1) * H, 1, m, logits, false, s);
      dense::head(xm_ + std::size_t(T - 1) * H, 1, m, logits_weak_, false, s);
    } else {
      pk::head_logits_dual(xf_ + std::size_t(T - 1) * H, xm_ + std::size_t(T - 1) * H, w.unembed.w.data,
                           f32p(w.unembed.scale), UNPADDED_VOCAB, logits, logits_weak_, s);
    }
    dense::loopcd_adaptive_combine(logits, logits_weak_, 1, UNPADDED_VOCAB, loopcd_omega_, loopcd_reduce_, s);
  } else if (w.unembed.blocked()) {  // (Q6_K head kept as int8 codes: the block-table GEMV)
    dense::head(xf_ + std::size_t(T - 1) * H, 1, blocked_head_mat(), logits, false, s);
  } else {
    pk::head_logits(xf_ + std::size_t(T - 1) * H, w.unembed.w.data, f32p(w.unembed.scale), UNPADDED_VOCAB, logits, s);
  }
  CUDA_CHECK(cudaGetLastError());
  st.length = pos0 + T;
}

void Prefiller::run_embed_packed(const std::int32_t* d_tokens, std::span<const int> seq_lens,
                                 const std::int32_t* d_seq_offsets, EmbeddingPooling pooling,
                                 std::optional<int> tap_visit, bool l2_normalize, float* d_out_pooled) {
  const int batch = static_cast<int>(seq_lens.size());
  int T = 0;
  for (int L : seq_lens) T += L;
  if (batch < 1 || T < 1 || T > max_chunk_) {
    throw std::invalid_argument("Prefiller::run_embed_packed: invalid packed batch size");
  }
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 ActT* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  auto bf16_rows = [&](const Dense& d) -> const ActT* {
    if (d.blocked()) {
      dense::dequant_blocked(
          dense::Matrix{d.w.data, nullptr, d.rows(), false, static_cast<const std::int64_t*>(d.blk_off.data),
                        static_cast<const std::int32_t*>(d.blk_soff.data),
                        static_cast<const std::uint16_t*>(d.q8_scale.data)},
          d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (d.q5_k()) {
      dense::dequant_q5_k_rows(d.w.data, d.rows(), d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, ActT* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  pk::widen_ids(d_tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  if (w.embed_q8()) {
    dense::embed_q8(ids_, T, w.embed.data, f32p(w.embed_norm), B0_, false, s);
  } else if (w.embed_q4()) {
    auto desc = ops::make_inkling_embed_q4_desc(T, VOCAB, false);
    desc.token_bounds = ops::TokenBoundsMode::PreValidated;
    const ops::EmbedQ4RmsNormArgs args{ids_, w.embed.data, f32p(w.embed_norm), B0_};
    ops::run_embed_q4_rmsnorm(desc, args, s);
  } else {
    dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  }

  const VisitPlan plan = make_visit_plan(loop_passes_);
  const int last_pass = plan.last_pass();
  int prev_v = -1;

  for (int v = 0; v < plan.num_visits; ++v) {
    const int l = plan.visits[std::size_t(v)].layer;
    const int p = plan.visits[std::size_t(v)].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, last_pass, B0_, B1_, B2_, E_);

    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[last_pass] : hid_;
      base = part_, write = y2_;
    }
    if (l == LOOP_END && loopcd_mode_ == LoopCdMode::Hidden && last_pass > 0) {
      dense::loopcd_hidden_from_residual(E_[last_pass], base, write, E_[last_pass - 1], T, loopcd_omega_, false, s);
      base = E_[last_pass];
      write = nullptr;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = abl_.active ? (mlp_dirs + std::size_t(pv) * H) : mlp_dirs;
    const float* w_prev = abl_.active ? (mlp_w + 4 * std::size_t(pv)) : mlp_w;
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }

    // Early-exit tap: matches Prefiller::run's TAP_VISITS hid tensor bit-for-bit, skipping the rest of
    // this visit, all subsequent visits, final_norm, and lm_head.
    if (tap_visit.has_value() && v == *tap_visit) {
      embed_pool_f32_cuda(hid, d_seq_offsets, batch, H, static_cast<int>(pooling), l2_normalize, d_out_pooled, s);
      CUDA_CHECK(cudaGetLastError());
      return;
    }

    if (msgs) {
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    }

    // Batched projections across all T packed tokens.
    const int R = proj_rows(layer.full);
    const ActT* wp = bf16_rows(layer.wproj);
    const bool single = !layer.full && !msgs;
    if (single) {
      gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
    } else {
      struct Part {
        int begin, end, input;
      };
      const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                            {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                            {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                            {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                            {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
      for (const Part& part : parts) {
        if (part.begin >= R) break;
        gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                   R, false, T, part.end - part.begin, H, s);
      }
    }

    // Zero-KV sequence-isolated attn_prep + attention using ephemeral scratch krows_/vrows_ and read-only zero_conv_.
    int off = 0;
    for (int b = 0; b < batch; ++b) {
      const int L = seq_lens[std::size_t(b)];
      pk::AttnPrepParams ap;
      ap.proj = proj_ + std::size_t(off) * R;
      ap.ld = R;
      ap.T = L;
      ap.pos0 = 0;
      ap.full = layer.full;
      ap.q_scale = f32p(layer.q_scale);
      ap.r_scale = f32p(layer.r_scale);
      ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
      ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
      ap.conv_k = zero_conv_;
      ap.conv_v = zero_conv_;
      ap.q = q_ + std::size_t(off) * HEADS * HEAD_DIM;
      ap.rel = rel16_ + std::size_t(off) * HEADS * D_REL;
      ap.k_out = krows_ + std::size_t(off) * KV_DIM;
      ap.v_out = vrows_ + std::size_t(off) * KV_DIM;
      ap.out_fp8 = false;
      ap.out_q4 = false;
      pk::attn_prep(ap, s);
      off += L;
    }
    pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);

    off = 0;
    for (int b = 0; b < batch; ++b) {
      const int L = seq_lens[std::size_t(b)];
      pk::AttnParams at;
      at.q = q_ + std::size_t(off) * HEADS * HEAD_DIM;
      at.k = krows_ + std::size_t(off) * KV_DIM;
      at.v = vrows_ + std::size_t(off) * KV_DIM;
      at.k_pos_base = 0;
      at.kv_fp8 = false;
      at.kv_q4 = false;
      at.kpos = nullptr;
      at.bounds = nullptr;
      at.L = L;
      at.T = L;
      at.pos0 = 0;
      at.window = layer.full ? 0 : WINDOW;
      at.rel = rel_ + std::size_t(off) * HEADS * REL_EXTENT;
      at.proj = proj_ + std::size_t(off) * R;
      at.ld = R;
      at.out = attn_out_ + std::size_t(off) * HEADS * HEAD_DIM;
      pk::attention(at, s);
      off += L;
    }

    // Batched output projection across all T tokens + sequence-isolated zero-state post-attention conv.
    const ActT* wo = bf16_rows(layer.wo);
    gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    off = 0;
    for (int b = 0; b < batch; ++b) {
      const int L = seq_lens[std::size_t(b)];
      pk::static_conv(wo_out_ + std::size_t(off) * H, H, L, H, f32p(layer.attn_conv), zero_conv_,
                      y_ + std::size_t(off) * H, s);
      off += L;
    }

    // Batched MLP mix + gates + router + grouped MoE across all T tokens.
    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    const float* r_attn = abl_.active ? (attn_dirs + std::size_t(v) * H) : attn_dirs;
    const float* w_attn = abl_.active ? (attn_w + 4 * std::size_t(v)) : attn_w;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
    }

    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const Experts& ex = layer.experts;
    const bool exl13 = is_exl3(ex.w13_fmt), exl2 = is_exl3(ex.w2_fmt);
    const int moe_tr = (exl13 || exl2) ? 32 : pk::moe_tile_rows(T);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s, moe_tr);
    pk::MoeExperts me{static_cast<int>(ex.w13_fmt),
                      ex.w13.data,
                      exl13 ? ex.w13_rin.data : ex.w13_scale.data,
                      exl13 ? nullptr : f32p(ex.w13_factor),
                      ex.w2.data,
                      exl2 ? ex.w2_rin.data : ex.w2_scale.data,
                      exl2 ? nullptr : f32p(ex.w2_factor),
                      static_cast<int>(ex.w2_fmt),
                      ex.w13_rout.data,
                      ex.w2_rout.data,
                      exl3_scratch_};
    me.tile_rows = moe_tr;
    if (w.expert_cache) {
      const auto eptrs = w.expert_cache->layer_ptrs(l);
      me.w13_ptrs = eptrs.w13;
      me.w13_scale_ptrs = eptrs.w13_scale;
      me.w13_rout_ptrs = eptrs.w13_rout;
      me.w2_ptrs = eptrs.w2;
      me.w2_scale_ptrs = eptrs.w2_scale;
      me.w2_rout_ptrs = eptrs.w2_rout;
      w.expert_cache->run_prefill_layer(l, counts_, s, [&](const std::uint8_t* mask) {
        me.active_mask = mask;
        pk::moe_gate_up(me, xm_, T, work, act_, s);
        pk::moe_down(me, act_, T, work, down_, s);
      });
    } else {
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      pk::moe_down(me, act_, T, work, down_, s);
    }
    pk::moe_combine(down_, wts_, T, moe_sum_, s);
    off = 0;
    for (int b = 0; b < batch; ++b) {
      const int L = seq_lens[std::size_t(b)];
      pk::static_conv(moe_sum_ + std::size_t(off) * H, H, L, H, f32p(layer.mlp_conv), zero_conv_,
                      y2_ + std::size_t(off) * H, s);
      off += L;
    }
    prev_v = v;
  }

  // Final read (skips unembed / lm_head GEMM completely).
  const float* r_last = abl_.active ? (mlp_dirs + std::size_t(prev_v) * H) : mlp_dirs;
  const float* w_last = abl_.active ? (mlp_w + 4 * std::size_t(prev_v)) : mlp_w;
  mix({B0_, B1_, B2_[last_pass], E_[last_pass]}, part_, y2_, r_last, w_last, nullptr, f32p(w.final_query), 1,
      f32p(w.final_norm), xf_, nullptr);
  embed_pool_act_cuda(reinterpret_cast<const std::uint16_t*>(xf_), kActDType == DType::FP16, d_seq_offsets, batch, H,
                      static_cast<int>(pooling), l2_normalize, d_out_pooled, s);
  CUDA_CHECK(cudaGetLastError());
}

void Prefiller::run_embed_chunk(State& st, const std::int32_t* d_tokens, int T, int pos0,
                                std::optional<int> tap_visit, float* d_out_f32, bool* out_is_act) {
  if (T < 1 || T > max_chunk_) throw std::invalid_argument("Prefiller::run_embed_chunk: T outside 1..max_chunk");
  if (pos0 < 0 || pos0 + T > st.max_ctx) throw std::invalid_argument("Prefiller::run_embed_chunk: exceeds max_ctx");
  const Weights& w = w_;
  const cudaStream_t s = stream_;
  const std::size_t slab = std::size_t(T) * H;
  const float* attn_dirs = f32p(abl_.attn_dirs);
  const float* attn_w = f32p(abl_.attn_w);
  const float* mlp_dirs = f32p(abl_.mlp_dirs);
  const float* mlp_w = f32p(abl_.mlp_w);

  auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                 const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                 ActT* out_bf16, float* out_f32) {
    dense::MixParams p;
    p.blocks = blocks.empty() ? nullptr : block_table(blocks);
    p.nblocks = int(blocks.size());
    p.base = base;
    p.write = write;
    p.abl_r = write ? abl_r : nullptr;
    p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
    p.abl_w = write ? abl_w : nullptr;
    p.partial_out = partial_out;
    p.queries = queries;
    p.Q = Q;
    p.norm_w = norm_w;
    p.out_bf16 = out_bf16;
    p.out_f32 = out_f32;
    p.M = T;
    dense::mix(p, false, s);
  };
  auto bf16_rows = [&](const Dense& d) -> const ActT* {
    if (d.blocked()) {
      dense::dequant_blocked(
          dense::Matrix{d.w.data, nullptr, d.rows(), false, static_cast<const std::int64_t*>(d.blk_off.data),
                        static_cast<const std::int32_t*>(d.blk_soff.data),
                        static_cast<const std::uint16_t*>(d.q8_scale.data)},
          d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (d.q5_k()) {
      dense::dequant_q5_k_rows(d.w.data, d.rows(), d.cols(), wscratch_, s);
      return wscratch_;
    }
    if (!d.fp8()) return bf16p(d.w);
    dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
    return wscratch_;
  };
  auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                   const float* norm_w, ActT* out) {
    pk::gate_inputs(rows, G, T, gin_, s);
    for (int g = 0; g < G; ++g) {
      const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
      gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
      gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                 GATE_RANK, s);
      gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                 GATE_RANK, s);
      pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                     s);
    }
  };

  pk::widen_ids(d_tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
  if (w.embed_q8()) {
    dense::embed_q8(ids_, T, w.embed.data, f32p(w.embed_norm), B0_, false, s);
  } else if (w.embed_q4()) {
    auto desc = ops::make_inkling_embed_q4_desc(T, VOCAB, false);
    desc.token_bounds = ops::TokenBoundsMode::PreValidated;
    const ops::EmbedQ4RmsNormArgs args{ids_, w.embed.data, f32p(w.embed_norm), B0_};
    ops::run_embed_q4_rmsnorm(desc, args, s);
  } else {
    dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);
  }

  const VisitPlan& plan = st.plan;
  const int last_pass = plan.last_pass();
  int prev_v = -1;
  for (int v = 0; v < plan.num_visits; ++v) {
    const int l = plan.visits[std::size_t(v)].layer;
    const int p = plan.visits[std::size_t(v)].pass;
    const Layer& layer = w.layers[l];
    const bool msgs = is_loop_layer(l) && p > 0;
    const bool boundary = l % BLOCK == 0;
    const std::vector<const float*> blocks = blocks_of(l, p, last_pass, B0_, B1_, B2_, E_);

    float* hid;
    const float* base;
    const float* write;
    if (v == 0) {
      hid = B0_, base = B0_, write = nullptr;
    } else if (l == LOOP_START && p > 0) {
      hid = B1_, base = B1_, write = nullptr;
    } else {
      hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[last_pass] : hid_;
      base = part_, write = y2_;
    }
    if (l == LOOP_END && loopcd_mode_ == LoopCdMode::Hidden && last_pass > 0) {
      dense::loopcd_hidden_from_residual(E_[last_pass], base, write, E_[last_pass - 1], T, loopcd_omega_, false, s);
      base = E_[last_pass];
      write = nullptr;
    }
    const int pv = prev_v >= 0 ? prev_v : 0;
    const float* r_prev = abl_.active ? (mlp_dirs + std::size_t(pv) * H) : mlp_dirs;
    const float* w_prev = abl_.active ? (mlp_w + 4 * std::size_t(pv)) : mlp_w;
    if (msgs) {
      const int i = l - LOOP_START;
      const float* src_base = i == 0 ? part_ : E_[p - 1];
      const float* src_write = i == 0 ? y2_ : nullptr;
      mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
          f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
      if (i == 0) write = nullptr;
    }
    const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
    const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
    float* partial_out = hid != base ? hid : nullptr;
    if (msgs) {
      mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
    } else {
      mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
          f32p(layer.attn_norm_mix), xin_, nullptr);
    }
    if (tap_visit.has_value() && v == *tap_visit) {
      CUDA_CHECK(cudaMemcpyAsync(d_out_f32, hid, slab * sizeof(float), cudaMemcpyDeviceToDevice, s));
      *out_is_act = false;
      st.length = pos0 + T;
      return;
    }
    if (msgs) {
      pk::GateRows rows;
      for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
      gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3),
            xin_);
    }
    const int R = proj_rows(layer.full);
    const ActT* wp = bf16_rows(layer.wproj);
    if (!layer.full && !msgs) {
      gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
    } else {
      struct Part {
        int begin, end, input;
      };
      const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                            {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                            {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                            {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                            {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
      for (const Part& part : parts) {
        if (part.begin >= R) break;
        gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                   R, false, T, part.end - part.begin, H, s);
      }
    }
    pk::AttnPrepParams ap;
    ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
    ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
    ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
    ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
    ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
    ap.q = q_, ap.rel = rel16_;
    pk::AttnParams at;
    at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
    if (layer.full) {
      const int fi = st.full_slot(v);
      const std::size_t off = std::size_t(pos0) * st.full_row_bytes();
      void* kc = st.k_full_at(fi).data;
      void* vc = st.v_full_at(fi).data;
      ap.k_out = static_cast<char*>(kc) + off, ap.v_out = static_cast<char*>(vc) + off;
      ap.out_fp8 = st.kv_fp8, ap.out_q4 = st.kv_q4;
      pk::attn_prep(ap, s);
      at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.kv_q4 = st.kv_q4, at.kpos = nullptr, at.bounds = nullptr;
      at.L = pos0 + T, at.window = 0;
    } else {
      const int ri = plan.ring_index[std::size_t(v)];
      auto* k_ring = static_cast<ActT*>(st.k_ring_at(ri).data);
      auto* v_ring = static_cast<ActT*>(st.v_ring_at(ri).data);
      std::int64_t* ring_pos = st.ring_pos_at(ri);
      pk::ring_gather(k_ring, v_ring, ring_pos, pos0, T, krows_, vrows_, kpos_, kpos_rows_, s);
      ap.k_out = krows_ + std::size_t(WINDOW) * KV_DIM, ap.v_out = vrows_ + std::size_t(WINDOW) * KV_DIM;
      ap.out_fp8 = false, ap.out_q4 = false;
      pk::attn_prep(ap, s);
      pk::ring_update(krows_, vrows_, pos0, T, k_ring, v_ring, ring_pos, s);
      pk::key_tile_bounds(kpos_, WINDOW + T, bounds_, s);
      at.k = krows_, at.v = vrows_, at.kv_fp8 = false, at.kv_q4 = false, at.kpos = kpos_, at.bounds = bounds_;
      at.L = WINDOW + T, at.window = WINDOW;
    }
    pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
    pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);
    pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
    pk::attention(at, s);

    const ActT* wo = bf16_rows(layer.wo);
    gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_out_, H, false, T, H, HEADS * HEAD_DIM, s);
    pk::static_conv(wo_out_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
    pk::conv_state_update(st.conv_attn_at(v), wo_out_, H, T, H, s);

    std::vector<const float*> mlp_blocks = blocks;
    if (boundary) mlp_blocks.push_back(hid);
    const float* mbase = boundary ? nullptr : hid;
    const float* r_attn = abl_.active ? (attn_dirs + std::size_t(v) * H) : attn_dirs;
    const float* w_attn = abl_.active ? (attn_w + 4 * std::size_t(v)) : attn_w;
    if (msgs) {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
      pk::GateRows rows;
      rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
      gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1),
            xm_);
    } else {
      mix(mlp_blocks, mbase, y_, r_attn, w_attn, part_, f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
    }
    gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
    CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
    pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
    const Experts& ex = layer.experts;
    const bool exl13 = is_exl3(ex.w13_fmt), exl2 = is_exl3(ex.w2_fmt);
    const int moe_tr = (exl13 || exl2) ? 32 : pk::moe_tile_rows(T);
    const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
    pk::moe_route(idx_, T, work, s, moe_tr);
    pk::MoeExperts me{static_cast<int>(ex.w13_fmt),
                      ex.w13.data,
                      exl13 ? ex.w13_rin.data : ex.w13_scale.data,
                      exl13 ? nullptr : f32p(ex.w13_factor),
                      ex.w2.data,
                      exl2 ? ex.w2_rin.data : ex.w2_scale.data,
                      exl2 ? nullptr : f32p(ex.w2_factor),
                      static_cast<int>(ex.w2_fmt),
                      ex.w13_rout.data,
                      ex.w2_rout.data,
                      exl3_scratch_};
    me.tile_rows = moe_tr;
    if (w.expert_cache) {
      const auto eptrs = w.expert_cache->layer_ptrs(l);
      me.w13_ptrs = eptrs.w13;
      me.w13_scale_ptrs = eptrs.w13_scale;
      me.w13_rout_ptrs = eptrs.w13_rout;
      me.w2_ptrs = eptrs.w2;
      me.w2_scale_ptrs = eptrs.w2_scale;
      me.w2_rout_ptrs = eptrs.w2_rout;
      w.expert_cache->run_prefill_layer(l, counts_, s, [&](const std::uint8_t* mask) {
        me.active_mask = mask;
        pk::moe_gate_up(me, xm_, T, work, act_, s);
        pk::moe_down(me, act_, T, work, down_, s);
      });
    } else {
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      pk::moe_down(me, act_, T, work, down_, s);
    }
    pk::moe_combine(down_, wts_, T, moe_sum_, s);
    pk::static_conv(moe_sum_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
    pk::conv_state_update(st.conv_mlp_at(v), moe_sum_, H, T, H, s);
    prev_v = v;
  }

  const float* r_last = abl_.active ? (mlp_dirs + std::size_t(prev_v) * H) : mlp_dirs;
  const float* w_last = abl_.active ? (mlp_w + 4 * std::size_t(prev_v)) : mlp_w;
  mix({B0_, B1_, B2_[last_pass], E_[last_pass]}, part_, y2_, r_last, w_last, nullptr, f32p(w.final_query), 1,
      f32p(w.final_norm), xf_, nullptr);
  *out_is_act = true;
  st.length = pos0 + T;
}

EmbeddingResult Prefiller::embed(const EmbeddingRequest& req, int max_ctx, const PlatosKeyGpu* platos_key) {
  if (req.inputs.empty()) throw std::invalid_argument("embed: inputs must not be empty");
  const int n_visits = num_visits();
  if (req.tap_layer.has_value() && (*req.tap_layer < 0 || *req.tap_layer >= n_visits)) {
    throw std::invalid_argument("embed: tap_layer " + std::to_string(*req.tap_layer) + " out of range [0, " +
                                std::to_string(n_visits - 1) + "]");
  }
  if (req.use_platos_key && !platos_key) {
    throw std::invalid_argument("embed: platos_key requested but server was started without --platos-key");
  }
  const int total_seqs = static_cast<int>(req.inputs.size());
  int total_tokens = 0;
  for (int i = 0; i < total_seqs; ++i) {
    const auto& seq = req.inputs[std::size_t(i)];
    if (seq.empty()) throw std::invalid_argument("embed: input sequence must not be empty");
    if (static_cast<int>(seq.size()) > max_ctx) {
      throw std::invalid_argument("embed: input sequence length " + std::to_string(seq.size()) +
                                  " exceeds max_model_len " + std::to_string(max_ctx));
    }
    for (std::int32_t tid : seq) {
      if (tid < 0 || tid >= VOCAB) {
        throw std::out_of_range("embed: token id " + std::to_string(tid) + " out of vocab range [0, " +
                                std::to_string(VOCAB) + ")");
      }
    }
    total_tokens += static_cast<int>(seq.size());
  }

  const int out_dim = req.use_platos_key ? platos_key->out_dim() : H;
  const std::size_t need_toks = std::size_t(max_chunk_) * sizeof(std::int32_t);
  if (!embed_toks_dev_.p || embed_toks_dev_.bytes < need_toks) embed_toks_dev_ = DeviceBuffer(need_toks);
  const std::size_t need_off = std::size_t(total_seqs + 1) * sizeof(std::int32_t);
  if (!embed_offsets_dev_.p || embed_offsets_dev_.bytes < need_off) embed_offsets_dev_ = DeviceBuffer(need_off);
  const std::size_t need_pool = std::size_t(total_seqs) * std::size_t(H) * sizeof(float);
  if (!embed_pooled_dev_.p || embed_pooled_dev_.bytes < need_pool) embed_pooled_dev_ = DeviceBuffer(need_pool);
  if (req.use_platos_key) {
    const std::size_t need_proj = std::size_t(total_seqs) * std::size_t(out_dim) * sizeof(float);
    if (!embed_proj_dev_.p || embed_proj_dev_.bytes < need_proj) embed_proj_dev_ = DeviceBuffer(need_proj);
  }

  const auto t_prefill_start = std::chrono::steady_clock::now();
  const bool pool_norm_now = req.normalize && !req.use_platos_key;

  // Greedily pack short sequences (L <= max_chunk_) into compute-bound GEMM batches of total tokens <= max_chunk_.
  std::vector<std::int32_t> packed_toks;
  std::vector<int> packed_lens;
  std::vector<std::int32_t> packed_offsets;
  packed_toks.reserve(std::size_t(max_chunk_));

  int idx = 0;
  while (idx < total_seqs) {
    const int seq_len = static_cast<int>(req.inputs[std::size_t(idx)].size());
    if (seq_len <= max_chunk_) {
      packed_toks.clear();
      packed_lens.clear();
      packed_offsets.clear();
      packed_offsets.push_back(0);
      const int batch_start = idx;
      int packed_T = 0;
      while (idx < total_seqs) {
        const auto& seq = req.inputs[std::size_t(idx)];
        const int L = static_cast<int>(seq.size());
        if (L > max_chunk_ || (packed_T > 0 && packed_T + L > max_chunk_)) break;
        packed_toks.insert(packed_toks.end(), seq.begin(), seq.end());
        packed_lens.push_back(L);
        packed_T += L;
        packed_offsets.push_back(packed_T);
        ++idx;
      }
      CUDA_CHECK(cudaMemcpyAsync(embed_toks_dev_.p, packed_toks.data(),
                                 packed_toks.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(embed_offsets_dev_.p, packed_offsets.data(),
                                 packed_offsets.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_));
      float* dst = static_cast<float*>(embed_pooled_dev_.p) + std::size_t(batch_start) * std::size_t(H);
      run_embed_packed(static_cast<const std::int32_t*>(embed_toks_dev_.p), packed_lens,
                       static_cast<const std::int32_t*>(embed_offsets_dev_.p), req.pooling, req.tap_layer,
                       pool_norm_now, dst);
    } else {
      // Transient single-sequence chunked fallback for inputs exceeding max_chunk_ (never touches chat KV).
      const auto& seq = req.inputs[std::size_t(idx)];
      const int L = static_cast<int>(seq.size());
      State tmp_state = make_state(L, /*kv_fp8=*/false, /*kv_q4=*/false, {}, /*kv_last_full=*/false, loop_passes_);
      reset_state(tmp_state, stream_);
      DeviceBuffer chunk_f32(std::size_t(max_chunk_) * std::size_t(H) * sizeof(float));
      std::vector<double> mean_acc(std::size_t(H), 0.0);
      std::vector<float> seq_out(std::size_t(H), 0.0f);
      std::vector<float> host_f32;
      std::vector<ActT> host_act;
      for (int pos0 = 0; pos0 < L; pos0 += max_chunk_) {
        const int T = std::min(max_chunk_, L - pos0);
        CUDA_CHECK(cudaMemcpyAsync(embed_toks_dev_.p, seq.data() + pos0, std::size_t(T) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream_));
        bool out_is_act = false;
        run_embed_chunk(tmp_state, static_cast<const std::int32_t*>(embed_toks_dev_.p), T, pos0, req.tap_layer,
                        static_cast<float*>(chunk_f32.p), &out_is_act);
        const bool need_chunk =
            (req.pooling == EmbeddingPooling::Mean) ||
            (req.pooling == EmbeddingPooling::Cls && pos0 == 0) ||
            (req.pooling == EmbeddingPooling::LastToken && pos0 + T == L);
        if (!need_chunk) continue;
        if (out_is_act) {
          host_act.resize(std::size_t(T) * std::size_t(H));
          CUDA_CHECK(cudaMemcpyAsync(host_act.data(), xf_, host_act.size() * sizeof(ActT), cudaMemcpyDeviceToHost,
                                     stream_));
          CUDA_CHECK(cudaStreamSynchronize(stream_));
          if (req.pooling == EmbeddingPooling::Cls) {
            for (int d = 0; d < H; ++d) seq_out[std::size_t(d)] = act::host_to_float(host_act[std::size_t(d)]);
          } else if (req.pooling == EmbeddingPooling::LastToken) {
            const std::size_t base = std::size_t(T - 1) * std::size_t(H);
            for (int d = 0; d < H; ++d) seq_out[std::size_t(d)] = act::host_to_float(host_act[base + std::size_t(d)]);
          } else {
            for (int t = 0; t < T; ++t) {
              const std::size_t base = std::size_t(t) * std::size_t(H);
              for (int d = 0; d < H; ++d) mean_acc[std::size_t(d)] += act::host_to_float(host_act[base + std::size_t(d)]);
            }
          }
        } else {
          host_f32.resize(std::size_t(T) * std::size_t(H));
          CUDA_CHECK(cudaMemcpyAsync(host_f32.data(), chunk_f32.p, host_f32.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));
          CUDA_CHECK(cudaStreamSynchronize(stream_));
          if (req.pooling == EmbeddingPooling::Cls) {
            std::memcpy(seq_out.data(), host_f32.data(), std::size_t(H) * sizeof(float));
          } else if (req.pooling == EmbeddingPooling::LastToken) {
            std::memcpy(seq_out.data(), host_f32.data() + std::size_t(T - 1) * std::size_t(H),
                        std::size_t(H) * sizeof(float));
          } else {
            for (int t = 0; t < T; ++t) {
              const std::size_t base = std::size_t(t) * std::size_t(H);
              for (int d = 0; d < H; ++d) mean_acc[std::size_t(d)] += host_f32[base + std::size_t(d)];
            }
          }
        }
      }
      if (req.pooling == EmbeddingPooling::Mean) {
        const double inv = 1.0 / static_cast<double>(L);
        for (int d = 0; d < H; ++d) seq_out[std::size_t(d)] = static_cast<float>(mean_acc[std::size_t(d)] * inv);
      }
      if (pool_norm_now) l2_normalize_rows_cpu(seq_out.data(), 1, H);
      float* dst = static_cast<float*>(embed_pooled_dev_.p) + std::size_t(idx) * std::size_t(H);
      CUDA_CHECK(cudaMemcpyAsync(dst, seq_out.data(), std::size_t(H) * sizeof(float), cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      ++idx;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  const double prefill_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();

  double platos_sec = 0.0;
  const float* final_dev = static_cast<const float*>(embed_pooled_dev_.p);
  if (req.use_platos_key) {
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, stream_));
    platos_key->project(static_cast<const float*>(embed_pooled_dev_.p), total_seqs,
                        static_cast<float*>(embed_proj_dev_.p), req.normalize, stream_);
    CUDA_CHECK(cudaEventRecord(ev1, stream_));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float pk_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pk_ms, ev0, ev1));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    platos_sec = static_cast<double>(pk_ms) * 1e-3;
    final_dev = static_cast<const float*>(embed_proj_dev_.p);
  }

  std::vector<float> flat(std::size_t(total_seqs) * std::size_t(out_dim));
  CUDA_CHECK(cudaMemcpyAsync(flat.data(), final_dev, flat.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  EmbeddingResult res;
  res.embeddings.resize(std::size_t(total_seqs));
  for (int b = 0; b < total_seqs; ++b) {
    const float* row = flat.data() + std::size_t(b) * std::size_t(out_dim);
    res.embeddings[std::size_t(b)].assign(row, row + out_dim);
  }
  res.stats.total_prompt_tokens = total_tokens;
  res.stats.prefill_seconds = prefill_sec;
  res.stats.platos_key_seconds = platos_sec;
  res.stats.effective_tap_layer = req.tap_layer.value_or(-1);
  res.stats.output_dim = out_dim;
  res.stats.kv_slots_allocated = 0;
  res.stats.lm_head_skipped = true;
  return res;
}

}  // namespace ingine::model
