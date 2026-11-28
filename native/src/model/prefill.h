// Native prompt processing: model.Model.prefill for one chunk, without PyTorch.
#pragma once

#include "core/act.h"
#include "core/arena.h"
#include "ingine/generator.h"
#include "ingine/platos_key.h"
#include "kernels/common/persona_abi.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, sets state.length = pos0 + T, returns the last
// token's logits (device fp32 [UNPADDED_VOCAB], softcapped like the head) and, if taps != nullptr, every row's
// DSpark taps (bf16 [T, 4, H], model.TAP_VISITS + final normed hidden).
//
// The visit sequence is fast.FastDecoder.decode's (the same block-state wiring and mix / message-gate kernels)
// with T rows; projections are cuBLASLt BF16 GEMMs over the dequantized weight rows (weights.dequant), the
// attention a flash-attention prefill kernel, the MoE grouped GEMMs over the experts decoded on the fly.
// Stream-ordered on `stream`; run() only synchronises for its first use of a block-pointer table.
// Embedding rows computed elsewhere (image tokens from the vision tower) for rows [row, row + count) of a
// chunk: src bf16 [count, HIDDEN] on the device replaces the token embeddings of those rows (which the model
// feeds in as they are: no embedding norm).
struct EmbedRows {
  int row = 0, count = 0;
  const ActT* src = nullptr;
};

class Prefiller {
 public:
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream, int loop_passes = PASSES,
            LoopCdMode loopcd_mode = LoopCdMode::Off, float loopcd_omega = 0.0f);
  ~Prefiller();
  Prefiller(const Prefiller&) = delete;
  Prefiller& operator=(const Prefiller&) = delete;

  // Tokens with a negative id must be covered by `rows` (their embedding is the image placeholder's otherwise).
  void run(State& state, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {}, const kernels::persona::PersonaDeviceTable* hooks = nullptr);

  // Zero-KV, no-lm_head batched embedding prefill over one or more token sequences.
  // Packs multi-string inputs with len <= max_chunk_ into compute-bound GEMM batches using only
  // Prefiller's ephemeral scratch (zero persistent KV allocation, never touches chat State or paged KV).
  // Supports early-exit `tap_layer` (0..num_visits-1, matching TAP_VISITS hid tensors) and in-GPU
  // Plato's Key projection. Sequences longer than max_chunk_ fall back to a transient local state
  // that is freed before return so active chat KV remains untouched.
  EmbeddingResult embed(const EmbeddingRequest& req, int max_ctx, const PlatosKeyGpu* platos_key = nullptr);

  int max_chunk() const { return max_chunk_; }
  int num_visits() const { return make_visit_plan(loop_passes_).num_visits; }
  std::size_t device_bytes() const;  // scratch held by this object

 private:
  void run_embed_packed(const std::int32_t* d_tokens, std::span<const int> seq_lens,
                        const std::int32_t* d_seq_offsets, EmbeddingPooling pooling, std::optional<int> tap_visit,
                        bool l2_normalize, float* d_out_pooled);
  void run_embed_chunk(State& state, const std::int32_t* d_tokens, int T, int pos0, std::optional<int> tap_visit,
                       float* d_out_f32, bool* out_is_act);
  const float* const* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  int loop_passes_;
  LoopCdMode loopcd_mode_ = LoopCdMode::Off;
  float loopcd_omega_ = 0.0f;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  // activations (rows = max_chunk), see prefill.cpp
  float *B0_, *B1_, *B2_[MAX_PASSES]{}, *E_[MAX_PASSES]{}, *hid_, *part_, *y_, *y2_;
  ActT *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  ActT *proj_, *q_, *rel16_, *rel_, *krows_, *vrows_, *attn_out_, *wo_out_;
  int* kpos_;
  int4* bounds_;
  ActT *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  ActT *act_, *down_, *moe_sum_, *exl3_scratch_;
  ActT* wscratch_;  // dequantized FP8 rows of wproj / wo (dense_fp8)
  float* logits_weak_ = nullptr;    // [UNPADDED_VOCAB] scratch when loopcd_mode_ == LogitsAdaptive
  float* loopcd_reduce_ = nullptr;  // [kLoopCdReduceFloatsPerRow] scratch when loopcd_mode_ == LogitsAdaptive
  float* zero_conv_ = nullptr;      // [3 * H] read-only zero buffer for stateless zero-KV embedding convs
  std::int64_t* ids_;
  int kpos_rows_;
  // Lazily allocated tiny device staging for embed() calls (token ids, seq offsets, pooled/projected rows).
  DeviceBuffer embed_toks_dev_;
  DeviceBuffer embed_offsets_dev_;
  DeviceBuffer embed_pooled_dev_;
  DeviceBuffer embed_proj_dev_;
};

}  // namespace ingine::model
