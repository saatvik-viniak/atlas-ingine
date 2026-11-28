#include "backends/qwen38/qwen38_dense_gpu.h"

#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_elastic_ops.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace ingine::backends::qwen38 {
namespace {

template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  std::size_t count = 0;
  DevBuf() = default;
  explicit DevBuf(std::size_t n) { allocate(n); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr(std::exchange(o.ptr, nullptr)), count(std::exchange(o.count, 0)) {}
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      free();
      ptr = std::exchange(o.ptr, nullptr);
      count = std::exchange(o.count, 0);
    }
    return *this;
  }
  ~DevBuf() { free(); }
  void allocate(std::size_t n) {
    free();
    count = n;
    if (n > 0) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  }
  void free() noexcept {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
  }
};

struct DevProj {
  int out_features = 0;
  int in_features = 0;
  Qwen38LinearQuantKind quant = Qwen38LinearQuantKind::Bf16;
  const __nv_bfloat16* w_bf16 = nullptr;
  const std::uint8_t* w_u8 = nullptr;
  const std::uint8_t* w_scales = nullptr;
  float weight_scale = 1.0f;
  float weight_scale_2 = 1.0f;
  float input_scale = 0.0f;
};

struct DevLayer {
  Qwen38LayerKind kind = Qwen38LayerKind::LinearAttention;
  const __nv_bfloat16* input_layernorm = nullptr;
  const __nv_bfloat16* post_attention_layernorm = nullptr;

  DevProj in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
  const float* a_log = nullptr;
  const float* dt_bias = nullptr;
  const float* conv1d_weight = nullptr;
  const float* linear_norm_weight = nullptr;

  DevProj q_proj, k_proj, v_proj, o_proj;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;

  DevProj gate_proj, up_proj, down_proj;
};

std::uint64_t align_up(std::uint64_t x, std::uint64_t a = 256) noexcept { return (x + (a - 1)) & ~(a - 1); }

std::uint64_t parse_header_len_le(std::span<const std::uint8_t> eight_bytes) {
  std::uint64_t h = 0;
  for (int i = 0; i < 8; ++i) h |= (static_cast<std::uint64_t>(eight_bytes[i]) << (8 * i));
  if (h == 0 || h > (64ull * 1024ull * 1024ull)) {
    throw std::invalid_argument("invalid safetensors header length during GPU load");
  }
  return h;
}

float fp16_to_f32(std::uint16_t bits) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16;
  const std::uint32_t exp = (bits >> 10) & 0x1fu;
  const std::uint32_t frac = bits & 0x3ffu;
  if (exp == 0) {
    if (frac == 0) {
      float out = 0.0f;
      std::memcpy(&out, &sign, 4);
      return out;
    }
    return std::ldexp(static_cast<float>(frac) / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
  }
  if (exp == 0x1fu) {
    const std::uint32_t inf_or_nan = sign | 0x7f800000u | (frac << 13);
    float out = 0.0f;
    std::memcpy(&out, &inf_or_nan, 4);
    return out;
  }
  const std::uint32_t f_bits = sign | ((exp + 112u) << 23) | (frac << 13);
  float out = 0.0f;
  std::memcpy(&out, &f_bits, 4);
  return out;
}

float decode_scalar_bytes(std::span<const std::uint8_t> raw, std::string_view dtype, std::string_view name) {
  if (dtype == "F32") {
    if (raw.size() != 4) throw std::invalid_argument("scalar F32 byte size != 4 for " + std::string(name));
    float v = 0.0f;
    std::memcpy(&v, raw.data(), 4);
    return v;
  }
  if (dtype == "BF16") {
    if (raw.size() != 2) throw std::invalid_argument("scalar BF16 byte size != 2 for " + std::string(name));
    std::uint16_t b = 0;
    std::memcpy(&b, raw.data(), 2);
    return bf16_to_f32(b);
  }
  if (dtype == "F16") {
    if (raw.size() != 2) throw std::invalid_argument("scalar F16 byte size != 2 for " + std::string(name));
    std::uint16_t b = 0;
    std::memcpy(&b, raw.data(), 2);
    return fp16_to_f32(b);
  }
  throw std::invalid_argument("unsupported scalar dtype '" + std::string(dtype) + "' for " + std::string(name));
}

void decode_f32_vec_bytes(std::span<const std::uint8_t> raw, std::string_view dtype, std::size_t expected_elems,
                          std::string_view name, std::vector<float>& dst, std::size_t dst_off) {
  if (dtype == "F32") {
    if (raw.size() != expected_elems * 4) throw std::invalid_argument("F32 byte mismatch for " + std::string(name));
    std::memcpy(dst.data() + dst_off, raw.data(), raw.size());
    return;
  }
  if (dtype == "BF16") {
    if (raw.size() != expected_elems * 2) throw std::invalid_argument("BF16 byte mismatch for " + std::string(name));
    for (std::size_t i = 0; i < expected_elems; ++i) {
      std::uint16_t b = 0;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      dst[dst_off + i] = bf16_to_f32(b);
    }
    return;
  }
  throw std::invalid_argument("unsupported 1D GDN dtype '" + std::string(dtype) + "' for " + std::string(name));
}

void stream_unpinned_shard(const Qwen38Manifest& manifest, const std::filesystem::path& model_dir,
                           const std::string& shard,
                           const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>& on_chunk,
                           const std::function<void()>& on_abort) {
  const std::filesystem::path sp = model_dir / shard;
  int fd = ::open(sp.c_str(), O_RDONLY);
  if (fd < 0) {
    if (on_abort) on_abort();
    throw std::invalid_argument("cannot open shard " + sp.string() + ": " + std::strerror(errno));
  }
  try {
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
      throw std::invalid_argument("not a regular file: " + sp.string());
    }
    const auto vf_it = manifest.verified_files.find(shard);
    if (vf_it != manifest.verified_files.end()) {
      if (static_cast<std::uint64_t>(st.st_dev) != vf_it->second.dev ||
          static_cast<std::uint64_t>(st.st_ino) != vf_it->second.ino ||
          static_cast<std::uint64_t>(st.st_size) != vf_it->second.byte_size) {
        throw std::invalid_argument("open FD mismatch for unpinned shard " + shard);
      }
    }
    std::vector<std::uint8_t> buf(4ull * 1024ull * 1024ull);
    std::uint64_t offset = 0;
    while (true) {
      const ssize_t got = ::pread(fd, buf.data(), buf.size(), static_cast<off_t>(offset));
      if (got < 0) {
        if (errno == EINTR) continue;
        throw std::invalid_argument("pread failed on " + sp.string() + ": " + std::strerror(errno));
      }
      if (got == 0) break;
      on_chunk(offset, std::span<const std::uint8_t>(buf.data(), static_cast<std::size_t>(got)));
      offset += static_cast<std::uint64_t>(got);
    }
    ::close(fd);
  } catch (...) {
    ::close(fd);
    if (on_abort) on_abort();
    throw;
  }
}

struct CopySpec {
  const Qwen38TensorInfo* info = nullptr;
  bool to_device = true;
  std::uint64_t dev_offset = 0;
  std::vector<std::uint8_t>* host_dst = nullptr;
};

struct ActiveTarget {
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
  bool to_device = true;
  std::uint64_t dev_offset = 0;
  std::vector<std::uint8_t>* host_dst = nullptr;
};

struct HostScalarEntry {
  float* target_field = nullptr;
  std::string dtype;
  std::vector<std::uint8_t> raw;
};

struct HostF32VecEntry {
  std::size_t f32_offset = 0;
  std::size_t elems = 0;
  std::string dtype;
  std::string name;
  std::vector<std::uint8_t> raw;
};

struct SavedCheckpoint {
  std::size_t token_count = 0;
  std::vector<float> conv_host;
  std::vector<float> rec_host;
  std::vector<float> last_logits;
  bool has_last_logits = false;
  int last_logits_pos = -1;
};

}  // namespace

struct Qwen38DenseSession::StateImpl {
  std::vector<DevBuf<float>> conv_states;
  std::vector<DevBuf<float>> rec_states;
  std::vector<DevBuf<__nv_bfloat16>> k_caches;
  std::vector<DevBuf<__nv_bfloat16>> v_caches;
  std::size_t total_state_bytes = 0;
  std::size_t gdn_conv_total_floats = 0;
  std::size_t gdn_rec_total_floats = 0;
  std::size_t kv_stride_elems = 0;

  Qwen38CheckpointIndex index{4};
  std::vector<SavedCheckpoint> checkpoints;
  std::vector<float> tip_logits;
  std::uint64_t last_restore_bytes = 0;
  double last_restore_seconds = 0.0;
  double last_reset_ms = 0.0;
};

struct Qwen38DenseBackend::Impl {
  DeviceContext ctx;
  Qwen38Manifest manifest;
  Qwen38DenseLoadOptions opts;

  DevBuf<std::uint8_t> weights_arena;
  DevBuf<float> f32_arena;

  Qwen38LinearQuantKind embed_quant = Qwen38LinearQuantKind::Bf16;
  const __nv_bfloat16* embed_tokens = nullptr;
  const std::uint8_t* embed_u8 = nullptr;
  const std::uint8_t* embed_scales = nullptr;
  float embed_scale = 1.0f;
  const __nv_bfloat16* final_norm = nullptr;
  DevProj lm_head;
  std::vector<DevLayer> layers;

  // Shared execution scratch buffers (FP32 activations for bit-exact batch invariance across M=1..256).
  DevBuf<float> act_quant_scratch;
  DevBuf<std::int32_t> token_ids_dev;
  DevBuf<int> gather_rows_dev;
  DevBuf<float> hidden;
  DevBuf<float> normed;
  DevBuf<float> residual;
  DevBuf<float> qkv;
  DevBuf<float> conv_out;
  DevBuf<float> z_buf;
  DevBuf<float> a_buf;
  DevBuf<float> b_buf;
  DevBuf<float> core_out;
  DevBuf<float> gdn_scratch;

  DevBuf<float> q_gate;
  DevBuf<float> k_raw;
  DevBuf<float> v_raw;
  DevBuf<float> q_tmp;
  DevBuf<float> gate_tmp;
  DevBuf<float> attn_out;
  DevBuf<float> attn_split_scratch;

  DevBuf<float> mlp_gate;
  DevBuf<float> mlp_up;

  static constexpr int kLogitBatchMax = 64;
  DevBuf<float> cap_hidden;
  DevBuf<float> cap_normed;
  DevBuf<float> logits_f32;
  std::vector<float> host_logits_stage;
  std::vector<std::int32_t> decode_single_tok;

  std::unique_ptr<Qwen38DenseSession> default_session;
  std::shared_ptr<const ingine::PlatosKeyGpu> platos_key_gpu;
  // Lazily allocated single-layer ephemeral scratch for zero-KV batched /v1/embeddings (never touches default_session).
  DevBuf<float> embed_conv_tmp;
  DevBuf<float> embed_rec_tmp;
  DevBuf<__nv_bfloat16> embed_k_tmp;
  DevBuf<__nv_bfloat16> embed_v_tmp;
  DevBuf<std::int32_t> embed_offsets_dev;
  DevBuf<float> embed_pooled_dev;
  DevBuf<float> embed_proj_dev;
  std::uint64_t scratch_bytes = 0;
  std::uint64_t max_state_bytes = 0;
  ingine::model::SessionLookupResult last_lookup;
  double last_lookup_ms = 0.0;
  double last_reset_ms = 0.0;

  explicit Impl(int dev_id) : ctx(dev_id) { ctx.bind_to_current_thread(); }

  void proj_forward_f32(const DevProj& p, const float* x, int m, float* out) {
    if (m <= 0) throw std::invalid_argument("proj_forward_f32 m <= 0");
    if (p.quant == Qwen38LinearQuantKind::Bf16) {
      qwen38_cuda_bf16_linear_f32_batched(x, p.w_bf16, m, p.out_features, p.in_features, out, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Fp8E4M3) {
      const float* x_use = x;
      if (opts.use_calibrated_fp8 && p.input_scale > 0.0f) {
        const std::size_t x_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(p.in_features);
        if (x_elems > act_quant_scratch.count) throw std::runtime_error("act_quant_scratch overflow");
        qwen38_cuda_quant_fp8_act_f32(x, p.input_scale, x_elems, act_quant_scratch.ptr, ctx.stream);
        x_use = act_quant_scratch.ptr;
      }
      qwen38_cuda_fp8_linear_f32_batched(x_use, p.w_u8, p.weight_scale, 0.0f, m, p.out_features, p.in_features, out,
                                         ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Atlas3) {
      qwen38_cuda_atlas3_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                            out, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Atlas2) {
      qwen38_cuda_atlas2_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                            out, ctx.stream);
    } else {
      qwen38_cuda_nvfp4_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                           out, ctx.stream);
    }
  }

  void lm_head_forward_f32(const float* x, int m, float* out_f32) {
    const DevProj& p = lm_head;
    if (m <= 0) throw std::invalid_argument("lm_head_forward_f32 m <= 0");
    if (p.quant == Qwen38LinearQuantKind::Bf16) {
      qwen38_cuda_bf16_linear_f32_batched(x, p.w_bf16, m, p.out_features, p.in_features, out_f32, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Fp8E4M3) {
      qwen38_cuda_fp8_linear_f32_batched(x, p.w_u8, p.weight_scale, 0.0f, m, p.out_features, p.in_features, out_f32,
                                         ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Nvfp4) {
      qwen38_cuda_nvfp4_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                           out_f32, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Atlas3) {
      qwen38_cuda_atlas3_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                            out_f32, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Atlas2) {
      qwen38_cuda_atlas2_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                            out_f32, ctx.stream);
    } else {
      throw std::invalid_argument("unsupported lm_head quant");
    }
  }

  void forward_range(Qwen38DenseSession& sess, std::span<const std::int32_t> token_ids,
                     std::span<const int> sorted_capture_positions, bool use_chunk_prefill,
                     const ingine::model::LogitCallback& on_logits, double* prefill_ms_out,
                     std::span<const std::uint8_t> token_mask = {}, bool capture_tip_logits = true) {
    ctx.bind_to_current_thread();
    const auto& c = manifest.config;
    const int S = static_cast<int>(token_ids.size());
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int pos_start = static_cast<int>(sess.committed_tokens().size());
    if (!token_mask.empty() && token_mask.size() != token_ids.size()) {
      throw std::invalid_argument("forward_range token_mask length mismatch");
    }
    if (S == 0) return;
    if (pos_start + S > opts.max_ctx) {
      throw std::invalid_argument("Qwen38DenseBackend sequence exceeds max_ctx");
    }
    sess.state_->index.clear_tip_logits();
    sess.state_->tip_logits.clear();

    const int step_cap = use_chunk_prefill ? opts.prefill_chunk : 1;
    std::size_t cap_idx = 0;
    while (cap_idx < sorted_capture_positions.size() && sorted_capture_positions[cap_idx] < pos_start) {
      ++cap_idx;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int cd = c.linear_conv_dim();
    const int nhv = c.linear_num_value_heads;
    const int nkh = c.linear_num_key_heads;
    const int k_dim = c.linear_key_head_dim;
    const int v_dim = c.linear_value_head_dim;

    for (int off = 0; off < S; off += step_cap) {
      const int M = std::min(step_cap, S - off);
      const int global_pos = pos_start + off;
      CUDA_CHECK(cudaMemcpyAsync(token_ids_dev.ptr, token_ids.data() + off,
                                 static_cast<std::size_t>(M) * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                 ctx.stream));
      if (embed_quant == Qwen38LinearQuantKind::Fp8E4M3) {
        qwen38_cuda_embed_lookup_fp8_f32(embed_u8, embed_scale, token_ids_dev.ptr, M, H, V, hidden.ptr, ctx.stream);
      } else if (embed_quant == Qwen38LinearQuantKind::Nvfp4) {
        qwen38_cuda_embed_lookup_nvfp4_f32(embed_u8, embed_scales, embed_scale, token_ids_dev.ptr, M, H, V, hidden.ptr,
                                           ctx.stream);
      } else {
        qwen38_cuda_embed_lookup_f32(embed_tokens, token_ids_dev.ptr, M, H, V, hidden.ptr, ctx.stream);
      }

      std::size_t lin_idx = 0;
      std::size_t full_idx = 0;
      for (std::size_t li = 0; li < layers.size(); ++li) {
        const auto& l = layers[li];
        qwen38_cuda_rmsnorm_zero_centered_f32(hidden.ptr, l.input_layernorm, c.rms_norm_eps, M, H, normed.ptr,
                                              ctx.stream);
        if (l.kind == Qwen38LayerKind::LinearAttention) {
          proj_forward_f32(l.in_proj_qkv, normed.ptr, M, qkv.ptr);
          proj_forward_f32(l.in_proj_z, normed.ptr, M, z_buf.ptr);
          proj_forward_f32(l.in_proj_a, normed.ptr, M, a_buf.ptr);
          proj_forward_f32(l.in_proj_b, normed.ptr, M, b_buf.ptr);

          float* conv_st = sess.state_->conv_states[lin_idx].ptr;
          float* rec_st = sess.state_->rec_states[lin_idx].ptr;
          qwen38_cuda_causal_conv1d_silu_prefill_f32(qkv.ptr, l.conv1d_weight, M, cd, c.linear_conv_kernel_dim,
                                                     conv_st, conv_out.ptr, ctx.stream);
          qwen38_cuda_gdn_recurrent_prefill_f32(conv_out.ptr, z_buf.ptr, a_buf.ptr, b_buf.ptr, l.a_log, l.dt_bias,
                                                l.linear_norm_weight, c.rms_norm_eps, M, nkh, nhv, k_dim, v_dim, rec_st,
                                                gdn_scratch.ptr, core_out.ptr, ctx.stream);
          proj_forward_f32(l.out_proj, core_out.ptr, M, residual.ptr);
          ++lin_idx;
        } else {
          proj_forward_f32(l.q_proj, normed.ptr, M, q_gate.ptr);
          proj_forward_f32(l.k_proj, normed.ptr, M, k_raw.ptr);
          proj_forward_f32(l.v_proj, normed.ptr, M, v_raw.ptr);
          qwen38_cuda_full_attn_gated_f32_kvbf16(
              q_gate.ptr, k_raw.ptr, v_raw.ptr, l.q_norm, l.k_norm, c.rms_norm_eps, c.rope_theta, M, global_pos,
              c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim,
              sess.state_->k_caches[full_idx].ptr, sess.state_->v_caches[full_idx].ptr, opts.max_ctx, q_tmp.ptr,
              gate_tmp.ptr, attn_out.ptr, ctx.stream, attn_split_scratch.ptr, attn_split_scratch.count,
              opts.use_fast_attn);
          proj_forward_f32(l.o_proj, attn_out.ptr, M, residual.ptr);
          ++full_idx;
        }

        qwen38_cuda_add_residual_f32(hidden.ptr, residual.ptr, static_cast<std::size_t>(M * H), ctx.stream);
        qwen38_cuda_rmsnorm_zero_centered_f32(hidden.ptr, l.post_attention_layernorm, c.rms_norm_eps, M, H,
                                              normed.ptr, ctx.stream);
        proj_forward_f32(l.gate_proj, normed.ptr, M, mlp_gate.ptr);
        proj_forward_f32(l.up_proj, normed.ptr, M, mlp_up.ptr);
        qwen38_cuda_swiglu_mul_f32(mlp_gate.ptr, mlp_up.ptr, mlp_gate.ptr, static_cast<std::size_t>(M * I),
                                   ctx.stream);
        proj_forward_f32(l.down_proj, mlp_gate.ptr, M, residual.ptr);
        qwen38_cuda_add_residual_f32(hidden.ptr, residual.ptr, static_cast<std::size_t>(M * H), ctx.stream);
      }

      std::vector<int> local_rows;
      std::vector<int> global_rows;
      std::vector<bool> emit_cb;
      const int tip_gp = pos_start + S - 1;
      while (cap_idx < sorted_capture_positions.size() && sorted_capture_positions[cap_idx] < global_pos + M) {
        const int gp = sorted_capture_positions[cap_idx];
        if (on_logits || gp == tip_gp) {
          local_rows.push_back(gp - global_pos);
          global_rows.push_back(gp);
          emit_cb.push_back(static_cast<bool>(on_logits));
        }
        ++cap_idx;
      }
      if (capture_tip_logits && (off + M == S) && (global_rows.empty() || global_rows.back() != tip_gp)) {
        local_rows.push_back(M - 1);
        global_rows.push_back(tip_gp);
        emit_cb.push_back(false);
      }

      if (!local_rows.empty()) {
        for (std::size_t b0 = 0; b0 < local_rows.size(); b0 += kLogitBatchMax) {
          const int bc = static_cast<int>(std::min<std::size_t>(kLogitBatchMax, local_rows.size() - b0));
          const float* x_cap = nullptr;
          if (bc == 1 && M == 1) {
            x_cap = hidden.ptr;
          } else {
            CUDA_CHECK(cudaMemcpyAsync(gather_rows_dev.ptr, local_rows.data() + b0,
                                       static_cast<std::size_t>(bc) * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));
            qwen38_cuda_gather_rows_f32(hidden.ptr, gather_rows_dev.ptr, bc, H, cap_hidden.ptr, ctx.stream);
            x_cap = cap_hidden.ptr;
          }
          qwen38_cuda_rmsnorm_zero_centered_f32(x_cap, final_norm, c.rms_norm_eps, bc, H, cap_normed.ptr, ctx.stream);
          lm_head_forward_f32(cap_normed.ptr, bc, logits_f32.ptr);
          const std::size_t copy_elems = static_cast<std::size_t>(bc) * static_cast<std::size_t>(V);
          CUDA_CHECK(cudaMemcpyAsync(host_logits_stage.data(), logits_f32.ptr, copy_elems * sizeof(float),
                                     cudaMemcpyDeviceToHost, ctx.stream));
          ctx.synchronize();
          for (int r = 0; r < bc; ++r) {
            const std::size_t row_idx = b0 + static_cast<std::size_t>(r);
            const int gp = global_rows[row_idx];
            const std::span<const float> row_logits(
                host_logits_stage.data() + static_cast<std::size_t>(r) * V, static_cast<std::size_t>(V));
            if (gp == tip_gp) {
              sess.state_->tip_logits.assign(row_logits.begin(), row_logits.end());
              sess.state_->index.set_tip_logits(gp);
            }
            if (on_logits && emit_cb[row_idx]) {
              on_logits(ingine::model::LogitStep{gp, row_logits});
            }
          }
        }
      }

      const std::span<const std::uint8_t> mask_slice =
          token_mask.empty() ? std::span<const std::uint8_t>{} : token_mask.subspan(off, static_cast<std::size_t>(M));
      sess.state_->index.append_committed(token_ids.subspan(off, static_cast<std::size_t>(M)), mask_slice);
    }

    ctx.synchronize();
    if (prefill_ms_out) {
      *prefill_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
  }
};

Qwen38DenseSession::Qwen38DenseSession(Qwen38DenseBackend& owner, std::size_t max_checkpoints)
    : owner_(owner), state_(std::make_unique<StateImpl>()) {
  state_->index = Qwen38CheckpointIndex(max_checkpoints);
  const auto& c = owner_.config();
  const int max_ctx = owner_.impl_->opts.max_ctx;
  const int cd = c.linear_conv_dim();
  const std::size_t conv_elems = static_cast<std::size_t>(cd) * static_cast<std::size_t>(c.linear_conv_kernel_dim);
  const std::size_t rec_elems = static_cast<std::size_t>(c.linear_num_value_heads) *
                                static_cast<std::size_t>(c.linear_key_head_dim) *
                                static_cast<std::size_t>(c.linear_value_head_dim);
  const std::size_t kv_stride = static_cast<std::size_t>(c.full_kv_proj_out_dim());
  const std::size_t kv_elems = static_cast<std::size_t>(max_ctx) * kv_stride;
  state_->kv_stride_elems = kv_stride;

  owner_.impl_->ctx.bind_to_current_thread();
  state_->conv_states.resize(static_cast<std::size_t>(c.num_linear_layers));
  state_->rec_states.resize(static_cast<std::size_t>(c.num_linear_layers));
  for (int i = 0; i < c.num_linear_layers; ++i) {
    state_->conv_states[static_cast<std::size_t>(i)].allocate(conv_elems);
    state_->rec_states[static_cast<std::size_t>(i)].allocate(rec_elems);
    state_->gdn_conv_total_floats += conv_elems;
    state_->gdn_rec_total_floats += rec_elems;
    state_->total_state_bytes += (conv_elems + rec_elems) * sizeof(float);
  }
  state_->k_caches.resize(static_cast<std::size_t>(c.num_full_layers));
  state_->v_caches.resize(static_cast<std::size_t>(c.num_full_layers));
  for (int i = 0; i < c.num_full_layers; ++i) {
    state_->k_caches[static_cast<std::size_t>(i)].allocate(kv_elems);
    state_->v_caches[static_cast<std::size_t>(i)].allocate(kv_elems);
    state_->total_state_bytes += kv_elems * sizeof(__nv_bfloat16) * 2ull;
  }
  reset();
}

Qwen38DenseSession::~Qwen38DenseSession() = default;

void Qwen38DenseSession::reset() {
  const auto t0 = std::chrono::steady_clock::now();
  owner_.impl_->ctx.bind_to_current_thread();
  for (auto& b : state_->conv_states) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(float), owner_.impl_->ctx.stream));
  }
  for (auto& b : state_->rec_states) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(float), owner_.impl_->ctx.stream));
  }
  for (auto& b : state_->k_caches) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(__nv_bfloat16), owner_.impl_->ctx.stream));
  }
  for (auto& b : state_->v_caches) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(__nv_bfloat16), owner_.impl_->ctx.stream));
  }
  owner_.impl_->ctx.synchronize();
  state_->last_reset_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  owner_.impl_->last_reset_ms = state_->last_reset_ms;
  std::vector<SavedCheckpoint>().swap(state_->checkpoints);
  state_->tip_logits.clear();
  state_->last_restore_bytes = 0;
  state_->last_restore_seconds = 0.0;
  state_->index.reset(owner_.state_identity());
}

std::size_t Qwen38DenseSession::state_bytes() const noexcept { return state_->total_state_bytes; }

std::span<const std::int32_t> Qwen38DenseSession::committed_tokens() const noexcept {
  return state_->index.committed_tokens();
}

std::span<const std::uint8_t> Qwen38DenseSession::committed_token_mask() const noexcept {
  return state_->index.committed_token_mask();
}

const ingine::model::BackendStateIdentity& Qwen38DenseSession::identity() const noexcept {
  return state_->index.identity();
}

std::size_t Qwen38DenseSession::restorable_prefix_len(std::size_t max_prefix_len) const noexcept {
  return state_->index.restorable_prefix_len(max_prefix_len);
}

void Qwen38DenseSession::truncate(std::size_t new_token_count) {
  const std::size_t old_tc = state_->index.committed_tokens().size();
  if (new_token_count > old_tc) {
    throw std::out_of_range("Qwen38DenseSession::truncate exceeds committed token count");
  }
  if (new_token_count == 0) {
    reset();
    return;
  }
  if (new_token_count == old_tc) {
    state_->last_restore_bytes = 0;
    state_->last_restore_seconds = 0.0;
    return;
  }

  auto it = std::find_if(state_->checkpoints.begin(), state_->checkpoints.end(),
                         [&](const SavedCheckpoint& ck) { return ck.token_count == new_token_count; });
  if (it == state_->checkpoints.end()) {
    throw std::logic_error("no saved GDN checkpoint at requested token_count");
  }

  const auto t0 = std::chrono::steady_clock::now();
  owner_.impl_->ctx.bind_to_current_thread();

  std::size_t conv_off = 0;
  for (auto& b : state_->conv_states) {
    if (b.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(b.ptr, it->conv_host.data() + conv_off, b.count * sizeof(float),
                                 cudaMemcpyHostToDevice, owner_.impl_->ctx.stream));
      conv_off += b.count;
    }
  }
  std::size_t rec_off = 0;
  for (auto& b : state_->rec_states) {
    if (b.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(b.ptr, it->rec_host.data() + rec_off, b.count * sizeof(float),
                                 cudaMemcpyHostToDevice, owner_.impl_->ctx.stream));
      rec_off += b.count;
    }
  }

  const std::size_t tail_tokens = old_tc - new_token_count;
  const std::size_t kv_off_elems = new_token_count * state_->kv_stride_elems;
  const std::size_t kv_clr_bytes = tail_tokens * state_->kv_stride_elems * sizeof(__nv_bfloat16);
  if (kv_clr_bytes > 0) {
    for (auto& b : state_->k_caches) {
      CUDA_CHECK(cudaMemsetAsync(b.ptr + kv_off_elems, 0, kv_clr_bytes, owner_.impl_->ctx.stream));
    }
    for (auto& b : state_->v_caches) {
      CUDA_CHECK(cudaMemsetAsync(b.ptr + kv_off_elems, 0, kv_clr_bytes, owner_.impl_->ctx.stream));
    }
  }
  owner_.impl_->ctx.synchronize();

  state_->last_restore_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  state_->last_restore_bytes =
      static_cast<std::uint64_t>(state_->gdn_conv_total_floats + state_->gdn_rec_total_floats) * sizeof(float);

  SavedCheckpoint kept = *it;
  state_->index.truncate_committed(new_token_count);
  (void)state_->index.prune_above(new_token_count);
  state_->checkpoints.erase(
      std::remove_if(state_->checkpoints.begin(), state_->checkpoints.end(),
                     [&](const SavedCheckpoint& ck) { return ck.token_count > new_token_count; }),
      state_->checkpoints.end());

  if (kept.has_last_logits && kept.last_logits_pos == static_cast<int>(new_token_count) - 1) {
    state_->tip_logits = std::move(kept.last_logits);
    state_->index.set_tip_logits(kept.last_logits_pos);
  } else {
    state_->tip_logits.clear();
  }
}

const Qwen38CheckpointIndex& Qwen38DenseSession::checkpoint_index() const noexcept { return state_->index; }

std::uint64_t Qwen38DenseSession::saved_checkpoint_bytes() const noexcept {
  std::uint64_t total = 0;
  for (const auto& ck : state_->checkpoints) {
    total += static_cast<std::uint64_t>(ck.conv_host.size() + ck.rec_host.size() + ck.last_logits.size()) *
             sizeof(float);
  }
  return total;
}

std::uint64_t Qwen38DenseSession::last_restore_bytes() const noexcept { return state_->last_restore_bytes; }

double Qwen38DenseSession::last_restore_seconds() const noexcept { return state_->last_restore_seconds; }

double Qwen38DenseSession::last_reset_ms() const noexcept { return state_->last_reset_ms; }

int Qwen38DenseSession::save_checkpoint() {
  const std::size_t tc = state_->index.committed_tokens().size();
  if (tc == 0) return 0;

  auto existing = std::find_if(state_->checkpoints.begin(), state_->checkpoints.end(),
                               [&](const SavedCheckpoint& ck) { return ck.token_count == tc; });
  if (existing != state_->checkpoints.end()) {
    if (!existing->has_last_logits && state_->index.has_tip_logits() &&
        state_->index.tip_logits_pos() == static_cast<int>(tc) - 1) {
      existing->last_logits = state_->tip_logits;
      existing->has_last_logits = true;
      existing->last_logits_pos = state_->index.tip_logits_pos();
    }
    return 0;
  }

  int evicted_cnt = 0;
  if (const auto evicted = state_->index.record_checkpoint(tc); evicted.has_value()) {
    state_->checkpoints.erase(
        std::remove_if(state_->checkpoints.begin(), state_->checkpoints.end(),
                       [&](const SavedCheckpoint& ck) { return ck.token_count == *evicted; }),
        state_->checkpoints.end());
    evicted_cnt = 1;
  }

  SavedCheckpoint ck;
  ck.token_count = tc;
  ck.conv_host.resize(state_->gdn_conv_total_floats);
  ck.rec_host.resize(state_->gdn_rec_total_floats);

  owner_.impl_->ctx.bind_to_current_thread();
  std::size_t conv_off = 0;
  for (const auto& b : state_->conv_states) {
    if (b.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(ck.conv_host.data() + conv_off, b.ptr, b.count * sizeof(float),
                                 cudaMemcpyDeviceToHost, owner_.impl_->ctx.stream));
      conv_off += b.count;
    }
  }
  std::size_t rec_off = 0;
  for (const auto& b : state_->rec_states) {
    if (b.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(ck.rec_host.data() + rec_off, b.ptr, b.count * sizeof(float),
                                 cudaMemcpyDeviceToHost, owner_.impl_->ctx.stream));
      rec_off += b.count;
    }
  }
  owner_.impl_->ctx.synchronize();

  if (state_->index.has_tip_logits() && state_->index.tip_logits_pos() == static_cast<int>(tc) - 1 &&
      !state_->tip_logits.empty()) {
    ck.last_logits = state_->tip_logits;
    ck.has_last_logits = true;
    ck.last_logits_pos = state_->index.tip_logits_pos();
  }
  state_->checkpoints.push_back(std::move(ck));
  std::sort(state_->checkpoints.begin(), state_->checkpoints.end(),
            [](const SavedCheckpoint& a, const SavedCheckpoint& b) { return a.token_count < b.token_count; });
  return evicted_cnt;
}

std::uint64_t Qwen38DenseSession::evict_checkpoints() {
  const std::uint64_t freed = saved_checkpoint_bytes();
  (void)state_->index.evict_all_checkpoints();
  std::vector<SavedCheckpoint>().swap(state_->checkpoints);
  return freed;
}

Qwen38GemmParityReport Qwen38DenseBackend::self_check_gemm_parity(int device_id) {
  DeviceContext ctx(device_id);
  ctx.bind_to_current_thread();

  constexpr int M = 17, N = 256, K = 5120;
  std::mt19937 rng(20261125u);
  std::normal_distribution<float> dist(0.0f, 0.25f);
  std::vector<float> x_f32(static_cast<std::size_t>(M * K));
  for (float& v : x_f32) v = dist(rng);

  std::vector<std::uint8_t> nv_p(static_cast<std::size_t>(N * (K / 2)));
  std::vector<std::uint8_t> nv_s(static_cast<std::size_t>(N * (K / 16)));
  for (auto& b : nv_p) b = static_cast<std::uint8_t>(rng() & 0xffu);
  for (auto& s : nv_s) s = static_cast<std::uint8_t>(0x38u + (rng() % 20));
  constexpr float kWs2 = 0.015f;

  std::vector<std::uint8_t> fp8_w(static_cast<std::size_t>(N * K));
  for (auto& b : fp8_w) b = quantize_e4m3fn(static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f);
  constexpr float kFp8Ws = 0.025f;
  constexpr float kFp8In = 0.125f;

  std::vector<std::uint16_t> bf_w(static_cast<std::size_t>(N * K));
  for (auto& b : bf_w) b = f32_to_bf16(dist(rng));

  DevBuf<float> d_x(x_f32.size()), d_xq(x_f32.size()), d_out_ref(static_cast<std::size_t>(M * N)),
      d_out_batched(static_cast<std::size_t>(M * N)), d_out_m1_row0(N), d_out_m1_row16(N);
  DevBuf<std::uint8_t> d_nv_p(nv_p.size()), d_nv_s(nv_s.size()), d_fp8_w(fp8_w.size());
  DevBuf<std::uint16_t> d_bf_w(bf_w.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x_f32.data(), x_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nv_p.ptr, nv_p.data(), nv_p.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nv_s.ptr, nv_s.data(), nv_s.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fp8_w.ptr, fp8_w.data(), fp8_w.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bf_w.ptr, bf_w.data(), bf_w.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

  auto dev_max_abs = [&](const float* a_dev, const float* b_dev, std::size_t count) -> float {
    std::vector<float> a(count), b(count);
    CUDA_CHECK(cudaMemcpy(a.data(), a_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), b_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    float mx = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
      if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
      mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
  };

  Qwen38GemmParityReport rep;
  float max_batch_inv = 0.0f;

  // 1. NVFP4 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  qwen38_cuda_nvfp4_linear_f32(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr + static_cast<std::size_t>(16 * K), d_nv_p.ptr, d_nv_s.ptr, kWs2, 1, N,
                                       K, d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.nvfp4_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 2. Calibrated FP8 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  qwen38_cuda_fp8_linear_f32(d_x.ptr, d_fp8_w.ptr, kFp8Ws, kFp8In, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_quant_fp8_act_f32(d_x.ptr, kFp8In, static_cast<std::size_t>(M * K), d_xq.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr, d_fp8_w.ptr, kFp8Ws, 0.0f, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr, d_fp8_w.ptr, kFp8Ws, 0.0f, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr + static_cast<std::size_t>(16 * K), d_fp8_w.ptr, kFp8Ws, 0.0f, 1, N, K,
                                     d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.fp8_cal_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 3. BF16 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  const auto* d_bf_w_ptr = reinterpret_cast<const __nv_bfloat16*>(d_bf_w.ptr);
  qwen38_cuda_bf16_linear_f32(d_x.ptr, d_bf_w_ptr, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr, d_bf_w_ptr, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr, d_bf_w_ptr, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr + static_cast<std::size_t>(16 * K), d_bf_w_ptr, 1, N, K,
                                      d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.bf16_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 4. GDN exact recurrent prefill invariance from non-zero initial state:
  // whole (273, 64 KiB/head smem across 64/256 boundaries), ragged+boundary splits {17,47,192,17},
  // forced same-order global fallback {64,192,17}, 273x M=1 seq prefill, and 273x Gate 1 recurrent_step.
  constexpr int kGdnSeq = 273, kGdnKh = 2, kGdnVh = 6, kGdnKd = 128, kGdnVd = 128;
  constexpr int kGdnKdTot = kGdnKh * kGdnKd;
  constexpr int kGdnVdTot = kGdnVh * kGdnVd;
  constexpr int kGdnCd = 2 * kGdnKdTot + kGdnVdTot;
  constexpr std::size_t kGdnStateElems =
      static_cast<std::size_t>(kGdnVh) * static_cast<std::size_t>(kGdnKd) * static_cast<std::size_t>(kGdnVd);

  std::vector<float> h_gdn_conv(static_cast<std::size_t>(kGdnSeq * kGdnCd));
  std::vector<float> h_gdn_z(static_cast<std::size_t>(kGdnSeq * kGdnVdTot));
  std::vector<float> h_gdn_a(static_cast<std::size_t>(kGdnSeq * kGdnVh));
  std::vector<float> h_gdn_b(static_cast<std::size_t>(kGdnSeq * kGdnVh));
  std::vector<float> h_gdn_alog(static_cast<std::size_t>(kGdnVh));
  std::vector<float> h_gdn_dtb(static_cast<std::size_t>(kGdnVh));
  std::vector<float> h_gdn_nw(static_cast<std::size_t>(kGdnVd));
  std::vector<float> h_gdn_st_init(kGdnStateElems);
  for (float& v : h_gdn_conv) v = dist(rng);
  for (float& v : h_gdn_z) v = dist(rng);
  for (float& v : h_gdn_a) v = dist(rng) * 4.0f;
  for (float& v : h_gdn_b) v = dist(rng) * 4.0f;
  for (int i = 0; i < kGdnVh; ++i) {
    h_gdn_alog[static_cast<std::size_t>(i)] = -2.0f + 1.3f * static_cast<float>(i);
    h_gdn_dtb[static_cast<std::size_t>(i)] = -3.0f + 2.6f * static_cast<float>(i);
  }
  for (float& v : h_gdn_nw) v = 1.0f + dist(rng) * 0.2f;
  for (float& v : h_gdn_st_init) v = dist(rng) * 0.1f;

  DevBuf<float> d_gdn_conv(h_gdn_conv.size()), d_gdn_z(h_gdn_z.size()), d_gdn_a(h_gdn_a.size()),
      d_gdn_b(h_gdn_b.size()), d_gdn_alog(h_gdn_alog.size()), d_gdn_dtb(h_gdn_dtb.size()), d_gdn_nw(h_gdn_nw.size()),
      d_gdn_st_whole(kGdnStateElems), d_gdn_st_split(kGdnStateElems), d_gdn_st_fb(kGdnStateElems),
      d_gdn_st_seq1(kGdnStateElems), d_gdn_st_gate1(kGdnStateElems),
      d_gdn_scratch(static_cast<std::size_t>(2 * kGdnSeq * kGdnKdTot)), d_gdn_out_whole(h_gdn_z.size()),
      d_gdn_out_split(h_gdn_z.size()), d_gdn_out_fb(h_gdn_z.size()), d_gdn_out_seq1(h_gdn_z.size()),
      d_gdn_out_gate1(h_gdn_z.size());

  CUDA_CHECK(cudaMemcpy(d_gdn_conv.ptr, h_gdn_conv.data(), h_gdn_conv.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_z.ptr, h_gdn_z.data(), h_gdn_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_a.ptr, h_gdn_a.data(), h_gdn_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_b.ptr, h_gdn_b.data(), h_gdn_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_alog.ptr, h_gdn_alog.data(), h_gdn_alog.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_dtb.ptr, h_gdn_dtb.data(), h_gdn_dtb.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_nw.ptr, h_gdn_nw.data(), h_gdn_nw.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (float* st_dst :
       {d_gdn_st_whole.ptr, d_gdn_st_split.ptr, d_gdn_st_fb.ptr, d_gdn_st_seq1.ptr, d_gdn_st_gate1.ptr}) {
    CUDA_CHECK(cudaMemcpy(st_dst, h_gdn_st_init.data(), kGdnStateElems * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Whole sequence across 64/256 boundaries.
  qwen38_cuda_gdn_recurrent_prefill_f32(d_gdn_conv.ptr, d_gdn_z.ptr, d_gdn_a.ptr, d_gdn_b.ptr, d_gdn_alog.ptr,
                                        d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, kGdnSeq, kGdnKh, kGdnVh, kGdnKd, kGdnVd,
                                        d_gdn_st_whole.ptr, d_gdn_scratch.ptr, d_gdn_out_whole.ptr, ctx.stream);

  // Ragged + 64/256 boundary splits: {17, 47, 192, 17} -> cumulative [17, 64, 256, 273].
  {
    int off = 0;
    for (int slen : {17, 47, 192, 17}) {
      qwen38_cuda_gdn_recurrent_prefill_f32(
          d_gdn_conv.ptr + static_cast<std::size_t>(off * kGdnCd),
          d_gdn_z.ptr + static_cast<std::size_t>(off * kGdnVdTot),
          d_gdn_a.ptr + static_cast<std::size_t>(off * kGdnVh),
          d_gdn_b.ptr + static_cast<std::size_t>(off * kGdnVh), d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f,
          slen, kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_split.ptr, d_gdn_scratch.ptr,
          d_gdn_out_split.ptr + static_cast<std::size_t>(off * kGdnVdTot), ctx.stream);
      off += slen;
    }
  }

  // Forced same-order global fallback splits {64, 192, 17}.
  {
    int off = 0;
    for (int slen : {64, 192, 17}) {
      qwen38_cuda_gdn_recurrent_prefill_f32(
          d_gdn_conv.ptr + static_cast<std::size_t>(off * kGdnCd),
          d_gdn_z.ptr + static_cast<std::size_t>(off * kGdnVdTot),
          d_gdn_a.ptr + static_cast<std::size_t>(off * kGdnVh),
          d_gdn_b.ptr + static_cast<std::size_t>(off * kGdnVh), d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f,
          slen, kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_fb.ptr, d_gdn_scratch.ptr,
          d_gdn_out_fb.ptr + static_cast<std::size_t>(off * kGdnVdTot), ctx.stream, true);
      off += slen;
    }
  }

  // M=1 sequential prefill + Gate 1 recurrent_step reference.
  for (int t = 0; t < kGdnSeq; ++t) {
    const float* conv_t = d_gdn_conv.ptr + static_cast<std::size_t>(t * kGdnCd);
    const float* z_t = d_gdn_z.ptr + static_cast<std::size_t>(t * kGdnVdTot);
    const float* a_t = d_gdn_a.ptr + static_cast<std::size_t>(t * kGdnVh);
    const float* b_t = d_gdn_b.ptr + static_cast<std::size_t>(t * kGdnVh);
    qwen38_cuda_gdn_recurrent_prefill_f32(conv_t, z_t, a_t, b_t, d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, 1,
                                          kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_seq1.ptr, d_gdn_scratch.ptr,
                                          d_gdn_out_seq1.ptr + static_cast<std::size_t>(t * kGdnVdTot), ctx.stream);
    qwen38_cuda_gdn_recurrent_step_f32(conv_t, conv_t + kGdnKdTot, conv_t + 2 * kGdnKdTot, z_t, a_t, b_t,
                                       d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, kGdnKh, kGdnVh, kGdnKd,
                                       kGdnVd, d_gdn_st_gate1.ptr, d_gdn_scratch.ptr,
                                       d_gdn_out_gate1.ptr + static_cast<std::size_t>(t * kGdnVdTot), ctx.stream);
  }
  ctx.synchronize();

  rep.batch_invariance_max_abs = max_batch_inv;
  float gdn_bi = 0.0f;
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_whole.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_whole.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_split.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_split.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_fb.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_fb.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  rep.gdn_batch_invariance_max_abs = gdn_bi;
  rep.gdn_prefill_vs_step_max_abs = std::max(dev_max_abs(d_gdn_out_whole.ptr, d_gdn_out_gate1.ptr, h_gdn_z.size()),
                                             dev_max_abs(d_gdn_st_whole.ptr, d_gdn_st_gate1.ptr, kGdnStateElems));

  // 5. Fast canonical-64 GQA attention parity vs reference and 0-ULP invariance across M=1/M=5/M=273 (#883).
  {
    constexpr int kAttnSeq = 273, kNh = 24, kNkvh = 4, kHd = 256, kRd = 64;
    constexpr int kQgDim = kNh * kHd * 2;
    constexpr int kKvDim = kNkvh * kHd;
    constexpr int kOutDim = kNh * kHd;
    std::vector<float> h_qg(static_cast<std::size_t>(kAttnSeq * kQgDim));
    std::vector<float> h_kr(static_cast<std::size_t>(kAttnSeq * kKvDim));
    std::vector<float> h_vr(static_cast<std::size_t>(kAttnSeq * kKvDim));
    std::vector<std::uint16_t> h_qnw(kHd), h_knw(kHd);
    for (float& v : h_qg) v = dist(rng);
    for (float& v : h_kr) v = dist(rng);
    for (float& v : h_vr) v = dist(rng);
    for (int i = 0; i < kHd; ++i) {
      h_qnw[static_cast<std::size_t>(i)] = f32_to_bf16(dist(rng) * 0.1f);
      h_knw[static_cast<std::size_t>(i)] = f32_to_bf16(dist(rng) * 0.1f);
    }

    DevBuf<float> d_qg(h_qg.size()), d_kr_orig(h_kr.size()), d_kr_tmp(h_kr.size()), d_vr(h_vr.size());
    DevBuf<std::uint16_t> d_qnw(kHd), d_knw(kHd);
    DevBuf<__nv_bfloat16> d_kc(static_cast<std::size_t>(kAttnSeq * kKvDim)),
        d_vc(static_cast<std::size_t>(kAttnSeq * kKvDim));
    DevBuf<float> d_qt(static_cast<std::size_t>(kAttnSeq * kOutDim)),
        d_gt(static_cast<std::size_t>(kAttnSeq * kOutDim)),
        d_out_ref(static_cast<std::size_t>(kAttnSeq * kOutDim)),
        d_out_fast_whole(static_cast<std::size_t>(kAttnSeq * kOutDim)),
        d_out_fast_split(static_cast<std::size_t>(kAttnSeq * kOutDim)),
        d_out_fast_seq1(static_cast<std::size_t>(kAttnSeq * kOutDim));
    const std::size_t sp_floats = qwen38_cuda_full_attn_split_scratch_floats(kNh, kAttnSeq, kHd);
    DevBuf<float> d_sp(sp_floats);

    CUDA_CHECK(cudaMemcpy(d_qg.ptr, h_qg.data(), h_qg.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kr_orig.ptr, h_kr.data(), h_kr.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vr.ptr, h_vr.data(), h_vr.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qnw.ptr, h_qnw.data(), kHd * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_knw.ptr, h_knw.data(), kHd * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    const auto* qnw_bf = reinterpret_cast<const __nv_bfloat16*>(d_qnw.ptr);
    const auto* knw_bf = reinterpret_cast<const __nv_bfloat16*>(d_knw.ptr);

    // Reference whole sequence.
    CUDA_CHECK(cudaMemcpyAsync(d_kr_tmp.ptr, d_kr_orig.ptr, h_kr.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                               ctx.stream));
    qwen38_cuda_full_attn_gated_f32_kvbf16(d_qg.ptr, d_kr_tmp.ptr, d_vr.ptr, qnw_bf, knw_bf, 1e-6f, 10000000.0f,
                                           kAttnSeq, 0, kNh, kNkvh, kHd, kRd, d_kc.ptr, d_vc.ptr, kAttnSeq, d_qt.ptr,
                                           d_gt.ptr, d_out_ref.ptr, ctx.stream, nullptr, 0, false);

    // Fast whole sequence (tiled prefill).
    CUDA_CHECK(cudaMemcpyAsync(d_kr_tmp.ptr, d_kr_orig.ptr, h_kr.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                               ctx.stream));
    qwen38_cuda_full_attn_gated_f32_kvbf16(d_qg.ptr, d_kr_tmp.ptr, d_vr.ptr, qnw_bf, knw_bf, 1e-6f, 10000000.0f,
                                           kAttnSeq, 0, kNh, kNkvh, kHd, kRd, d_kc.ptr, d_vc.ptr, kAttnSeq, d_qt.ptr,
                                           d_gt.ptr, d_out_fast_whole.ptr, ctx.stream, d_sp.ptr, sp_floats, true);

    // Fast ragged splits {17, 47, 5, 192, 12} covering 64/128/256 boundaries and short-tail split-KV.
    CUDA_CHECK(cudaMemcpyAsync(d_kr_tmp.ptr, d_kr_orig.ptr, h_kr.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                               ctx.stream));
    {
      int off = 0;
      for (int slen : {17, 47, 5, 192, 12}) {
        qwen38_cuda_full_attn_gated_f32_kvbf16(
            d_qg.ptr + static_cast<std::size_t>(off * kQgDim), d_kr_tmp.ptr + static_cast<std::size_t>(off * kKvDim),
            d_vr.ptr + static_cast<std::size_t>(off * kKvDim), qnw_bf, knw_bf, 1e-6f, 10000000.0f, slen, off, kNh,
            kNkvh, kHd, kRd, d_kc.ptr, d_vc.ptr, kAttnSeq, d_qt.ptr, d_gt.ptr,
            d_out_fast_split.ptr + static_cast<std::size_t>(off * kOutDim), ctx.stream, d_sp.ptr, sp_floats, true);
        off += slen;
      }
    }

    // Fast 273x M=1 split-KV decode.
    CUDA_CHECK(cudaMemcpyAsync(d_kr_tmp.ptr, d_kr_orig.ptr, h_kr.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                               ctx.stream));
    for (int t = 0; t < kAttnSeq; ++t) {
      qwen38_cuda_full_attn_gated_f32_kvbf16(
          d_qg.ptr + static_cast<std::size_t>(t * kQgDim), d_kr_tmp.ptr + static_cast<std::size_t>(t * kKvDim),
          d_vr.ptr + static_cast<std::size_t>(t * kKvDim), qnw_bf, knw_bf, 1e-6f, 10000000.0f, 1, t, kNh, kNkvh, kHd,
          kRd, d_kc.ptr, d_vc.ptr, kAttnSeq, d_qt.ptr, d_gt.ptr,
          d_out_fast_seq1.ptr + static_cast<std::size_t>(t * kOutDim), ctx.stream, d_sp.ptr, sp_floats, true);
    }
    ctx.synchronize();

    const std::size_t tot_out = static_cast<std::size_t>(kAttnSeq * kOutDim);
    rep.attn_fast_vs_ref_max_abs = dev_max_abs(d_out_ref.ptr, d_out_fast_whole.ptr, tot_out);
    float attn_bi = 0.0f;
    attn_bi = std::max(attn_bi, dev_max_abs(d_out_fast_whole.ptr, d_out_fast_seq1.ptr, tot_out));
    attn_bi = std::max(attn_bi, dev_max_abs(d_out_fast_split.ptr, d_out_fast_seq1.ptr, tot_out));
    rep.attn_batch_invariance_max_abs = attn_bi;
  }

  rep.passed = (rep.nvfp4_max_abs <= 1e-4f && rep.fp8_cal_max_abs <= 1e-4f && rep.bf16_max_abs <= 1e-4f &&
                rep.batch_invariance_max_abs == 0.0f && rep.gdn_prefill_vs_step_max_abs <= 1e-6f &&
                rep.gdn_batch_invariance_max_abs == 0.0f && rep.attn_fast_vs_ref_max_abs <= 1e-4f &&
                rep.attn_batch_invariance_max_abs == 0.0f);
  return rep;
}

Qwen38DenseBackend::Qwen38DenseBackend(const Qwen38DenseLoadOptions& options)
    : impl_(std::make_unique<Impl>(options.device_id)) {
  if (options.max_ctx <= 0 || options.prefill_chunk <= 0) {
    throw std::invalid_argument("Qwen38DenseBackend max_ctx and prefill_chunk must be > 0");
  }
  impl_->opts = options;

  const auto t_sha0 = std::chrono::steady_clock::now();
  if (options.verify_pins) {
    if (options.expected_pins.has_value()) {
      impl_->manifest = Qwen38Manifest::inspect_pinned(options.model_dir, *options.expected_pins, options.revision);
    } else if (const auto def_spec = qwen38_default_pin_spec_for_dir(options.model_dir); def_spec.has_value()) {
      impl_->manifest = Qwen38Manifest::inspect_pinned(options.model_dir, *def_spec, options.revision);
    } else {
      impl_->manifest = Qwen38Manifest::inspect(options.model_dir, false, options.revision);
    }
  } else {
    impl_->manifest = Qwen38Manifest::inspect(options.model_dir, false, options.revision);
  }
  const double sha_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sha0).count();

  const auto& m = impl_->manifest;
  const auto& c = m.config;
  c.validate();
  if (options.max_ctx > c.max_position_embeddings) {
    throw std::invalid_argument("max_ctx exceeds model max_position_embeddings");
  }

  ingine::model::RuntimeOptions ropt;
  ropt.model_dir = options.model_dir;
  ropt.max_ctx = options.max_ctx;
  ropt.prefill_chunk = options.prefill_chunk;
  ropt.dense_fp8 = (c.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4);
  ropt.prefix_cache = options.prefix_cache;
  ropt.max_batch = 1;
  profile_ = make_qwen38_worker_profile(m, ropt, "dense_bf16", "dense");
  profile_.state_identity.act_dtype = ingine::model::ActivationDtype::Fp32;
  profile_.state_identity.numeric_abi =
      std::string(!m.elastic_profile.empty()
                      ? ("act=f32;elastic=" + m.elastic_profile + ";gdn=f32;kv=bf16")
                      : ((c.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4)
                             ? "act=f32;attn=fp8_e4m3_cal;mlp=nvfp4_w4a32;gdn=f32;kv=bf16"
                             : "act=f32;attn=bf16;mlp=bf16;gdn=f32;kv=bf16")) +
      (options.use_fast_attn ? ";attn_kernel=gqa64" : ";attn_kernel=ref");
  profile_.state_identity.kv_format = "gdn_f32_conv_f32_kv_bf16";

  load_stats_.pins_verified =
      options.verify_pins && m.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified &&
      m.checkpoint.shard_digests_verified;
  load_stats_.cold_manifest_sha_ms = sha_ms;

  std::size_t vram_free_before = 0, vram_total = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_before, &vram_total));

  const auto t_ld0 = std::chrono::steady_clock::now();

  // Plan device arena + small F32/scalar staging across every active text tensor.
  std::uint64_t dev_arena_bytes = 0;
  std::unordered_map<std::string, CopySpec> copy_specs;
  std::deque<HostScalarEntry> scalar_entries;
  std::deque<HostF32VecEntry> f32_entries;

  auto plan_dev = [&](const std::string& name) -> std::uint64_t {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing active tensor: " + name);
    const std::uint64_t off = align_up(dev_arena_bytes);
    dev_arena_bytes = off + it->second.byte_size();
    copy_specs.emplace(name, CopySpec{&it->second, true, off, nullptr});
    return off;
  };

  auto plan_scalar = [&](const std::string& name, float* dst_field) {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing scalar tensor: " + name);
    scalar_entries.push_back(HostScalarEntry{dst_field, it->second.dtype, {}});
    copy_specs.emplace(name, CopySpec{&it->second, false, 0, &scalar_entries.back().raw});
  };

  std::size_t f32_total_elems = 0;
  auto plan_f32_vec = [&](const std::string& name, std::size_t elems) -> std::size_t {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing 1D GDN tensor: " + name);
    const std::size_t off = f32_total_elems;
    f32_total_elems += elems;
    f32_entries.push_back(HostF32VecEntry{off, elems, it->second.dtype, name, {}});
    copy_specs.emplace(name, CopySpec{&it->second, false, 0, &f32_entries.back().raw});
    return off;
  };

  struct PlannedProj {
    DevProj* dst = nullptr;
    std::uint64_t main_off = 0;
    std::uint64_t scale_off = 0;
  };
  std::vector<PlannedProj> planned_projs;
  planned_projs.reserve(static_cast<std::size_t>(c.num_hidden_layers) * 9u + 1u);

  const bool is_elastic = !m.elastic_profile.empty();
  auto plan_proj = [&](const std::string& prefix, int out_f, int in_f, Qwen38LinearQuantKind default_q, DevProj* dst) {
    const Qwen38LinearQuantKind q = m.quant_for(prefix, default_q);
    dst->out_features = out_f;
    dst->in_features = in_f;
    dst->quant = q;
    PlannedProj pp;
    pp.dst = dst;
    pp.main_off = plan_dev(prefix + ".weight");
    const bool has_in_scale = m.active_tensors.find(prefix + ".input_scale") != m.active_tensors.end();
    if (q == Qwen38LinearQuantKind::Fp8E4M3) {
      plan_scalar(prefix + ".weight_scale", &dst->weight_scale);
      if (!is_elastic || has_in_scale) plan_scalar(prefix + ".input_scale", &dst->input_scale);
    } else if (q == Qwen38LinearQuantKind::Nvfp4 || q == Qwen38LinearQuantKind::Atlas3 ||
               q == Qwen38LinearQuantKind::Atlas2) {
      pp.scale_off = plan_dev(prefix + ".weight_scale");
      plan_scalar(prefix + ".weight_scale_2", &dst->weight_scale_2);
      if (!is_elastic || has_in_scale) plan_scalar(prefix + ".input_scale", &dst->input_scale);
    }
    planned_projs.push_back(pp);
  };

  impl_->embed_quant = m.quant_for("model.language_model.embed_tokens", Qwen38LinearQuantKind::Bf16);
  const std::uint64_t embed_off = plan_dev("model.language_model.embed_tokens.weight");
  std::uint64_t embed_scale_off = 0;
  if (impl_->embed_quant == Qwen38LinearQuantKind::Fp8E4M3) {
    plan_scalar("model.language_model.embed_tokens.weight_scale", &impl_->embed_scale);
  } else if (impl_->embed_quant == Qwen38LinearQuantKind::Nvfp4) {
    embed_scale_off = plan_dev("model.language_model.embed_tokens.weight_scale");
    plan_scalar("model.language_model.embed_tokens.weight_scale_2", &impl_->embed_scale);
  }
  const std::uint64_t norm_off = plan_dev("model.language_model.norm.weight");
  if (!c.tie_word_embeddings) {
    plan_proj("lm_head", c.vocab_size, c.hidden_size, c.lm_head_quant(), &impl_->lm_head);
  }

  impl_->layers.resize(static_cast<std::size_t>(c.num_hidden_layers));
  struct LayerOffsets {
    std::uint64_t in_norm = 0, post_norm = 0, q_norm = 0, k_norm = 0;
    std::size_t a_log = 0, dt_bias = 0, conv1d = 0, lin_norm = 0;
  };
  std::vector<LayerOffsets> layer_offs(static_cast<std::size_t>(c.num_hidden_layers));

  for (int li = 0; li < c.num_hidden_layers; ++li) {
    auto& l = impl_->layers[static_cast<std::size_t>(li)];
    auto& lo = layer_offs[static_cast<std::size_t>(li)];
    l.kind = c.layer_types[static_cast<std::size_t>(li)];
    const std::string lp = "model.language_model.layers." + std::to_string(li);
    lo.in_norm = plan_dev(lp + ".input_layernorm.weight");
    lo.post_norm = plan_dev(lp + ".post_attention_layernorm.weight");
    if (l.kind == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      lo.a_log = plan_f32_vec(attn + ".A_log", static_cast<std::size_t>(c.linear_num_value_heads));
      lo.dt_bias = plan_f32_vec(attn + ".dt_bias", static_cast<std::size_t>(c.linear_num_value_heads));
      lo.conv1d = plan_f32_vec(
          attn + ".conv1d.weight",
          static_cast<std::size_t>(c.linear_conv_dim()) * static_cast<std::size_t>(c.linear_conv_kernel_dim));
      lo.lin_norm = plan_f32_vec(attn + ".norm.weight", static_cast<std::size_t>(c.linear_value_head_dim));
      plan_proj(attn + ".in_proj_a", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16, &l.in_proj_a);
      plan_proj(attn + ".in_proj_b", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16, &l.in_proj_b);
      plan_proj(attn + ".in_proj_qkv", c.linear_conv_dim(), c.hidden_size, c.linear_attn_quant(), &l.in_proj_qkv);
      plan_proj(attn + ".in_proj_z", c.linear_value_dim(), c.hidden_size, c.linear_attn_quant(), &l.in_proj_z);
      plan_proj(attn + ".out_proj", c.hidden_size, c.linear_value_dim(), c.linear_attn_quant(), &l.out_proj);
    } else {
      const std::string attn = lp + ".self_attn";
      lo.q_norm = plan_dev(attn + ".q_norm.weight");
      lo.k_norm = plan_dev(attn + ".k_norm.weight");
      plan_proj(attn + ".q_proj", c.full_q_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.q_proj);
      plan_proj(attn + ".k_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.k_proj);
      plan_proj(attn + ".v_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.v_proj);
      plan_proj(attn + ".o_proj", c.hidden_size, c.num_attention_heads * c.head_dim, c.full_attn_quant(), &l.o_proj);
    }
    plan_proj(lp + ".mlp.gate_proj", c.intermediate_size, c.hidden_size, c.mlp_quant(), &l.gate_proj);
    plan_proj(lp + ".mlp.up_proj", c.intermediate_size, c.hidden_size, c.mlp_quant(), &l.up_proj);
    plan_proj(lp + ".mlp.down_proj", c.hidden_size, c.intermediate_size, c.mlp_quant(), &l.down_proj);
  }

  impl_->weights_arena.allocate(dev_arena_bytes);
  impl_->f32_arena.allocate(f32_total_elems);

  // Group CopySpecs by shard and stream directly to device / tiny staging vectors.
  std::unordered_map<std::string, std::vector<const CopySpec*>> by_shard;
  for (const auto& [name, spec] : copy_specs) {
    by_shard[spec.info->shard_file].push_back(&spec);
  }

  const bool is_pinned = load_stats_.pins_verified;
  for (const std::string& shard : m.shard_files) {
    const auto shard_it = by_shard.find(shard);
    if (shard_it == by_shard.end() || shard_it->second.empty()) {
      if (is_pinned) {
        (void)m.stream_verified_file(options.model_dir, shard, {}, {}, {}, 4ull << 20);
      }
      continue;
    }

    const auto& specs = shard_it->second;
    std::array<std::uint8_t, 8> hdr_bytes{};
    std::uint8_t hdr_have = 0;
    bool targets_built = false;
    std::vector<ActiveTarget> targets;
    std::size_t next_target = 0;

    auto on_chunk = [&](std::uint64_t off, std::span<const std::uint8_t> chunk) {
      if (off < 8) {
        const std::uint64_t take = std::min<std::uint64_t>(8 - off, chunk.size());
        std::memcpy(hdr_bytes.data() + off, chunk.data(), static_cast<std::size_t>(take));
        hdr_have = static_cast<std::uint8_t>(std::max<std::uint64_t>(hdr_have, off + take));
      }
      if (!targets_built && hdr_have == 8) {
        const std::uint64_t data_base = 8ull + parse_header_len_le(hdr_bytes);
        targets.reserve(specs.size());
        for (const CopySpec* sp : specs) {
          if (!sp->to_device) sp->host_dst->assign(sp->info->byte_size(), 0u);
          targets.push_back(ActiveTarget{data_base + sp->info->data_offset_begin,
                                         data_base + sp->info->data_offset_end, sp->to_device, sp->dev_offset,
                                         sp->host_dst});
        }
        std::sort(targets.begin(), targets.end(),
                  [](const ActiveTarget& a, const ActiveTarget& b) { return a.file_begin < b.file_begin; });
        targets_built = true;
      }
      if (!targets_built) return;
      const std::uint64_t c0 = off;
      const std::uint64_t c1 = off + chunk.size();
      while (next_target < targets.size() && targets[next_target].file_end <= c0) ++next_target;
      for (std::size_t ti = next_target; ti < targets.size(); ++ti) {
        const ActiveTarget& tg = targets[ti];
        if (tg.file_begin >= c1) break;
        const std::uint64_t s0 = std::max(c0, tg.file_begin);
        const std::uint64_t s1 = std::min(c1, tg.file_end);
        const std::size_t n = static_cast<std::size_t>(s1 - s0);
        const std::uint8_t* src = chunk.data() + static_cast<std::size_t>(s0 - c0);
        const std::uint64_t dst_off = s0 - tg.file_begin;
        if (tg.to_device) {
          CUDA_CHECK(cudaMemcpy(impl_->weights_arena.ptr + tg.dev_offset + dst_off, src, n, cudaMemcpyHostToDevice));
        } else {
          std::memcpy(tg.host_dst->data() + dst_off, src, n);
        }
      }
    };

    auto on_abort = [&]() {
      impl_->weights_arena.free();
      impl_->f32_arena.free();
    };

    if (is_pinned) {
      (void)m.stream_verified_file(options.model_dir, shard, on_chunk, {}, on_abort, 4ull << 20);
    } else {
      stream_unpinned_shard(m, options.model_dir, shard, on_chunk, on_abort);
    }
  }

  // Decode staged scalar scales and 1D F32 GDN arrays (ONLY after all shards verified/loaded).
  for (const auto& se : scalar_entries) {
    *se.target_field = decode_scalar_bytes(se.raw, se.dtype, "scalar");
  }
  std::vector<float> host_f32_all(f32_total_elems, 0.0f);
  for (const auto& fe : f32_entries) {
    decode_f32_vec_bytes(fe.raw, fe.dtype, fe.elems, fe.name, host_f32_all, fe.f32_offset);
  }
  if (f32_total_elems > 0) {
    CUDA_CHECK(cudaMemcpy(impl_->f32_arena.ptr, host_f32_all.data(), f32_total_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // Resolve device pointers inside arenas.
  const std::uint8_t* wbase = impl_->weights_arena.ptr;
  const float* fbase = impl_->f32_arena.ptr;
  if (impl_->embed_quant == Qwen38LinearQuantKind::Bf16) {
    impl_->embed_tokens = reinterpret_cast<const __nv_bfloat16*>(wbase + embed_off);
  } else if (impl_->embed_quant == Qwen38LinearQuantKind::Fp8E4M3) {
    impl_->embed_u8 = wbase + embed_off;
  } else {
    impl_->embed_u8 = wbase + embed_off;
    impl_->embed_scales = wbase + embed_scale_off;
  }
  impl_->final_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + norm_off);
  for (const auto& pp : planned_projs) {
    if (pp.dst->quant == Qwen38LinearQuantKind::Bf16) {
      pp.dst->w_bf16 = reinterpret_cast<const __nv_bfloat16*>(wbase + pp.main_off);
    } else if (pp.dst->quant == Qwen38LinearQuantKind::Fp8E4M3) {
      pp.dst->w_u8 = wbase + pp.main_off;
    } else {
      pp.dst->w_u8 = wbase + pp.main_off;
      pp.dst->w_scales = wbase + pp.scale_off;
    }
  }
  if (c.tie_word_embeddings) {
    impl_->lm_head.out_features = c.vocab_size;
    impl_->lm_head.in_features = c.hidden_size;
    impl_->lm_head.quant = Qwen38LinearQuantKind::Bf16;
    impl_->lm_head.w_bf16 = impl_->embed_tokens;
  }

  for (int li = 0; li < c.num_hidden_layers; ++li) {
    auto& l = impl_->layers[static_cast<std::size_t>(li)];
    const auto& lo = layer_offs[static_cast<std::size_t>(li)];
    l.input_layernorm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.in_norm);
    l.post_attention_layernorm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.post_norm);
    if (l.kind == Qwen38LayerKind::LinearAttention) {
      l.a_log = fbase + lo.a_log;
      l.dt_bias = fbase + lo.dt_bias;
      l.conv1d_weight = fbase + lo.conv1d;
      l.linear_norm_weight = fbase + lo.lin_norm;
    } else {
      l.q_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.q_norm);
      l.k_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.k_norm);
    }
  }

  // Allocate reusable execution scratch + default session.
  const int P = options.prefill_chunk;
  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  const int V = c.vocab_size;
  const int cd = c.linear_conv_dim();
  const int kd = c.linear_key_dim();
  const int vd = c.linear_value_dim();
  const int nhv = c.linear_num_value_heads;
  const int qg_dim = c.full_q_proj_out_dim();
  const int kv_dim = c.full_kv_proj_out_dim();
  const int attn_dim = c.num_attention_heads * c.head_dim;

  const std::size_t max_k_elems =
      static_cast<std::size_t>(P) * static_cast<std::size_t>(std::max({H, I, vd, attn_dim}));

  impl_->act_quant_scratch.allocate(max_k_elems);
  impl_->token_ids_dev.allocate(static_cast<std::size_t>(P));
  impl_->gather_rows_dev.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax));
  impl_->hidden.allocate(static_cast<std::size_t>(P * H));
  impl_->normed.allocate(static_cast<std::size_t>(P * H));
  impl_->residual.allocate(static_cast<std::size_t>(P * H));
  impl_->qkv.allocate(static_cast<std::size_t>(P * cd));
  impl_->conv_out.allocate(static_cast<std::size_t>(P * cd));
  impl_->z_buf.allocate(static_cast<std::size_t>(P * vd));
  impl_->a_buf.allocate(static_cast<std::size_t>(P * nhv));
  impl_->b_buf.allocate(static_cast<std::size_t>(P * nhv));
  impl_->core_out.allocate(static_cast<std::size_t>(P * vd));
  impl_->gdn_scratch.allocate(static_cast<std::size_t>(P) * static_cast<std::size_t>(kd) * 2ull);

  impl_->q_gate.allocate(static_cast<std::size_t>(P * qg_dim));
  impl_->k_raw.allocate(static_cast<std::size_t>(P * kv_dim));
  impl_->v_raw.allocate(static_cast<std::size_t>(P * kv_dim));
  impl_->q_tmp.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->gate_tmp.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->attn_out.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->attn_split_scratch.allocate(
      qwen38_cuda_full_attn_split_scratch_floats(c.num_attention_heads, options.max_ctx, c.head_dim));
  impl_->mlp_gate.allocate(static_cast<std::size_t>(P * I));
  impl_->mlp_up.allocate(static_cast<std::size_t>(P * I));
  impl_->cap_hidden.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax * H));
  impl_->cap_normed.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax * H));
  impl_->logits_f32.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax) * static_cast<std::size_t>(V));
  impl_->host_logits_stage.resize(static_cast<std::size_t>(Impl::kLogitBatchMax) * static_cast<std::size_t>(V));
  impl_->decode_single_tok.resize(1);

  impl_->scratch_bytes =
      (impl_->act_quant_scratch.count + impl_->hidden.count + impl_->normed.count + impl_->residual.count +
       impl_->qkv.count + impl_->conv_out.count + impl_->z_buf.count + impl_->a_buf.count + impl_->b_buf.count +
       impl_->core_out.count + impl_->q_gate.count + impl_->k_raw.count + impl_->v_raw.count + impl_->q_tmp.count +
       impl_->gate_tmp.count + impl_->attn_out.count + impl_->attn_split_scratch.count + impl_->mlp_gate.count +
       impl_->mlp_up.count + impl_->cap_hidden.count + impl_->cap_normed.count + impl_->gdn_scratch.count +
       impl_->logits_f32.count) *
          sizeof(float) +
      (impl_->token_ids_dev.count + impl_->gather_rows_dev.count) * sizeof(std::int32_t);

  impl_->default_session.reset(new Qwen38DenseSession(*this));
  impl_->max_state_bytes = impl_->default_session->state_bytes();

  if (!options.platos_key.empty()) {
    impl_->platos_key_gpu =
        ingine::PlatosKeyGpu::load(options.platos_key, options.platos_key_metadata, c.hidden_size, impl_->ctx.stream);
    impl_->platos_key_gpu->key()->validate_binding(model_id(), model_family(), c.hidden_size, c.num_hidden_layers);
  }

  std::size_t vram_free_after = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_after, &vram_total));

  struct rusage ru {};
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    load_stats_.host_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
  }
  load_stats_.cold_safetensors_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ld0).count();
  load_stats_.weights_device_bytes = dev_arena_bytes + f32_total_elems * sizeof(float);
  load_stats_.state_device_bytes = impl_->max_state_bytes;
  load_stats_.scratch_device_bytes = impl_->scratch_bytes;
  load_stats_.peak_vram_bytes = vram_free_before > vram_free_after ? (vram_free_before - vram_free_after) : 0ull;
}

Qwen38DenseBackend::~Qwen38DenseBackend() = default;

const Qwen38Manifest& Qwen38DenseBackend::manifest() const noexcept { return impl_->manifest; }

std::string Qwen38DenseBackend::model_id() const { return profile_.model_id; }

int Qwen38DenseBackend::max_model_len() const { return profile_.max_ctx; }

ingine::model::PrefixCacheStats Qwen38DenseBackend::cache_stats() const {
  ingine::model::PrefixCacheStats cs;
  if (impl_->opts.prefix_cache && impl_->default_session) {
    cs.host_entries =
        static_cast<std::uint64_t>(impl_->default_session->checkpoint_index().saved_checkpoint_lengths().size());
    cs.host_used_bytes = static_cast<std::size_t>(impl_->default_session->saved_checkpoint_bytes());
  }
  return cs;
}

void Qwen38DenseBackend::reset_session() { impl_->default_session->reset(); }

std::unique_ptr<ingine::model::BackendSession> Qwen38DenseBackend::create_session() {
  return std::unique_ptr<ingine::model::BackendSession>(new Qwen38DenseSession(*this));
}

ingine::model::BackendSession* Qwen38DenseBackend::acquire_session(int slot) {
  if (slot != 0) return nullptr;
  return impl_->default_session.get();
}

void Qwen38DenseBackend::release_session(int /*slot*/) {}

ingine::model::ResidencyReport Qwen38DenseBackend::residency_report() const {
  ingine::model::ResidencyReport r;
  r.weights_device_bytes = load_stats_.weights_device_bytes;
  r.state_device_bytes = impl_->default_session ? impl_->default_session->state_bytes() : impl_->max_state_bytes;
  r.device_bytes = r.weights_device_bytes + r.state_device_bytes + load_stats_.scratch_device_bytes;
  const std::uint64_t ckpt_bytes = impl_->default_session ? impl_->default_session->saved_checkpoint_bytes() : 0ull;
  struct rusage ru {};
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    r.process_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
  }
  r.host_ram_bytes = std::max(r.process_rss_bytes, ckpt_bytes);
  return r;
}

std::uint64_t Qwen38DenseBackend::evict_caches(std::uint64_t /*target_free_bytes*/) {
  return impl_->default_session ? impl_->default_session->evict_checkpoints() : 0ull;
}

const ingine::model::SessionLookupResult& Qwen38DenseBackend::last_lookup() const noexcept {
  return impl_->last_lookup;
}

double Qwen38DenseBackend::last_lookup_ms() const noexcept { return impl_->last_lookup_ms; }

double Qwen38DenseBackend::last_reset_ms() const noexcept { return impl_->last_reset_ms; }

void Qwen38DenseBackend::configure_883_ablation(bool pre_suffix_checkpoint, bool use_fast_attn) {
  impl_->opts.pre_suffix_checkpoint = pre_suffix_checkpoint;
  impl_->opts.use_fast_attn = use_fast_attn;
  const auto& c = manifest().config;
  profile_.state_identity.numeric_abi =
      std::string((c.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4)
                      ? "act=f32;attn=fp8_e4m3_cal;mlp=nvfp4_w4a32;gdn=f32;kv=bf16"
                      : "act=f32;attn=bf16;mlp=bf16;gdn=f32;kv=bf16") +
      (use_fast_attn ? ";attn_kernel=gqa64" : ";attn_kernel=ref");
  if (impl_->default_session) {
    impl_->default_session->reset();
  }
}

ingine::model::AdmissionEstimate Qwen38DenseBackend::admission_estimate(int max_ctx, int max_batch) const {
  ingine::model::AdmissionEstimate est;
  const auto& c = manifest().config;
  const auto& mb = manifest().memory_budget;
  if (max_batch != 1 || max_ctx <= 0 || max_ctx > c.max_position_embeddings) {
    est.fits_current_budget = false;
    est.reason = "qwen38-dense Phase 1 supports max_batch=1 and 0 < max_ctx <= max_position_embeddings";
    return est;
  }
  const std::uint64_t kv_bytes = static_cast<std::uint64_t>(max_ctx) * mb.kv_bytes_per_token;
  const std::uint64_t state_bytes = mb.gdn_recurrent_bytes + mb.gdn_conv_bytes * 2ull + kv_bytes;
  est.required_device_bytes = load_stats_.weights_device_bytes + state_bytes + load_stats_.scratch_device_bytes;
  std::size_t vram_free = 0, vram_total = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
  est.fits_current_budget = (est.required_device_bytes <= vram_total);
  return est;
}

void Qwen38DenseBackend::evaluate_teacher_forced_mode(std::span<const std::int32_t> token_ids,
                                                      std::span<const int> capture_positions, bool use_chunk_prefill,
                                                      const ingine::model::LogitCallback& on_logits,
                                                      double* prefill_ms_out, Qwen38DenseSession* session,
                                                      bool reset_before) {
  if (token_ids.empty()) throw std::invalid_argument("evaluate_teacher_forced empty token_ids");
  const int V = manifest().config.vocab_size;
  for (std::int32_t tid : token_ids) {
    if (tid < 0 || tid >= V) throw std::invalid_argument("evaluate_teacher_forced token_id out of range");
  }
  std::vector<int> caps(capture_positions.begin(), capture_positions.end());
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  for (int pos : caps) {
    if (pos < 0 || pos >= static_cast<int>(token_ids.size())) {
      throw std::invalid_argument("evaluate_teacher_forced capture position out of bounds");
    }
  }
  Qwen38DenseSession& sess = session ? *session : *impl_->default_session;
  if (reset_before) sess.reset();
  if (reset_before) {
    impl_->forward_range(sess, token_ids, caps, use_chunk_prefill, on_logits, prefill_ms_out);
  } else {
    const int base = static_cast<int>(sess.committed_tokens().size());
    for (int& p : caps) p += base;
    impl_->forward_range(sess, token_ids, caps, use_chunk_prefill, on_logits, prefill_ms_out);
  }
}

void Qwen38DenseBackend::evaluate_teacher_forced_on_session(Qwen38DenseSession& sess,
                                                            const ingine::model::BackendRequest& req,
                                                            std::span<const int> capture_positions,
                                                            const ingine::model::LogitCallback& on_logits,
                                                            double* prefill_ms_out) {
  ingine::model::validate_request_against_capabilities(profile_.capabilities, profile_.max_ctx, req.prompt, 1,
                                                       ingine::model::Sampling{});
  if (!req.token_mask.empty() && req.token_mask.size() != req.prompt.ids.size()) {
    throw std::invalid_argument("token_mask size mismatch");
  }

  std::vector<int> caps(capture_positions.begin(), capture_positions.end());
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  for (int pos : caps) {
    if (pos < 0 || pos >= static_cast<int>(req.prompt.ids.size())) {
      throw std::invalid_argument("evaluate_teacher_forced capture position out of bounds");
    }
  }

  ingine::model::BackendStateIdentity req_id = profile_.state_identity;
  if (!req.cache_namespace.empty()) req_id.cache_namespace = req.cache_namespace;
  req_id.media_sha256 = ingine::model::hash_prompt_media(req.prompt.images);

  const bool allow_reuse = impl_->opts.prefix_cache && req.use_prefix_cache;
  const auto tl0 = std::chrono::steady_clock::now();
  ingine::model::SessionLookupResult lk = ingine::model::restore_session_prefix(
      sess, ingine::model::SessionLookupQuery{req_id, req.prompt.ids, req.token_mask, req.prompt.images, allow_reuse});
  impl_->last_lookup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl0).count();

  const int V = manifest().config.vocab_size;
  if (lk.hit && !caps.empty() && caps.front() < static_cast<int>(lk.reused_tokens)) {
    const bool tip_covers_only_last =
        (caps.size() == 1 || caps[1] >= static_cast<int>(lk.reused_tokens)) &&
        caps.front() == static_cast<int>(lk.reused_tokens) - 1 && sess.state_->index.has_tip_logits() &&
        sess.state_->index.tip_logits_pos() == caps.front() &&
        sess.state_->tip_logits.size() == static_cast<std::size_t>(V);
    if (!tip_covers_only_last) {
      const std::size_t pre_restore_tip = sess.committed_tokens().size();
      const std::size_t prev = sess.restorable_prefix_len(static_cast<std::size_t>(caps.front()));
      if (prev > 0 && sess.try_restore_prefix(prev)) {
        lk.reused_tokens = prev;
        lk.reuse =
            (prev == pre_restore_tip) ? ingine::model::Reuse::Extend : ingine::model::Reuse::Checkpoint;
        lk.reason = (lk.reuse == ingine::model::Reuse::Extend) ? "exact_prefix_match" : "exact_checkpoint_match";
      } else {
        lk = ingine::model::SessionLookupResult{false, 0, ingine::model::Reuse::None,
                                                "capture_before_reused_prefix"};
      }
    }
  }

  impl_->last_lookup = lk;
  if (!lk.hit) {
    sess.reset();
    ingine::model::BackendStateIdentity sess_id = req_id;
    if (!allow_reuse) {
      // Isolate bypassed requests so subsequent prefix-cached requests cannot reuse or extend this session.
      sess_id.checkpoint.verification_level = ingine::model::VerificationLevel::Unverified;
    }
    sess.state_->index.set_identity(std::move(sess_id));
  }

  std::vector<int> rem_caps;
  rem_caps.reserve(caps.size());
  for (int pos : caps) {
    if (lk.hit && pos == static_cast<int>(lk.reused_tokens) - 1 && sess.state_->index.has_tip_logits() &&
        sess.state_->index.tip_logits_pos() == pos && sess.state_->tip_logits.size() == static_cast<std::size_t>(V)) {
      if (on_logits) {
        on_logits(ingine::model::LogitStep{pos, sess.state_->tip_logits});
      }
    } else {
      rem_caps.push_back(pos);
    }
  }

  if (lk.reused_tokens < req.prompt.ids.size()) {
    const std::span<const std::int32_t> suffix_ids = req.prompt.ids.subspan(lk.reused_tokens);
    const std::span<const std::uint8_t> suffix_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>{} : req.token_mask.subspan(lk.reused_tokens);
    impl_->forward_range(sess, suffix_ids, rem_caps, true, on_logits, prefill_ms_out, suffix_mask);
  }
  if (allow_reuse && req_id.is_reusable_with(req_id)) {
    (void)sess.save_checkpoint();
  }
}

void Qwen38DenseBackend::evaluate_teacher_forced(std::span<const std::int32_t> token_ids,
                                                 std::span<const int> capture_positions, bool use_prefix_cache,
                                                 const ingine::model::LogitCallback& on_logits,
                                                 std::span<const std::uint8_t> token_mask) {
  ingine::model::BackendRequest req;
  req.prompt.ids = token_ids;
  req.token_mask = token_mask;
  req.use_prefix_cache = use_prefix_cache;
  evaluate_teacher_forced_on_session(*impl_->default_session, req, capture_positions, on_logits, nullptr);
}

void Qwen38DenseBackend::evaluate_teacher_forced(const ingine::model::BackendRequest& req,
                                                 std::span<const int> capture_positions,
                                                 const ingine::model::LogitCallback& on_logits) {
  evaluate_teacher_forced_on_session(*impl_->default_session, req, capture_positions, on_logits, nullptr);
}

void Qwen38DenseBackend::generate_greedy_exact(std::span<const std::int32_t> prompt_ids, int max_new_tokens,
                                               bool use_chunk_prefill, bool stop_on_eos,
                                               std::vector<std::int32_t>& out_tokens, double* prefill_ms_out,
                                               double* decode_ms_out, Qwen38DenseSession* session,
                                               const ingine::model::LogitCallback& on_step_logits) {
  out_tokens.clear();
  if (prompt_ids.empty() || max_new_tokens <= 0) return;
  const int V = manifest().config.vocab_size;
  for (std::int32_t tid : prompt_ids) {
    if (tid < 0 || tid >= V) throw std::invalid_argument("generate_greedy_exact prompt token out of range");
  }
  Qwen38DenseSession& sess = session ? *session : *impl_->default_session;
  sess.reset();

  std::int32_t next_tok = -1;
  auto pick_argmax = [&](const ingine::model::LogitStep& step) {
    float best_v = -std::numeric_limits<float>::infinity();
    std::int32_t best_id = 0;
    for (int i = 0; i < V; ++i) {
      const float lv = step.logits[static_cast<std::size_t>(i)];
      if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit in greedy generation");
      if (lv > best_v) {
        best_v = lv;
        best_id = i;
      }
    }
    next_tok = best_id;
    if (on_step_logits) on_step_logits(step);
  };

  const std::array<int, 1> last_cap = {static_cast<int>(prompt_ids.size()) - 1};
  impl_->forward_range(sess, prompt_ids, last_cap, use_chunk_prefill, pick_argmax, prefill_ms_out);

  const auto td0 = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    out_tokens.push_back(next_tok);
    if (stop_on_eos && (next_tok == 248044 || next_tok == 248046)) break;
    if (step + 1 == max_new_tokens) break;
    impl_->decode_single_tok[0] = next_tok;
    const std::array<int, 1> cap_cur = {static_cast<int>(sess.committed_tokens().size())};
    impl_->forward_range(sess, impl_->decode_single_tok, cap_cur, false, pick_argmax, nullptr);
  }
  if (decode_ms_out) {
    *decode_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
  }
}

ingine::model::GenerateStats Qwen38DenseBackend::generate_request_on_session(
    Qwen38DenseSession& sess, const ingine::model::BackendRequest& req, int max_new_tokens,
    const ingine::model::Sampling& sampling, std::span<const std::int32_t> stop_ids,
    const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
    const ingine::model::PrefillCallback& on_prefill, const ingine::model::LogitCallback& on_step_logits) {
  if (sampling.logit_masker) {
    throw std::invalid_argument("qwen38-dense does not support token-level logit_masker constrained decoding");
  }
  ingine::model::validate_request_against_capabilities(profile_.capabilities, profile_.max_ctx, req.prompt,
                                                       max_new_tokens, sampling);
  if (!req.token_mask.empty() && req.token_mask.size() != req.prompt.ids.size()) {
    throw std::invalid_argument("BackendRequest token_mask length must equal prompt.ids length");
  }

  int eff_max_new =
      sampling.forced.empty() ? max_new_tokens : std::min<int>(max_new_tokens, static_cast<int>(sampling.forced.size()));
  const int n_prompt = static_cast<int>(req.prompt.ids.size());
  if (n_prompt + std::int64_t(eff_max_new) > profile_.max_ctx) {
    eff_max_new = std::max(0, profile_.max_ctx - n_prompt);
  }
  ingine::model::GenerateStats stats;
  stats.prompt_tokens = n_prompt;
  if (eff_max_new <= 0) return stats;

  const int V = manifest().config.vocab_size;
  std::mt19937_64 rng(sampling.seed.value_or(0x9e3779b97f4a7c15ull));

  auto select_token = [&](std::span<const float> logits) -> std::int32_t {
    if (sampling.temperature <= 0.0f || sampling.top_k == 1) {
      float best_v = -std::numeric_limits<float>::infinity();
      std::int32_t best_id = 0;
      for (int i = 0; i < V; ++i) {
        if (std::find(sampling.banned_tokens.begin(), sampling.banned_tokens.end(), i) != sampling.banned_tokens.end()) {
          continue;
        }
        const float lv = logits[static_cast<std::size_t>(i)];
        if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit during generate()");
        if (lv > best_v) {
          best_v = lv;
          best_id = i;
        }
      }
      return best_id;
    }

    std::vector<std::pair<float, std::int32_t>> cand;
    cand.reserve(static_cast<std::size_t>(sampling.top_k > 0 ? std::min(sampling.top_k, V) : 256));
    const float inv_t = 1.0f / sampling.temperature;
    for (int i = 0; i < V; ++i) {
      if (std::find(sampling.banned_tokens.begin(), sampling.banned_tokens.end(), i) != sampling.banned_tokens.end()) {
        continue;
      }
      const float lv = logits[static_cast<std::size_t>(i)];
      if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit during generate()");
      cand.emplace_back(lv * inv_t, i);
    }
    if (sampling.top_k > 0 && sampling.top_k < static_cast<int>(cand.size())) {
      std::partial_sort(cand.begin(), cand.begin() + sampling.top_k, cand.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
      cand.resize(static_cast<std::size_t>(sampling.top_k));
    } else {
      std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    const float max_l = cand.front().first;
    std::vector<double> probs(cand.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      probs[i] = std::exp(static_cast<double>(cand[i].first - max_l));
      sum += probs[i];
    }
    double cum = 0.0;
    std::size_t keep = cand.size();
    if (sampling.top_p > 0.0f && sampling.top_p < 1.0f) {
      for (std::size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i] / sum;
        if (cum >= static_cast<double>(sampling.top_p)) {
          keep = i + 1;
          break;
        }
      }
    }
    double sub_sum = 0.0;
    for (std::size_t i = 0; i < keep; ++i) sub_sum += probs[i];
    std::uniform_real_distribution<double> u(0.0, sub_sum);
    double r = u(rng);
    for (std::size_t i = 0; i < keep; ++i) {
      r -= probs[i];
      if (r <= 0.0) return cand[i].second;
    }
    return cand[keep - 1].second;
  };

  ingine::model::BackendStateIdentity req_id = profile_.state_identity;
  if (!req.cache_namespace.empty()) req_id.cache_namespace = req.cache_namespace;
  req_id.media_sha256 = ingine::model::hash_prompt_media(req.prompt.images);

  const bool allow_reuse = impl_->opts.prefix_cache && req.use_prefix_cache;
  const auto tl0 = std::chrono::steady_clock::now();
  ingine::model::SessionLookupResult lookup = ingine::model::restore_session_prefix(
      sess, ingine::model::SessionLookupQuery{req_id, req.prompt.ids, req.token_mask, req.prompt.images, allow_reuse});
  impl_->last_lookup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl0).count();

  const std::size_t S = req.prompt.ids.size();
  if (lookup.hit && lookup.reused_tokens == S) {
    const bool tip_ok = sess.state_->index.has_tip_logits() &&
                        sess.state_->index.tip_logits_pos() == static_cast<int>(S) - 1 &&
                        sess.state_->tip_logits.size() == static_cast<std::size_t>(V);
    if (!tip_ok) {
      const std::size_t prev = (S > 1) ? sess.restorable_prefix_len(S - 1) : 0;
      if (prev > 0 && sess.try_restore_prefix(prev)) {
        lookup.reused_tokens = prev;
        lookup.reuse = ingine::model::Reuse::Checkpoint;
        lookup.reason = "exact_checkpoint_match";
      } else {
        lookup = ingine::model::SessionLookupResult{false, 0, ingine::model::Reuse::None, "tip_logits_unavailable"};
      }
    }
  }

  impl_->last_lookup = lookup;
  if (!lookup.hit) {
    sess.reset();
    ingine::model::BackendStateIdentity sess_id = req_id;
    if (!allow_reuse) {
      sess_id.checkpoint.verification_level = ingine::model::VerificationLevel::Unverified;
    }
    sess.state_->index.set_identity(std::move(sess_id));
    stats.restored_bytes = 0ull;
    stats.restore_seconds = 0.0;
  } else if (lookup.reuse == ingine::model::Reuse::Checkpoint) {
    stats.restored_bytes = sess.last_restore_bytes();
    stats.restore_seconds = sess.last_restore_seconds();
  } else {
    stats.restored_bytes = 0ull;
    stats.restore_seconds = 0.0;
  }

  stats.reuse = lookup.reuse;
  stats.reused_tokens = static_cast<int>(lookup.reused_tokens);

  std::int32_t next_tok = 0;
  double pf_ms = 0.0;
  int evictions = 0;
  const bool can_save_ckpt = allow_reuse && req_id.is_reusable_with(req_id);
  const std::size_t pre_suffix_pos =
      (can_save_ckpt && impl_->opts.pre_suffix_checkpoint) ? find_qwen35_pre_suffix_checkpoint(req.prompt.ids) : 0;

  if (lookup.reused_tokens == S) {
    next_tok = select_token(sess.state_->tip_logits);
  } else if (pre_suffix_pos > lookup.reused_tokens && pre_suffix_pos < S) {
    // Checkpoint GDN recurrent+conv state immediately before the trailing Qwen3.5 assistant
    // generation suffix so multi-turn follow-ups that alter thinking/tool formatting reuse
    // the entire prior conversation prefix with zero extra lm_head GEMM (#883).
    const std::size_t pre_len = pre_suffix_pos - lookup.reused_tokens;
    const std::span<const std::int32_t> pre_ids = req.prompt.ids.subspan(lookup.reused_tokens, pre_len);
    const std::span<const std::uint8_t> pre_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>{} : req.token_mask.subspan(lookup.reused_tokens, pre_len);
    impl_->forward_range(sess, pre_ids, {}, true, {}, &pf_ms, pre_mask, false);
    evictions += sess.save_checkpoint();

    const std::span<const std::int32_t> tail_ids = req.prompt.ids.subspan(pre_suffix_pos);
    const std::span<const std::uint8_t> tail_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>{} : req.token_mask.subspan(pre_suffix_pos);
    const std::array<int, 1> last_cap = {static_cast<int>(S) - 1};
    impl_->forward_range(sess, tail_ids, last_cap, true,
                         [&](const ingine::model::LogitStep& st) { next_tok = select_token(st.logits); }, &pf_ms,
                         tail_mask, true);
  } else {
    const std::span<const std::int32_t> suffix_ids = req.prompt.ids.subspan(lookup.reused_tokens);
    const std::span<const std::uint8_t> suffix_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>{} : req.token_mask.subspan(lookup.reused_tokens);
    const std::array<int, 1> last_cap = {static_cast<int>(S) - 1};
    impl_->forward_range(sess, suffix_ids, last_cap, true,
                         [&](const ingine::model::LogitStep& st) { next_tok = select_token(st.logits); }, &pf_ms,
                         suffix_mask);
  }

  if (can_save_ckpt) {
    evictions += sess.save_checkpoint();
  }
  stats.cache_evictions = evictions;
  stats.prefill_seconds = pf_ms / 1000.0;
  if (on_prefill) on_prefill(stats);

  if (on_step_logits) {
    on_step_logits(ingine::model::LogitStep{static_cast<int>(S) - 1, sess.state_->tip_logits});
  }

  const auto td0 = std::chrono::steady_clock::now();
  for (int i = 0; i < eff_max_new; ++i) {
    const std::int32_t emit = sampling.forced.empty() ? next_tok : sampling.forced[static_cast<std::size_t>(i)];
    stats.generated_tokens = i + 1;
    const std::array<std::int32_t, 1> one = {emit};
    const bool cont = on_tokens ? on_tokens(one) : true;
    const bool stopped = std::find(stop_ids.begin(), stop_ids.end(), emit) != stop_ids.end();
    if (i + 1 == eff_max_new || stopped || !cont) break;
    impl_->decode_single_tok[0] = emit;
    const std::array<std::uint8_t, 1> decode_mask_arr = {1u};
    const std::span<const std::uint8_t> decode_mask = sess.committed_token_mask().empty()
                                                          ? std::span<const std::uint8_t>{}
                                                          : std::span<const std::uint8_t>(decode_mask_arr);
    const std::array<int, 1> cur_cap = {static_cast<int>(sess.committed_tokens().size())};
    impl_->forward_range(sess, impl_->decode_single_tok, cur_cap, false,
                         [&](const ingine::model::LogitStep& st) {
                           next_tok = select_token(st.logits);
                           if (on_step_logits) on_step_logits(st);
                         },
                         nullptr, decode_mask);
  }
  stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
  return stats;
}

ingine::model::GenerateStats Qwen38DenseBackend::generate_request(
    const ingine::model::BackendRequest& req, int max_new_tokens, const ingine::model::Sampling& sampling,
    std::span<const std::int32_t> stop_ids, const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
    const ingine::model::PrefillCallback& on_prefill) {
  return generate_request_on_session(*impl_->default_session, req, max_new_tokens, sampling, stop_ids, on_tokens,
                                     on_prefill);
}

ingine::model::GenerateStats Qwen38DenseBackend::generate(
    const ingine::model::Prompt& prompt, int max_new_tokens, const ingine::model::Sampling& sampling,
    std::span<const std::int32_t> stop_ids, const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
    const ingine::model::PrefillCallback& on_prefill) {
  ingine::model::BackendRequest req;
  req.prompt = prompt;
  req.cache_namespace = prompt.cache_namespace;
  req.use_prefix_cache = impl_->opts.prefix_cache && !prompt.bypass_prefix_cache;
  return generate_request_on_session(*impl_->default_session, req, max_new_tokens, sampling, stop_ids, on_tokens,
                                     on_prefill);
}

ingine::model::GenerateStats Qwen38DenseBackend::generate(
    std::span<const std::int32_t> prompt, int max_new_tokens, const ingine::model::Sampling& sampling,
    std::span<const std::int32_t> stop_ids, const std::function<bool(std::span<const std::int32_t>)>& on_tokens) {
  ingine::model::Prompt p;
  p.ids = prompt;
  return generate(p, max_new_tokens, sampling, stop_ids, on_tokens, nullptr);
}

ingine::model::GenerateStats Qwen38DenseBackend::warm_start(const ingine::model::Prompt& prompt,
                                                            const ingine::model::Sampling& sampling) {
  if (!impl_->opts.prefix_cache) {
    throw std::invalid_argument("warm_start requires prefix caching to be enabled");
  }
  if (prompt.ids.empty()) {
    throw std::invalid_argument("warm_start: prompt is empty");
  }
  if (prompt.bypass_prefix_cache) {
    throw std::invalid_argument("warm_start: prefix cache bypass requested");
  }
  if (!prompt.images.empty()) {
    throw std::invalid_argument("warm_start: images are not supported in startup warm-start prompts");
  }
  if (static_cast<int>(prompt.ids.size()) >= profile_.max_ctx) {
    throw std::invalid_argument("warm_start: prompt of " + std::to_string(prompt.ids.size()) +
                                " tokens leaves no generation room within max_ctx " +
                                std::to_string(profile_.max_ctx));
  }
  ingine::model::validate_request_against_capabilities(profile_.capabilities, profile_.max_ctx, prompt, 1, sampling);
  ingine::model::BackendStateIdentity req_id = profile_.state_identity;
  if (!prompt.cache_namespace.empty()) req_id.cache_namespace = prompt.cache_namespace;
  if (!req_id.is_reusable_with(req_id)) {
    throw std::invalid_argument("warm_start requires verified reusable checkpoint identity");
  }
  ingine::model::BackendRequest req;
  req.prompt = prompt;
  req.cache_namespace = prompt.cache_namespace;
  req.use_prefix_cache = true;
  double pf_ms = 0.0;
  evaluate_teacher_forced_on_session(*impl_->default_session, req, {}, nullptr, &pf_ms);
  const auto& sess = *impl_->default_session;
  const auto committed = sess.committed_tokens();
  const bool exact_retained = sess.identity().is_reusable_with(req_id) && committed.size() == prompt.ids.size() &&
                              ingine::model::prefix_matches_exact(committed, prompt.ids, prompt.ids.size()) &&
                              sess.restorable_prefix_len(prompt.ids.size()) == prompt.ids.size();
  if (!exact_retained) {
    throw std::runtime_error("warm_start: Qwen38 session could not retain exact reusable prefix and checkpoint");
  }
  ingine::model::GenerateStats stats;
  stats.prompt_tokens = static_cast<int>(prompt.ids.size());
  stats.reused_tokens = static_cast<int>(impl_->last_lookup.reused_tokens);
  stats.reuse = impl_->last_lookup.reuse;
  stats.prefill_seconds = pf_ms / 1000.0;
  stats.restored_bytes = 0ull;
  stats.restore_seconds = 0.0;
  return stats;
}

bool Qwen38DenseBackend::has_platos_key() const { return impl_->platos_key_gpu != nullptr; }

int Qwen38DenseBackend::platos_key_out_dim() const {
  return impl_->platos_key_gpu ? impl_->platos_key_gpu->out_dim() : 0;
}

std::string Qwen38DenseBackend::platos_key_tap_spec() const {
  return impl_->platos_key_gpu ? impl_->platos_key_gpu->metadata().tap_spec : "";
}

std::shared_ptr<const PlatosKey> Qwen38DenseBackend::platos_key() const {
  return impl_->platos_key_gpu ? impl_->platos_key_gpu->key() : nullptr;
}

ingine::model::EmbeddingResult Qwen38DenseBackend::embed(const ingine::model::EmbeddingRequest& req) {
  impl_->ctx.bind_to_current_thread();
  const auto& c = impl_->manifest.config;
  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  const int V = c.vocab_size;
  const int P = impl_->opts.prefill_chunk;
  const int num_layers = static_cast<int>(impl_->layers.size());

  if (req.inputs.empty()) throw std::invalid_argument("embed: inputs must not be empty");
  if (req.tap_layer.has_value() && (*req.tap_layer < 0 || *req.tap_layer >= num_layers)) {
    throw std::invalid_argument("embed: tap_layer " + std::to_string(*req.tap_layer) + " out of range [0, " +
                                std::to_string(num_layers - 1) + "]");
  }
  if (req.use_platos_key && !impl_->platos_key_gpu) {
    throw std::invalid_argument("embed: platos_key requested but server was started without --platos-key");
  }
  if (req.use_platos_key) {
    impl_->platos_key_gpu->key()->validate_request_tap(req.tap_layer);
    impl_->platos_key_gpu->key()->validate_request_identity(req.cartridge_id.value_or(""),
                                                           req.cartridge_sha256.value_or(""));
  }

  const int total_seqs = static_cast<int>(req.inputs.size());
  int total_tokens = 0;
  for (int i = 0; i < total_seqs; ++i) {
    const auto& seq = req.inputs[static_cast<std::size_t>(i)];
    if (seq.empty()) throw std::invalid_argument("embed: input sequence must not be empty");
    if (static_cast<int>(seq.size()) > impl_->opts.max_ctx) {
      throw std::invalid_argument("embed: input sequence length " + std::to_string(seq.size()) +
                                  " exceeds max_model_len " + std::to_string(impl_->opts.max_ctx));
    }
    for (std::int32_t tid : seq) {
      if (tid < 0 || tid >= V) {
        throw std::out_of_range("embed: token id " + std::to_string(tid) + " out of vocab range [0, " +
                                std::to_string(V) + ")");
      }
    }
    total_tokens += static_cast<int>(seq.size());
  }

  // Allocate tiny single-layer ephemeral scratch on first embed() call (reused across all 64 layers; ~4.2 MB total).
  const int cd = c.linear_conv_dim();
  const int nhv = c.linear_num_value_heads;
  const int nkh = c.linear_num_key_heads;
  const int k_dim = c.linear_key_head_dim;
  const int v_dim = c.linear_value_head_dim;
  const int vd = c.linear_value_dim();
  const int qg_dim = c.full_q_proj_out_dim();
  const int kv_dim = c.full_kv_proj_out_dim();
  const int attn_dim = c.num_attention_heads * c.head_dim;

  const std::size_t conv_elems = static_cast<std::size_t>(cd) * static_cast<std::size_t>(c.linear_conv_kernel_dim);
  const std::size_t rec_elems = static_cast<std::size_t>(nhv) * static_cast<std::size_t>(k_dim) *
                                static_cast<std::size_t>(v_dim);
  const std::size_t kv_tmp_elems = static_cast<std::size_t>(P) * static_cast<std::size_t>(kv_dim);
  if (impl_->embed_conv_tmp.count < conv_elems) impl_->embed_conv_tmp.allocate(conv_elems);
  if (impl_->embed_rec_tmp.count < rec_elems) impl_->embed_rec_tmp.allocate(rec_elems);
  if (impl_->embed_k_tmp.count < kv_tmp_elems) impl_->embed_k_tmp.allocate(kv_tmp_elems);
  if (impl_->embed_v_tmp.count < kv_tmp_elems) impl_->embed_v_tmp.allocate(kv_tmp_elems);

  const int out_dim = req.use_platos_key ? impl_->platos_key_gpu->out_dim() : H;
  if (impl_->embed_offsets_dev.count < static_cast<std::size_t>(total_seqs + 1)) {
    impl_->embed_offsets_dev.allocate(static_cast<std::size_t>(total_seqs + 1));
  }
  const std::size_t need_pool = static_cast<std::size_t>(total_seqs) * static_cast<std::size_t>(H);
  if (impl_->embed_pooled_dev.count < need_pool) impl_->embed_pooled_dev.allocate(need_pool);
  if (req.use_platos_key) {
    const std::size_t need_proj = static_cast<std::size_t>(total_seqs) * static_cast<std::size_t>(out_dim);
    if (impl_->embed_proj_dev.count < need_proj) impl_->embed_proj_dev.allocate(need_proj);
  }

  const int stop_layer = req.tap_layer.value_or(num_layers - 1);
  const bool apply_final_norm = !req.tap_layer.has_value();
  const bool pool_norm_now = req.normalize && !req.use_platos_key;

  const auto t_prefill_start = std::chrono::steady_clock::now();

  std::vector<std::int32_t> packed_toks;
  std::vector<int> packed_lens;
  std::vector<std::int32_t> packed_offsets;
  packed_toks.reserve(static_cast<std::size_t>(P));

  int idx = 0;
  while (idx < total_seqs) {
    const int seq_len = static_cast<int>(req.inputs[static_cast<std::size_t>(idx)].size());
    if (seq_len <= P) {
      packed_toks.clear();
      packed_lens.clear();
      packed_offsets.clear();
      packed_offsets.push_back(0);
      const int batch_start = idx;
      int packed_T = 0;
      while (idx < total_seqs) {
        const auto& seq = req.inputs[static_cast<std::size_t>(idx)];
        const int L = static_cast<int>(seq.size());
        if (L > P || (packed_T > 0 && packed_T + L > P)) break;
        packed_toks.insert(packed_toks.end(), seq.begin(), seq.end());
        packed_lens.push_back(L);
        packed_T += L;
        packed_offsets.push_back(packed_T);
        ++idx;
      }

      const int pack_b = static_cast<int>(packed_lens.size());
      CUDA_CHECK(cudaMemcpyAsync(impl_->token_ids_dev.ptr, packed_toks.data(),
                                 packed_toks.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                 impl_->ctx.stream));
      CUDA_CHECK(cudaMemcpyAsync(impl_->embed_offsets_dev.ptr, packed_offsets.data(),
                                 packed_offsets.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                 impl_->ctx.stream));

      if (impl_->embed_quant == Qwen38LinearQuantKind::Fp8E4M3) {
        qwen38_cuda_embed_lookup_fp8_f32(impl_->embed_u8, impl_->embed_scale, impl_->token_ids_dev.ptr, packed_T, H, V,
                                         impl_->hidden.ptr, impl_->ctx.stream);
      } else if (impl_->embed_quant == Qwen38LinearQuantKind::Nvfp4) {
        qwen38_cuda_embed_lookup_nvfp4_f32(impl_->embed_u8, impl_->embed_scales, impl_->embed_scale,
                                           impl_->token_ids_dev.ptr, packed_T, H, V, impl_->hidden.ptr,
                                           impl_->ctx.stream);
      } else {
        qwen38_cuda_embed_lookup_f32(impl_->embed_tokens, impl_->token_ids_dev.ptr, packed_T, H, V, impl_->hidden.ptr,
                                     impl_->ctx.stream);
      }

      for (int li = 0; li <= stop_layer; ++li) {
        const auto& l = impl_->layers[static_cast<std::size_t>(li)];
        qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, l.input_layernorm, c.rms_norm_eps, packed_T, H,
                                              impl_->normed.ptr, impl_->ctx.stream);
        if (l.kind == Qwen38LayerKind::LinearAttention) {
          // Packed compute-bound GEMMs across all packed_T tokens, then sequence-isolated zero-state conv+GDN.
          impl_->proj_forward_f32(l.in_proj_qkv, impl_->normed.ptr, packed_T, impl_->qkv.ptr);
          impl_->proj_forward_f32(l.in_proj_z, impl_->normed.ptr, packed_T, impl_->z_buf.ptr);
          impl_->proj_forward_f32(l.in_proj_a, impl_->normed.ptr, packed_T, impl_->a_buf.ptr);
          impl_->proj_forward_f32(l.in_proj_b, impl_->normed.ptr, packed_T, impl_->b_buf.ptr);

          int off = 0;
          for (int b = 0; b < pack_b; ++b) {
            const int L = packed_lens[static_cast<std::size_t>(b)];
            CUDA_CHECK(cudaMemsetAsync(impl_->embed_conv_tmp.ptr, 0, conv_elems * sizeof(float), impl_->ctx.stream));
            CUDA_CHECK(cudaMemsetAsync(impl_->embed_rec_tmp.ptr, 0, rec_elems * sizeof(float), impl_->ctx.stream));
            qwen38_cuda_causal_conv1d_silu_prefill_f32(
                impl_->qkv.ptr + static_cast<std::size_t>(off) * cd, l.conv1d_weight, L, cd, c.linear_conv_kernel_dim,
                impl_->embed_conv_tmp.ptr, impl_->conv_out.ptr + static_cast<std::size_t>(off) * cd, impl_->ctx.stream);
            qwen38_cuda_gdn_recurrent_prefill_f32(
                impl_->conv_out.ptr + static_cast<std::size_t>(off) * cd,
                impl_->z_buf.ptr + static_cast<std::size_t>(off) * vd,
                impl_->a_buf.ptr + static_cast<std::size_t>(off) * nhv,
                impl_->b_buf.ptr + static_cast<std::size_t>(off) * nhv, l.a_log, l.dt_bias, l.linear_norm_weight,
                c.rms_norm_eps, L, nkh, nhv, k_dim, v_dim, impl_->embed_rec_tmp.ptr, impl_->gdn_scratch.ptr,
                impl_->core_out.ptr + static_cast<std::size_t>(off) * vd, impl_->ctx.stream);
            off += L;
          }
          impl_->proj_forward_f32(l.out_proj, impl_->core_out.ptr, packed_T, impl_->residual.ptr);
        } else {
          // Packed QKV GEMMs across all packed_T tokens, then sequence-isolated causal attention into ephemeral KV tmp.
          impl_->proj_forward_f32(l.q_proj, impl_->normed.ptr, packed_T, impl_->q_gate.ptr);
          impl_->proj_forward_f32(l.k_proj, impl_->normed.ptr, packed_T, impl_->k_raw.ptr);
          impl_->proj_forward_f32(l.v_proj, impl_->normed.ptr, packed_T, impl_->v_raw.ptr);

          int off = 0;
          for (int b = 0; b < pack_b; ++b) {
            const int L = packed_lens[static_cast<std::size_t>(b)];
            qwen38_cuda_full_attn_gated_f32_kvbf16(
                impl_->q_gate.ptr + static_cast<std::size_t>(off) * qg_dim,
                impl_->k_raw.ptr + static_cast<std::size_t>(off) * kv_dim,
                impl_->v_raw.ptr + static_cast<std::size_t>(off) * kv_dim, l.q_norm, l.k_norm, c.rms_norm_eps,
                c.rope_theta, L, /*pos_offset=*/0, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                c.rotary_dim, impl_->embed_k_tmp.ptr, impl_->embed_v_tmp.ptr, P,
                impl_->q_tmp.ptr + static_cast<std::size_t>(off) * attn_dim,
                impl_->gate_tmp.ptr + static_cast<std::size_t>(off) * attn_dim,
                impl_->attn_out.ptr + static_cast<std::size_t>(off) * attn_dim, impl_->ctx.stream,
                impl_->attn_split_scratch.ptr, impl_->attn_split_scratch.count, impl_->opts.use_fast_attn);
            off += L;
          }
          impl_->proj_forward_f32(l.o_proj, impl_->attn_out.ptr, packed_T, impl_->residual.ptr);
        }

        qwen38_cuda_add_residual_f32(impl_->hidden.ptr, impl_->residual.ptr,
                                     static_cast<std::size_t>(packed_T) * static_cast<std::size_t>(H),
                                     impl_->ctx.stream);
        qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, l.post_attention_layernorm, c.rms_norm_eps, packed_T,
                                              H, impl_->normed.ptr, impl_->ctx.stream);
        impl_->proj_forward_f32(l.gate_proj, impl_->normed.ptr, packed_T, impl_->mlp_gate.ptr);
        impl_->proj_forward_f32(l.up_proj, impl_->normed.ptr, packed_T, impl_->mlp_up.ptr);
        qwen38_cuda_swiglu_mul_f32(impl_->mlp_gate.ptr, impl_->mlp_up.ptr, impl_->mlp_gate.ptr,
                                   static_cast<std::size_t>(packed_T) * static_cast<std::size_t>(I),
                                   impl_->ctx.stream);
        impl_->proj_forward_f32(l.down_proj, impl_->mlp_gate.ptr, packed_T, impl_->residual.ptr);
        qwen38_cuda_add_residual_f32(impl_->hidden.ptr, impl_->residual.ptr,
                                     static_cast<std::size_t>(packed_T) * static_cast<std::size_t>(H),
                                     impl_->ctx.stream);
      }

      const float* pool_src = impl_->hidden.ptr;
      if (apply_final_norm) {
        qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, impl_->final_norm, c.rms_norm_eps, packed_T, H,
                                              impl_->normed.ptr, impl_->ctx.stream);
        pool_src = impl_->normed.ptr;
      }
      // Completely skip lm_head GEMM! Pool directly into embed_pooled_dev.
      float* dst = impl_->embed_pooled_dev.ptr + static_cast<std::size_t>(batch_start) * static_cast<std::size_t>(H);
      ingine::embed_pool_f32_cuda(pool_src, impl_->embed_offsets_dev.ptr, pack_b, H, static_cast<int>(req.pooling),
                                  pool_norm_now, dst, impl_->ctx.stream);
    } else {
      // Transient chunked fallback for a single long input > prefill_chunk (never mutates default_session).
      const auto& seq = req.inputs[static_cast<std::size_t>(idx)];
      const int L = static_cast<int>(seq.size());
      Qwen38DenseSession tmp_sess(*this, /*max_checkpoints=*/0);
      std::vector<double> mean_acc(static_cast<std::size_t>(H), 0.0);
      std::vector<float> seq_out(static_cast<std::size_t>(H), 0.0f);
      std::vector<float> host_chunk;
      for (int off = 0; off < L; off += P) {
        const int M = std::min(P, L - off);
        CUDA_CHECK(cudaMemcpyAsync(impl_->token_ids_dev.ptr, seq.data() + off,
                                   static_cast<std::size_t>(M) * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   impl_->ctx.stream));
        if (impl_->embed_quant == Qwen38LinearQuantKind::Fp8E4M3) {
          qwen38_cuda_embed_lookup_fp8_f32(impl_->embed_u8, impl_->embed_scale, impl_->token_ids_dev.ptr, M, H, V,
                                           impl_->hidden.ptr, impl_->ctx.stream);
        } else if (impl_->embed_quant == Qwen38LinearQuantKind::Nvfp4) {
          qwen38_cuda_embed_lookup_nvfp4_f32(impl_->embed_u8, impl_->embed_scales, impl_->embed_scale,
                                             impl_->token_ids_dev.ptr, M, H, V, impl_->hidden.ptr, impl_->ctx.stream);
        } else {
          qwen38_cuda_embed_lookup_f32(impl_->embed_tokens, impl_->token_ids_dev.ptr, M, H, V, impl_->hidden.ptr,
                                       impl_->ctx.stream);
        }
        std::size_t lin_idx = 0;
        std::size_t full_idx = 0;
        for (int li = 0; li <= stop_layer; ++li) {
          const auto& l = impl_->layers[static_cast<std::size_t>(li)];
          qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, l.input_layernorm, c.rms_norm_eps, M, H,
                                                impl_->normed.ptr, impl_->ctx.stream);
          if (l.kind == Qwen38LayerKind::LinearAttention) {
            impl_->proj_forward_f32(l.in_proj_qkv, impl_->normed.ptr, M, impl_->qkv.ptr);
            impl_->proj_forward_f32(l.in_proj_z, impl_->normed.ptr, M, impl_->z_buf.ptr);
            impl_->proj_forward_f32(l.in_proj_a, impl_->normed.ptr, M, impl_->a_buf.ptr);
            impl_->proj_forward_f32(l.in_proj_b, impl_->normed.ptr, M, impl_->b_buf.ptr);
            qwen38_cuda_causal_conv1d_silu_prefill_f32(impl_->qkv.ptr, l.conv1d_weight, M, cd,
                                                       c.linear_conv_kernel_dim,
                                                       tmp_sess.state_->conv_states[lin_idx].ptr, impl_->conv_out.ptr,
                                                       impl_->ctx.stream);
            qwen38_cuda_gdn_recurrent_prefill_f32(impl_->conv_out.ptr, impl_->z_buf.ptr, impl_->a_buf.ptr,
                                                  impl_->b_buf.ptr, l.a_log, l.dt_bias, l.linear_norm_weight,
                                                  c.rms_norm_eps, M, nkh, nhv, k_dim, v_dim,
                                                  tmp_sess.state_->rec_states[lin_idx].ptr, impl_->gdn_scratch.ptr,
                                                  impl_->core_out.ptr, impl_->ctx.stream);
            impl_->proj_forward_f32(l.out_proj, impl_->core_out.ptr, M, impl_->residual.ptr);
            ++lin_idx;
          } else {
            impl_->proj_forward_f32(l.q_proj, impl_->normed.ptr, M, impl_->q_gate.ptr);
            impl_->proj_forward_f32(l.k_proj, impl_->normed.ptr, M, impl_->k_raw.ptr);
            impl_->proj_forward_f32(l.v_proj, impl_->normed.ptr, M, impl_->v_raw.ptr);
            qwen38_cuda_full_attn_gated_f32_kvbf16(
                impl_->q_gate.ptr, impl_->k_raw.ptr, impl_->v_raw.ptr, l.q_norm, l.k_norm, c.rms_norm_eps, c.rope_theta,
                M, off, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim,
                tmp_sess.state_->k_caches[full_idx].ptr, tmp_sess.state_->v_caches[full_idx].ptr, impl_->opts.max_ctx,
                impl_->q_tmp.ptr, impl_->gate_tmp.ptr, impl_->attn_out.ptr, impl_->ctx.stream,
                impl_->attn_split_scratch.ptr, impl_->attn_split_scratch.count, impl_->opts.use_fast_attn);
            impl_->proj_forward_f32(l.o_proj, impl_->attn_out.ptr, M, impl_->residual.ptr);
            ++full_idx;
          }
          qwen38_cuda_add_residual_f32(impl_->hidden.ptr, impl_->residual.ptr,
                                       static_cast<std::size_t>(M) * static_cast<std::size_t>(H), impl_->ctx.stream);
          qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, l.post_attention_layernorm, c.rms_norm_eps, M, H,
                                                impl_->normed.ptr, impl_->ctx.stream);
          impl_->proj_forward_f32(l.gate_proj, impl_->normed.ptr, M, impl_->mlp_gate.ptr);
          impl_->proj_forward_f32(l.up_proj, impl_->normed.ptr, M, impl_->mlp_up.ptr);
          qwen38_cuda_swiglu_mul_f32(impl_->mlp_gate.ptr, impl_->mlp_up.ptr, impl_->mlp_gate.ptr,
                                     static_cast<std::size_t>(M) * static_cast<std::size_t>(I), impl_->ctx.stream);
          impl_->proj_forward_f32(l.down_proj, impl_->mlp_gate.ptr, M, impl_->residual.ptr);
          qwen38_cuda_add_residual_f32(impl_->hidden.ptr, impl_->residual.ptr,
                                       static_cast<std::size_t>(M) * static_cast<std::size_t>(H), impl_->ctx.stream);
        }
        const bool need_chunk =
            (req.pooling == ingine::model::EmbeddingPooling::Mean) ||
            (req.pooling == ingine::model::EmbeddingPooling::Cls && off == 0) ||
            (req.pooling == ingine::model::EmbeddingPooling::LastToken && off + M == L);
        if (!need_chunk) continue;
        const float* src = impl_->hidden.ptr;
        if (apply_final_norm) {
          qwen38_cuda_rmsnorm_zero_centered_f32(impl_->hidden.ptr, impl_->final_norm, c.rms_norm_eps, M, H,
                                                impl_->normed.ptr, impl_->ctx.stream);
          src = impl_->normed.ptr;
        }
        host_chunk.resize(static_cast<std::size_t>(M) * static_cast<std::size_t>(H));
        CUDA_CHECK(cudaMemcpyAsync(host_chunk.data(), src, host_chunk.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   impl_->ctx.stream));
        impl_->ctx.synchronize();
        if (req.pooling == ingine::model::EmbeddingPooling::Cls) {
          std::memcpy(seq_out.data(), host_chunk.data(), static_cast<std::size_t>(H) * sizeof(float));
        } else if (req.pooling == ingine::model::EmbeddingPooling::LastToken) {
          std::memcpy(seq_out.data(), host_chunk.data() + static_cast<std::size_t>(M - 1) * static_cast<std::size_t>(H),
                      static_cast<std::size_t>(H) * sizeof(float));
        } else {
          for (int t = 0; t < M; ++t) {
            const std::size_t base = static_cast<std::size_t>(t) * static_cast<std::size_t>(H);
            for (int d = 0; d < H; ++d) mean_acc[static_cast<std::size_t>(d)] += host_chunk[base + d];
          }
        }
      }
      if (req.pooling == ingine::model::EmbeddingPooling::Mean) {
        const double inv = 1.0 / static_cast<double>(L);
        for (int d = 0; d < H; ++d) seq_out[static_cast<std::size_t>(d)] = static_cast<float>(mean_acc[d] * inv);
      }
      if (pool_norm_now) ingine::l2_normalize_rows_cpu(seq_out.data(), 1, H);
      float* dst = impl_->embed_pooled_dev.ptr + static_cast<std::size_t>(idx) * static_cast<std::size_t>(H);
      CUDA_CHECK(cudaMemcpyAsync(dst, seq_out.data(), static_cast<std::size_t>(H) * sizeof(float),
                                 cudaMemcpyHostToDevice, impl_->ctx.stream));
      impl_->ctx.synchronize();
      ++idx;
    }
  }
  impl_->ctx.synchronize();
  const double prefill_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();

  double platos_sec = 0.0;
  const float* final_dev = impl_->embed_pooled_dev.ptr;
  if (req.use_platos_key) {
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, impl_->ctx.stream));
    impl_->platos_key_gpu->project(impl_->embed_pooled_dev.ptr, total_seqs, impl_->embed_proj_dev.ptr, req.normalize,
                                   impl_->ctx.stream);
    CUDA_CHECK(cudaEventRecord(ev1, impl_->ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float pk_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pk_ms, ev0, ev1));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    platos_sec = static_cast<double>(pk_ms) * 1e-3;
    final_dev = impl_->embed_proj_dev.ptr;
  }

  std::vector<float> flat(static_cast<std::size_t>(total_seqs) * static_cast<std::size_t>(out_dim));
  CUDA_CHECK(cudaMemcpyAsync(flat.data(), final_dev, flat.size() * sizeof(float), cudaMemcpyDeviceToHost,
                             impl_->ctx.stream));
  impl_->ctx.synchronize();

  ingine::model::EmbeddingResult res;
  res.embeddings.resize(static_cast<std::size_t>(total_seqs));
  for (int b = 0; b < total_seqs; ++b) {
    const float* row = flat.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim);
    res.embeddings[static_cast<std::size_t>(b)].assign(row, row + out_dim);
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

void register_qwen38_cuda_backend() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto& reg = ingine::model::BackendRegistry::instance();
    if (reg.find(ingine::model::ModelFamily::Qwen3_5).has_value()) return;
    reg.register_backend(ingine::model::BackendRegistration{
        ingine::model::ModelFamily::Qwen3_5,
        "qwen38-dense",
        [](const ingine::model::RuntimeOptions& options) -> std::unique_ptr<ingine::model::ModelBackend> {
          const Qwen38RuntimeDispatchPlan plan = resolve_qwen38_runtime_options(options);
          Qwen38DenseLoadOptions load_opts;
          load_opts.model_dir = plan.normalized_options.model_dir;
          load_opts.max_ctx = plan.max_ctx;
          load_opts.prefill_chunk = plan.prefill_chunk;
          load_opts.verify_pins = plan.verify_pins;
          load_opts.use_calibrated_fp8 = plan.use_calibrated_fp8;
          load_opts.prefix_cache = plan.prefix_cache;
          load_opts.platos_key = plan.normalized_options.platos_key;
          load_opts.platos_key_metadata = plan.normalized_options.platos_key_metadata;
          if (const char* env_ps = std::getenv("INGINE_QWEN38_PRE_SUFFIX_CKPT"); env_ps && env_ps[0] == '0') {
            load_opts.pre_suffix_checkpoint = false;
          }
          if (const char* env_fa = std::getenv("INGINE_QWEN38_FAST_ATTN"); env_fa && env_fa[0] == '0') {
            load_opts.use_fast_attn = false;
          }
          load_opts.revision = plan.revision;
          return std::make_unique<Qwen38DenseBackend>(load_opts);
        },
    });
  });
}

}  // namespace ingine::backends::qwen38
