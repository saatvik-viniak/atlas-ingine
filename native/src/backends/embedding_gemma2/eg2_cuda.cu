#include "backends/embedding_gemma2/eg2_cuda.h"
#include "backends/embedding_gemma2/eg2_audio.h"
#include "backends/embedding_gemma2/eg2_vision.h"

#include "core/act.h"
#include "core/device.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine::backends::embedding_gemma2 {
namespace {

constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("Eg2Cuda cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  std::size_t count = 0;

  DevBuf() = default;
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

  void ensure(std::size_t n) {
    if (n <= count) return;
    free();
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
      count = n;
    }
  }
  void upload(const std::vector<T>& host, cudaStream_t stream) {
    ensure(host.size());
    if (!host.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
  }
  std::uint64_t bytes() const noexcept { return static_cast<std::uint64_t>(count) * sizeof(T); }
  void free() noexcept {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
  }
};

// ---- Architecture-independent BF16 <-> FP32 bit helpers (zero FP16 anywhere) ----

template <typename ElemT>
__device__ __forceinline__ float load_elem(const ElemT* p);

template <>
__device__ __forceinline__ float load_elem<float>(const float* p) {
  return *p;
}

template <>
__device__ __forceinline__ float load_elem<std::uint16_t>(const std::uint16_t* p) {
  return __uint_as_float(static_cast<std::uint32_t>(*p) << 16);
}

template <typename ElemT>
__device__ __forceinline__ void store_elem(ElemT* p, float v);

template <>
__device__ __forceinline__ void store_elem<float>(float* p, float v) {
  *p = v;
}

template <>
__device__ __forceinline__ void store_elem<std::uint16_t>(std::uint16_t* p, float v) {
  const std::uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    *p = static_cast<std::uint16_t>((u >> 16) | 0x40u);
    return;
  }
  *p = static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float gelu_tanh_dev(float x) {
  const float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float sum = (static_cast<int>(threadIdx.x) < nwarps) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum(sum);
  if (threadIdx.x == 0) smem[0] = sum;
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float m = (static_cast<int>(threadIdx.x) < nwarps) ? smem[threadIdx.x] : -1e30f;
  if (warp == 0) m = warp_reduce_max(m);
  if (threadIdx.x == 0) smem[0] = m;
  __syncthreads();
  return smem[0];
}

// ---- Kernels ----

template <typename ElemT>
__global__ void embed_lookup_scaled_kernel(const ElemT* __restrict__ table, const std::int32_t* __restrict__ token_ids,
                                           int T, int H, float scale, ElemT* __restrict__ out) {
  const int t = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= T || d >= H) return;
  const std::int32_t tid = token_ids[t];
  const float v = load_elem(&table[static_cast<std::size_t>(tid) * static_cast<std::size_t>(H) + d]) * scale;
  store_elem(&out[static_cast<std::size_t>(t) * static_cast<std::size_t>(H) + d], v);
}

// Direct multiplicative RMSNorm (NO +1.0 offset); weight == nullptr => unscaled (v_norm).
template <typename ElemT>
__global__ void rms_norm_kernel(const ElemT* __restrict__ x, const ElemT* __restrict__ weight, int rows, int dim,
                                float eps, ElemT* __restrict__ out) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const ElemT* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  ElemT* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);

  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&xr[d]);
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    float v = load_elem(&xr[d]) * inv_rms;
    if (weight != nullptr) v *= load_elem(&weight[d]);
    store_elem(&yr[d], v);
  }
}

// Fused post-RMSNorm + residual add: hidden[r, d] += rms_norm(sub[r, d], weight[d]).
template <typename ElemT>
__global__ void rms_norm_residual_add_kernel(const ElemT* __restrict__ sub, const ElemT* __restrict__ weight, int rows,
                                             int dim, float eps, ElemT* __restrict__ hidden) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const ElemT* sr = sub + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  ElemT* hr = hidden + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);

  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&sr[d]);
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float normed = load_elem(&sr[d]) * inv_rms * load_elem(&weight[d]);
    store_elem(&hr[d], load_elem(&hr[d]) + normed);
  }
}

// Fused post-RMSNorm + residual add + layer_scalar multiply (end of PLE sub-block).
template <typename ElemT>
__global__ void rms_norm_residual_scale_kernel(const ElemT* __restrict__ sub, const ElemT* __restrict__ weight,
                                               int rows, int dim, float eps, float layer_scalar,
                                               ElemT* __restrict__ hidden) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const ElemT* sr = sub + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  ElemT* hr = hidden + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);

  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&sr[d]);
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float normed = load_elem(&sr[d]) * inv_rms * load_elem(&weight[d]);
    store_elem(&hr[d], (load_elem(&hr[d]) + normed) * layer_scalar);
  }
}

// Fused per-head RMSNorm + rotate_half RoPE on [T, n_heads, head_dim] in-place.
template <typename ElemT>
__global__ void head_rmsnorm_rope_kernel(ElemT* __restrict__ x, const ElemT* __restrict__ weight,
                                         const std::int32_t* __restrict__ pos_ids, int T, int n_heads, int head_dim,
                                         float eps, double rope_theta) {
  const int row = static_cast<int>(blockIdx.x);
  const int total_rows = T * n_heads;
  if (row >= total_rows) return;
  const int t = row / n_heads;
  const int pos = pos_ids[t];
  const int half = head_dim >> 1;
  __shared__ float smem[32];

  ElemT* xr = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(head_dim);
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&xr[d]);
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(head_dim) + eps);

  for (int j = static_cast<int>(threadIdx.x); j < half; j += static_cast<int>(blockDim.x)) {
    const float x1 = load_elem(&xr[j]) * inv_rms * load_elem(&weight[j]);
    const float x2 = load_elem(&xr[j + half]) * inv_rms * load_elem(&weight[j + half]);
    const double inv_freq = 1.0 / pow(rope_theta, (2.0 * static_cast<double>(j)) / static_cast<double>(head_dim));
    const double angle = static_cast<double>(pos) * inv_freq;
    const float c = static_cast<float>(cos(angle));
    const float s = static_cast<float>(sin(angle));
    store_elem(&xr[j], x1 * c - x2 * s);
    store_elem(&xr[j + half], x2 * c + x1 * s);
  }
}

// Bidirectional online-softmax Flash-style attention (scaling = 1.0, zero T^2 VRAM).
// Supports both GQA sliding_attention (|q - k| <= sliding_window) and MQA full_attention,
// packed ragged batches via seq_of_tok[T] and offsets[B+1].
// Launched with blockDim.x = 128, grid = (T, n_heads). Supports head_dim up to 1024 (8 elems/thread).
template <typename ElemT>
__global__ void bidir_attention_kernel(const ElemT* __restrict__ q, const ElemT* __restrict__ k,
                                       const ElemT* __restrict__ v, const std::int32_t* __restrict__ offsets,
                                       const std::int32_t* __restrict__ seq_of_tok, int T, int n_heads, int n_kv,
                                       int head_dim, int sliding_window, bool is_sliding, ElemT* __restrict__ out) {
  const int qi = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  if (qi >= T || h >= n_heads) return;

  const int kv_groups = n_heads / n_kv;
  const int kvh = h / kv_groups;
  const int seq = seq_of_tok[qi];
  const int seq_start = offsets[seq];
  const int seq_end = offsets[seq + 1];
  const int k_begin = is_sliding ? max(seq_start, qi - sliding_window) : seq_start;
  const int k_end = is_sliding ? min(seq_end - 1, qi + sliding_window) : (seq_end - 1);

  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kMaxRegs = 8;  // supports head_dim up to 128 * 8 = 1024 (EG2 uses 256 and 512)
  float q_reg[kMaxRegs];
  float acc[kMaxRegs];

  const ElemT* qp = q + (static_cast<std::size_t>(qi) * static_cast<std::size_t>(n_heads) + h) * head_dim;
  int n_regs = 0;
  for (int d = tid; d < head_dim; d += 128) {
    q_reg[n_regs] = load_elem(&qp[d]);
    acc[n_regs] = 0.0f;
    ++n_regs;
  }

  __shared__ float smem[32];
  float m_val = -3.402823466e+38f;
  float l_val = 0.0f;

  for (int ki = k_begin; ki <= k_end; ++ki) {
    const ElemT* kp = k + (static_cast<std::size_t>(ki) * static_cast<std::size_t>(n_kv) + kvh) * head_dim;
    float partial_dot = 0.0f;
    for (int r = 0; r < n_regs; ++r) {
      const int d = tid + r * 128;
      partial_dot += q_reg[r] * load_elem(&kp[d]);
    }
    const float score = block_reduce_sum(partial_dot, smem);
    const float m_new = fmaxf(m_val, score);
    const float alpha = expf(m_val - m_new);
    const float p = expf(score - m_new);
    l_val = l_val * alpha + p;
    m_val = m_new;

    const ElemT* vp = v + (static_cast<std::size_t>(ki) * static_cast<std::size_t>(n_kv) + kvh) * head_dim;
    for (int r = 0; r < n_regs; ++r) {
      const int d = tid + r * 128;
      acc[r] = acc[r] * alpha + p * load_elem(&vp[d]);
    }
  }

  const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
  ElemT* op = out + (static_cast<std::size_t>(qi) * static_cast<std::size_t>(n_heads) + h) * head_dim;
  for (int r = 0; r < n_regs; ++r) {
    const int d = tid + r * 128;
    store_elem(&op[d], acc[r] * inv_l);
  }
}

template <typename ElemT>
__global__ void geglu_mul_kernel(ElemT* __restrict__ gate, const ElemT* __restrict__ up, std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float g = load_elem(&gate[idx]);
  const float u = load_elem(&up[idx]);
  store_elem(&gate[idx], gelu_tanh_dev(g) * u);
}

// gate[t, d] = gelu_tanh(gate[t, d]) * ple[t, layer_idx, d]
template <typename ElemT>
__global__ void ple_gate_mul_kernel(ElemT* __restrict__ gate, const ElemT* __restrict__ ple, int T, int num_layers,
                                    int layer_idx, int H) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(T) * static_cast<std::size_t>(H);
  if (idx >= total) return;
  const int t = static_cast<int>(idx / static_cast<std::size_t>(H));
  const int d = static_cast<int>(idx % static_cast<std::size_t>(H));
  const std::size_t ple_off =
      (static_cast<std::size_t>(t) * static_cast<std::size_t>(num_layers) + static_cast<std::size_t>(layer_idx)) *
          static_cast<std::size_t>(H) +
      static_cast<std::size_t>(d);
  const float g = load_elem(&gate[idx]);
  const float p = load_elem(&ple[ple_off]);
  store_elem(&gate[idx], gelu_tanh_dev(g) * p);
}

// Pool ragged sequences [T, H] -> [B, H] FP32 (0 = Mean, 1 = LastToken, 2 = Cls).
template <typename ElemT>
__global__ void pool_to_f32_kernel(const ElemT* __restrict__ hidden, const std::int32_t* __restrict__ offsets, int B,
                                   int H, int pooling_mode, float* __restrict__ out) {
  const int b = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (b >= B || d >= H) return;
  const int start = offsets[b];
  const int end = offsets[b + 1];
  const int len = end - start;
  float val = 0.0f;
  if (len > 0) {
    if (pooling_mode == 1) {
      val = load_elem(&hidden[static_cast<std::size_t>(end - 1) * static_cast<std::size_t>(H) + d]);
    } else if (pooling_mode == 2) {
      val = load_elem(&hidden[static_cast<std::size_t>(start) * static_cast<std::size_t>(H) + d]);
    } else {
      double acc = 0.0;
      for (int t = start; t < end; ++t) {
        acc += static_cast<double>(load_elem(&hidden[static_cast<std::size_t>(t) * static_cast<std::size_t>(H) + d]));
      }
      val = static_cast<float>(acc / static_cast<double>(len));
    }
  }
  out[static_cast<std::size_t>(b) * static_cast<std::size_t>(H) + d] = val;
}

// Project pooled FP32 [B, H] @ W[E, H]^T -> FP32 [B, E] (exact FP32 matvec per output element).
template <typename ElemT>
__global__ void project_pooled_to_f32_kernel(const float* __restrict__ pooled, const ElemT* __restrict__ weight, int B,
                                             int H, int E, float* __restrict__ out) {
  const int b = static_cast<int>(blockIdx.y);
  const int e = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (b >= B || e >= E) return;
  const float* xr = pooled + static_cast<std::size_t>(b) * static_cast<std::size_t>(H);
  const ElemT* wr = weight + static_cast<std::size_t>(e) * static_cast<std::size_t>(H);
  float acc = 0.0f;
  for (int k = 0; k < H; ++k) {
    acc += xr[k] * load_elem(&wr[k]);
  }
  out[static_cast<std::size_t>(b) * static_cast<std::size_t>(E) + e] = acc;
}

__global__ void truncate_and_l2_norm_f32_kernel(const float* __restrict__ src, int B, int src_dim, int dst_dim,
                                                bool normalize, float* __restrict__ dst) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= B) return;
  __shared__ float smem[32];
  const float* sr = src + static_cast<std::size_t>(b) * static_cast<std::size_t>(src_dim);
  float* dr = dst + static_cast<std::size_t>(b) * static_cast<std::size_t>(dst_dim);

  float inv_norm = 1.0f;
  if (normalize) {
    float ss = 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < dst_dim; d += static_cast<int>(blockDim.x)) {
      const float v = sr[d];
      ss += v * v;
    }
    const float total_ss = block_reduce_sum(ss, smem);
    inv_norm = (total_ss > 1e-24f) ? rsqrtf(total_ss) : 1.0f;
  }
  for (int d = static_cast<int>(threadIdx.x); d < dst_dim; d += static_cast<int>(blockDim.x)) {
    dr[d] = sr[d] * inv_norm;
  }
}

// ---- Phase 2 Vision Tower + Multimodal Scatter Kernels ----

template <typename ElemT>
__global__ void scale_patches_to_elem_kernel(const float* __restrict__ pv_01, ElemT* __restrict__ out, std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  store_elem(&out[idx], 2.0f * (pv_01[idx] - 0.5f));
}

template <typename ElemT>
__global__ void add_2d_pos_embed_kernel(ElemT* __restrict__ hidden, const ElemT* __restrict__ pos_table,
                                        const std::int32_t* __restrict__ pos_x, const std::int32_t* __restrict__ pos_y,
                                        int N, int pos_emb_size, int VH) {
  const int n = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (n >= N || d >= VH) return;
  const int px = pos_x[n];
  const int py = pos_y[n];
  const std::size_t stride = static_cast<std::size_t>(pos_emb_size) * static_cast<std::size_t>(VH);
  const float ex = load_elem(&pos_table[static_cast<std::size_t>(px) * static_cast<std::size_t>(VH) + d]);
  const float ey = load_elem(&pos_table[stride + static_cast<std::size_t>(py) * static_cast<std::size_t>(VH) + d]);
  ElemT* dst = &hidden[static_cast<std::size_t>(n) * static_cast<std::size_t>(VH) + d];
  store_elem(dst, load_elem(dst) + ex + ey);
}

// Fused per-head RMSNorm + 2D axial RoPE (first head_dim/2 dims by pos_x, second head_dim/2 by pos_y).
template <typename ElemT>
__global__ void head_rmsnorm_axial_2d_rope_kernel(ElemT* __restrict__ x, const ElemT* __restrict__ weight,
                                                  const std::int32_t* __restrict__ pos_x,
                                                  const std::int32_t* __restrict__ pos_y, int N, int n_heads,
                                                  int head_dim, float eps, double rope_theta) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= N * n_heads) return;
  const int n = row / n_heads;
  const double px = static_cast<double>(pos_x[n]);
  const double py = static_cast<double>(pos_y[n]);
  const int spatial_dim = head_dim >> 1;
  const int quarter = spatial_dim >> 1;
  __shared__ float smem[32];

  ElemT* xr = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(head_dim);
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&xr[d]);
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(head_dim) + eps);

  for (int j = static_cast<int>(threadIdx.x); j < quarter; j += static_cast<int>(blockDim.x)) {
    const double inv_freq = 1.0 / pow(rope_theta, (2.0 * static_cast<double>(j)) / static_cast<double>(spatial_dim));
    // X-axis half [0 .. spatial_dim - 1]
    {
      const float x1 = load_elem(&xr[j]) * inv_rms * load_elem(&weight[j]);
      const float x2 = load_elem(&xr[j + quarter]) * inv_rms * load_elem(&weight[j + quarter]);
      const double ax = px * inv_freq;
      const float c = static_cast<float>(cos(ax));
      const float s = static_cast<float>(sin(ax));
      store_elem(&xr[j], x1 * c - x2 * s);
      store_elem(&xr[j + quarter], x2 * c + x1 * s);
    }
    // Y-axis half [spatial_dim .. head_dim - 1]
    {
      const int b = spatial_dim;
      const float y1 = load_elem(&xr[b + j]) * inv_rms * load_elem(&weight[b + j]);
      const float y2 = load_elem(&xr[b + j + quarter]) * inv_rms * load_elem(&weight[b + j + quarter]);
      const double ay = py * inv_freq;
      const float c = static_cast<float>(cos(ay));
      const float s = static_cast<float>(sin(ay));
      store_elem(&xr[b + j], y1 * c - y2 * s);
      store_elem(&xr[b + j + quarter], y2 * c + y1 * s);
    }
  }
}

// 3x3 spatial average pool -> * sqrt(VH) in FP32 -> unscaled RMSNorm (eps=1e-6).
// Launched with grid = oh * ow, blockDim.x = 256.
template <typename ElemT>
__global__ void vision_spatial_pool_norm_kernel(const ElemT* __restrict__ hidden, int ph, int pw, int k, int VH,
                                                float eps, ElemT* __restrict__ out) {
  const int oh = ph / k;
  const int ow = pw / k;
  const int o_idx = static_cast<int>(blockIdx.x);
  if (o_idx >= oh * ow) return;
  const int oy = o_idx / ow;
  const int ox = o_idx % ow;
  const float scale = sqrtf(static_cast<float>(VH)) / static_cast<float>(k * k);
  __shared__ float smem[32];

  // First pass: accumulate k*k patches into shared/register and compute sum of squares
  // (VH=768, blockDim=256 => 3 elements per thread).
  constexpr int kMaxPerThread = 8;
  float vals[kMaxPerThread];
  int n_vals = 0;
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < VH; d += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
    for (int dy = 0; dy < k; ++dy) {
      for (int dx = 0; dx < k; ++dx) {
        const int p_idx = (oy * k + dy) * pw + (ox * k + dx);
        acc += load_elem(&hidden[static_cast<std::size_t>(p_idx) * static_cast<std::size_t>(VH) + d]);
      }
    }
    const float v = acc * scale;
    vals[n_vals++] = v;
    ss += v * v;
  }
  const float total_ss = block_reduce_sum(ss, smem);
  const float inv_rms = rsqrtf(total_ss / static_cast<float>(VH) + eps);
  ElemT* dst = out + static_cast<std::size_t>(o_idx) * static_cast<std::size_t>(VH);
  int vi = 0;
  for (int d = static_cast<int>(threadIdx.x); d < VH; d += static_cast<int>(blockDim.x)) {
    store_elem(&dst[d], vals[vi++] * inv_rms);
  }
}

template <typename ElemT>
__global__ void scatter_rows_kernel(ElemT* __restrict__ dst_hidden, const ElemT* __restrict__ src_soft,
                                    const std::int32_t* __restrict__ dst_indices, int n_soft, int H) {
  const int s = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (s >= n_soft || d >= H) return;
  const int dst_t = dst_indices[s];
  dst_hidden[static_cast<std::size_t>(dst_t) * static_cast<std::size_t>(H) + d] =
      src_soft[static_cast<std::size_t>(s) * static_cast<std::size_t>(H) + d];
}

// Vision full-attention layout helpers for strided-batched cuBLASLt GEMM.
template <typename ElemT>
__global__ void transpose_nhd_to_hnd_kernel(const ElemT* __restrict__ src, ElemT* __restrict__ dst, int N, int H,
                                            int D) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(N) * static_cast<std::size_t>(H) * static_cast<std::size_t>(D);
  if (idx >= total) return;
  const int d = static_cast<int>(idx % static_cast<std::size_t>(D));
  const std::size_t nh = idx / static_cast<std::size_t>(D);
  const int h = static_cast<int>(nh % static_cast<std::size_t>(H));
  const int n = static_cast<int>(nh / static_cast<std::size_t>(H));
  dst[(static_cast<std::size_t>(h) * static_cast<std::size_t>(N) + static_cast<std::size_t>(n)) *
          static_cast<std::size_t>(D) +
      static_cast<std::size_t>(d)] = src[idx];
}

template <typename ElemT>
__global__ void transpose_nhd_to_hdn_kernel(const ElemT* __restrict__ src, ElemT* __restrict__ dst, int N, int H,
                                            int D) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(N) * static_cast<std::size_t>(H) * static_cast<std::size_t>(D);
  if (idx >= total) return;
  const int d = static_cast<int>(idx % static_cast<std::size_t>(D));
  const std::size_t nh = idx / static_cast<std::size_t>(D);
  const int h = static_cast<int>(nh % static_cast<std::size_t>(H));
  const int n = static_cast<int>(nh / static_cast<std::size_t>(H));
  dst[(static_cast<std::size_t>(h) * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)) *
          static_cast<std::size_t>(N) +
      static_cast<std::size_t>(n)] = src[idx];
}

template <typename ElemT>
__global__ void transpose_hnd_to_nhd_kernel(const ElemT* __restrict__ src, ElemT* __restrict__ dst, int N, int H,
                                            int D) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(N) * static_cast<std::size_t>(H) * static_cast<std::size_t>(D);
  if (idx >= total) return;
  const int d = static_cast<int>(idx % static_cast<std::size_t>(D));
  const std::size_t nh = idx / static_cast<std::size_t>(D);
  const int h = static_cast<int>(nh % static_cast<std::size_t>(H));
  const int n = static_cast<int>(nh / static_cast<std::size_t>(H));
  dst[idx] = src[(static_cast<std::size_t>(h) * static_cast<std::size_t>(N) + static_cast<std::size_t>(n)) *
                     static_cast<std::size_t>(D) +
                 static_cast<std::size_t>(d)];
}

// Exact FP32 row softmax over [rows, cols] -> ElemT probabilities.
template <typename ElemT>
__global__ void softmax_rows_f32_to_elem_kernel(const float* __restrict__ scores, ElemT* __restrict__ probs, int rows,
                                                int cols) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const float* sr = scores + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
  ElemT* dr = probs + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);

  float local_max = -1e30f;
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    local_max = fmaxf(local_max, sr[c]);
  }
  const float row_max = block_reduce_max(local_max, smem);

  float local_sum = 0.0f;
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    local_sum += expf(sr[c] - row_max);
  }
  const float inv_sum = 1.0f / block_reduce_sum(local_sum, smem);

  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    store_elem(&dr[c], expf(sr[c] - row_max) * inv_sum);
  }
}

// ---- Phase 3 Audio Conformer Kernels (zero FP16; strictly BF16 or FP32) ----

template <typename ElemT>
__global__ void audio_copy_f32_to_elem_kernel(const float* __restrict__ src, ElemT* __restrict__ dst, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  store_elem(&dst[i], src[i]);
}

template <typename ElemT>
__global__ void audio_clamp_copy_kernel(const ElemT* __restrict__ src, ElemT* __restrict__ dst, std::size_t n,
                                        float lo, float hi) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = fminf(hi, fmaxf(lo, load_elem(&src[i])));
  store_elem(&dst[i], v);
}

// Fused SSCP Conv2d(k=3, stride=2, pad=1, bias=False) + mean-centered LayerNorm(C_out, no bias) + ReLU.
// Input: in_chw [C_in, H_in, W_in] with optional in_time_mask [H_in] (0 => zero frame before conv).
// Output: out_chw [C_out, H_out, W_out]. Launched with grid = H_out * W_out, block = 128 (>= C_out).
template <typename ElemT>
__global__ void audio_sscp_conv_ln_relu_kernel(const ElemT* __restrict__ in_chw,
                                               const std::uint8_t* __restrict__ in_time_mask,
                                               const ElemT* __restrict__ conv_w, const ElemT* __restrict__ norm_w,
                                               int C_in, int H_in, int W_in, int C_out, float eps,
                                               ElemT* __restrict__ out_chw) {
  const int H_out = (H_in + 2 - 3) / 2 + 1;
  const int W_out = (W_in + 2 - 3) / 2 + 1;
  const int pos = static_cast<int>(blockIdx.x);
  if (pos >= H_out * W_out) return;
  const int oh = pos / W_out;
  const int ow = pos % W_out;
  const int oc = static_cast<int>(threadIdx.x);
  __shared__ float smem[32];

  float val = 0.0f;
  if (oc < C_out) {
    const ElemT* w_oc = conv_w + static_cast<std::size_t>(oc) * static_cast<std::size_t>(C_in) * 9u;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
      const ElemT* in_ic =
          in_chw + static_cast<std::size_t>(ic) * static_cast<std::size_t>(H_in) * static_cast<std::size_t>(W_in);
      const ElemT* w_ic = w_oc + static_cast<std::size_t>(ic) * 9u;
      for (int kh = 0; kh < 3; ++kh) {
        const int ih = oh * 2 + kh - 1;
        if (ih < 0 || ih >= H_in) continue;
        if (in_time_mask != nullptr && in_time_mask[ih] == 0) continue;
        const ElemT* in_row = in_ic + static_cast<std::size_t>(ih) * static_cast<std::size_t>(W_in);
        for (int kw = 0; kw < 3; ++kw) {
          const int iw = ow * 2 + kw - 1;
          if (iw < 0 || iw >= W_in) continue;
          acc += load_elem(&in_row[iw]) * load_elem(&w_ic[kh * 3 + kw]);
        }
      }
    }
    val = acc;
  }

  const float sum_v = block_reduce_sum(oc < C_out ? val : 0.0f, smem);
  const float mean = sum_v / static_cast<float>(C_out);
  const float diff = (oc < C_out) ? (val - mean) : 0.0f;
  const float sum_sq = block_reduce_sum(diff * diff, smem);
  const float inv = rsqrtf(sum_sq / static_cast<float>(C_out) + eps);
  if (oc < C_out) {
    const float normed = diff * inv * load_elem(&norm_w[oc]);
    const float relu = normed > 0.0f ? normed : 0.0f;
    store_elem(&out_chw[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(H_out) +
                         static_cast<std::size_t>(oh)) *
                            static_cast<std::size_t>(W_out) +
                        static_cast<std::size_t>(ow)],
               relu);
  }
}

template <typename ElemT>
__global__ void audio_sscp_permute_tfc_kernel(const ElemT* __restrict__ in_chw, int C1, int T_seq, int F2,
                                              ElemT* __restrict__ out_tfc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(F2) * static_cast<std::size_t>(C1);
  if (idx >= total) return;
  const int c = static_cast<int>(idx % static_cast<std::size_t>(C1));
  const std::size_t tf = idx / static_cast<std::size_t>(C1);
  const int f = static_cast<int>(tf % static_cast<std::size_t>(F2));
  const int t = static_cast<int>(tf / static_cast<std::size_t>(F2));
  out_tfc[idx] = in_chw[(static_cast<std::size_t>(c) * static_cast<std::size_t>(T_seq) + static_cast<std::size_t>(t)) *
                            static_cast<std::size_t>(F2) +
                        static_cast<std::size_t>(f)];
}

template <typename ElemT>
__global__ void audio_clamp_rmsnorm_kernel(const ElemT* __restrict__ x, const ElemT* __restrict__ weight, int rows,
                                           int dim, float eps, float grad_clip, ElemT* __restrict__ out) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const ElemT* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  ElemT* yr = out + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = fminf(grad_clip, fmaxf(-grad_clip, load_elem(&xr[d])));
    ss += v * v;
  }
  const float inv = rsqrtf(block_reduce_sum(ss, smem) / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = fminf(grad_clip, fmaxf(-grad_clip, load_elem(&xr[d])));
    const float w = (weight != nullptr) ? load_elem(&weight[d]) : 1.0f;
    store_elem(&yr[d], v * inv * w);
  }
}

template <typename ElemT>
__global__ void audio_clamp_rmsnorm_residual_scale_kernel(const ElemT* __restrict__ sub,
                                                          const ElemT* __restrict__ weight, int rows, int dim,
                                                          float eps, float grad_clip, float res_scale,
                                                          ElemT* __restrict__ hidden) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  const ElemT* sr = sub + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  ElemT* hr = hidden + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = fminf(grad_clip, fmaxf(-grad_clip, load_elem(&sr[d])));
    ss += v * v;
  }
  const float inv = rsqrtf(block_reduce_sum(ss, smem) / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = fminf(grad_clip, fmaxf(-grad_clip, load_elem(&sr[d])));
    const float norm_v = v * inv * load_elem(&weight[d]);
    store_elem(&hr[d], load_elem(&hr[d]) + res_scale * norm_v);
  }
}

template <typename ElemT>
__global__ void audio_silu_inplace_kernel(ElemT* __restrict__ buf, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = load_elem(&buf[i]);
  store_elem(&buf[i], x / (1.0f + expf(-x)));
}

template <typename ElemT>
__global__ void audio_glu_kernel(const ElemT* __restrict__ in_2h, ElemT* __restrict__ out_h, int T_seq, int AH) {
  const int t = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= T_seq || d >= AH) return;
  const ElemT* sr = in_2h + static_cast<std::size_t>(t) * static_cast<std::size_t>(2 * AH);
  const float a = load_elem(&sr[d]);
  const float b = load_elem(&sr[AH + d]);
  store_elem(&out_h[static_cast<std::size_t>(t) * static_cast<std::size_t>(AH) + d], a / (1.0f + expf(-b)));
}

template <typename ElemT>
__global__ void audio_causal_dw_conv1d_kernel(const ElemT* __restrict__ in_th, const ElemT* __restrict__ dw_w,
                                              int T_seq, int AH, int K, float grad_clip, ElemT* __restrict__ out_th) {
  const int t = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (t >= T_seq || d >= AH) return;
  const int left_pad = K - 1;
  const ElemT* w = dw_w + static_cast<std::size_t>(d) * static_cast<std::size_t>(K);
  float acc = 0.0f;
  for (int ki = 0; ki < K; ++ki) {
    const int ti = t - left_pad + ki;
    if (ti >= 0) {
      acc += load_elem(&in_th[static_cast<std::size_t>(ti) * static_cast<std::size_t>(AH) + d]) * load_elem(&w[ki]);
    }
  }
  acc = fminf(grad_clip, fmaxf(-grad_clip, acc));
  store_elem(&out_th[static_cast<std::size_t>(t) * static_cast<std::size_t>(AH) + d], acc);
}

template <typename ElemT>
__global__ void audio_residual_add_kernel(ElemT* __restrict__ hidden, const ElemT* __restrict__ sub, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  store_elem(&hidden[i], load_elem(&hidden[i]) + load_elem(&sub[i]));
}

// Causal sliding local relative-position attention (left=13, right=0, softcap=50.0, per_dim_scale).
// Launched with grid = (T_seq, n_heads), blockDim.x = 128 (matches hd = 128).
template <typename ElemT>
__global__ void audio_local_rel_attn_kernel(const ElemT* __restrict__ q, const ElemT* __restrict__ k,
                                            const ElemT* __restrict__ v, const ElemT* __restrict__ rel_k,
                                            const float* __restrict__ per_dim_scale,
                                            const std::uint8_t* __restrict__ valid_mask, int T_seq, int n_heads, int hd,
                                            int max_past, float q_scale, float k_scale, float softcap,
                                            ElemT* __restrict__ out) {
  const int t = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  if (t >= T_seq || h >= n_heads) return;
  const int AH = n_heads * hd;
  ElemT* op = out + static_cast<std::size_t>(t) * static_cast<std::size_t>(AH) + static_cast<std::size_t>(h) * hd;

  if (valid_mask[t] == 0) {
    for (int d = static_cast<int>(threadIdx.x); d < hd; d += static_cast<int>(blockDim.x)) store_elem(&op[d], 0.0f);
    return;
  }

  __shared__ float smem[32];
  __shared__ float s_weights[32];
  __shared__ int s_kidx[32];

  const ElemT* qp = q + static_cast<std::size_t>(t) * static_cast<std::size_t>(AH) + static_cast<std::size_t>(h) * hd;
  // Cache scaled query in registers (hd=128, blockDim=128 => 1 elem/thread; loop supports hd up to 256).
  float q_reg[2] = {0.0f, 0.0f};
  int q_cnt = 0;
  for (int d = static_cast<int>(threadIdx.x); d < hd; d += static_cast<int>(blockDim.x)) {
    const float p = per_dim_scale[d];
    const float sp = (p > 20.0f) ? p : ((p < -20.0f) ? expf(p) : log1pf(expf(p)));
    q_reg[q_cnt++] = load_elem(&qp[d]) * q_scale * sp;
  }

  const int k_lo = (t > max_past) ? (t - max_past) : 0;
  int n_k = 0;
  float max_s = -1e30f;
  for (int kk = k_lo; kk <= t; ++kk) {
    if (valid_mask[kk] == 0) continue;
    const int rel_idx = max_past - (t - kk);
    const ElemT* kp =
        k + static_cast<std::size_t>(kk) * static_cast<std::size_t>(AH) + static_cast<std::size_t>(h) * hd;
    const ElemT* rp =
        rel_k + static_cast<std::size_t>(rel_idx) * static_cast<std::size_t>(AH) + static_cast<std::size_t>(h) * hd;
    float local_dot = 0.0f;
    int qi = 0;
    for (int d = static_cast<int>(threadIdx.x); d < hd; d += static_cast<int>(blockDim.x)) {
      const float k_val = load_elem(&kp[d]) * k_scale + load_elem(&rp[d]);
      local_dot += q_reg[qi++] * k_val;
    }
    const float dot = block_reduce_sum(local_dot, smem);
    const float s = tanhf(dot / softcap) * softcap;
    if (threadIdx.x == 0) {
      s_weights[n_k] = s;
      s_kidx[n_k] = kk;
    }
    if (s > max_s) max_s = s;
    ++n_k;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum_e = 0.0f;
    for (int j = 0; j < n_k; ++j) {
      const float e = expf(s_weights[j] - max_s);
      s_weights[j] = e;
      sum_e += e;
    }
    const float inv = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
    for (int j = 0; j < n_k; ++j) s_weights[j] *= inv;
  }
  __syncthreads();

  for (int d = static_cast<int>(threadIdx.x); d < hd; d += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
    for (int j = 0; j < n_k; ++j) {
      const int kk = s_kidx[j];
      const ElemT* vp =
          v + static_cast<std::size_t>(kk) * static_cast<std::size_t>(AH) + static_cast<std::size_t>(h) * hd;
      acc += s_weights[j] * load_elem(&vp[d]);
    }
    store_elem(&op[d], acc);
  }
}

template <typename ElemT>
__global__ void audio_bias_unscaled_rmsnorm_kernel(ElemT* __restrict__ x, const ElemT* __restrict__ bias, int rows,
                                                   int dim, float eps) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  __shared__ float smem[32];
  ElemT* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(dim);
  float ss = 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    const float v = load_elem(&xr[d]) + load_elem(&bias[d]);
    store_elem(&xr[d], v);
    ss += v * v;
  }
  const float inv = rsqrtf(block_reduce_sum(ss, smem) / static_cast<float>(dim) + eps);
  for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
    store_elem(&xr[d], load_elem(&xr[d]) * inv);
  }
}

template <typename ElemT>
__global__ void audio_compact_rows_kernel(const ElemT* __restrict__ src, const std::int32_t* __restrict__ src_indices,
                                          int n_valid, int dim, ElemT* __restrict__ dst) {
  const int r = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (r >= n_valid || d >= dim) return;
  const int s = src_indices[r];
  dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dim) + d] =
      src[static_cast<std::size_t>(s) * static_cast<std::size_t>(dim) + d];
}

void validate_dimensions(const Eg2Config& cfg, std::optional<int> dimensions, int base_dim) {
  if (!dimensions.has_value()) return;
  const int d = *dimensions;
  if (d <= 0 || d > base_dim) {
    throw std::invalid_argument("EmbeddingGemma2: requested dimensions " + std::to_string(d) +
                                " out of range [1, " + std::to_string(base_dim) + "]");
  }
  if (base_dim == cfg.embedding_dim && !cfg.matryoshka_dims.empty()) {
    if (std::find(cfg.matryoshka_dims.begin(), cfg.matryoshka_dims.end(), d) == cfg.matryoshka_dims.end()) {
      throw std::invalid_argument("EmbeddingGemma2: requested dimensions " + std::to_string(d) +
                                  " is not in supported Matryoshka dims {768, 512, 256, 128}");
    }
  }
}

}  // namespace

struct Eg2CudaBackend::Impl {
  template <typename ElemT>
  struct LayerDeviceWeights {
    DevBuf<ElemT> input_layernorm;
    DevBuf<ElemT> q_proj;
    DevBuf<ElemT> k_proj;
    DevBuf<ElemT> v_proj;
    DevBuf<ElemT> o_proj;
    DevBuf<ElemT> q_norm;
    DevBuf<ElemT> k_norm;
    DevBuf<ElemT> post_attention_layernorm;
    DevBuf<ElemT> pre_feedforward_layernorm;
    DevBuf<ElemT> gate_proj;
    DevBuf<ElemT> up_proj;
    DevBuf<ElemT> down_proj;
    DevBuf<ElemT> post_feedforward_layernorm;
    DevBuf<ElemT> per_layer_input_gate;
    DevBuf<ElemT> per_layer_projection;
    DevBuf<ElemT> post_per_layer_input_norm;
    float layer_scalar = 1.0f;
  };

  template <typename ElemT>
  struct WeightBank {
    DevBuf<ElemT> embed_tokens;
    DevBuf<ElemT> per_layer_model_projection;
    DevBuf<ElemT> per_layer_projection_norm;
    std::vector<LayerDeviceWeights<ElemT>> layers;
    DevBuf<ElemT> norm;
    DevBuf<ElemT> embedding_projection;
    std::uint64_t total_bytes = 0;

    // Reusable activation scratch (grows monotonically up to peak request size).
    DevBuf<ElemT> hidden;
    DevBuf<ElemT> ple;
    DevBuf<ElemT> norm_buf;
    DevBuf<ElemT> sub_out;
    DevBuf<ElemT> q_buf;
    DevBuf<ElemT> k_buf;
    DevBuf<ElemT> v_buf;
    DevBuf<ElemT> attn_ctx;
    DevBuf<ElemT> gate_buf;
    DevBuf<ElemT> up_buf;
    DevBuf<ElemT> ple_gate;
  };

  template <typename ElemT>
  struct VisionLayerDeviceWeights {
    DevBuf<ElemT> input_layernorm;
    DevBuf<ElemT> q_proj;
    DevBuf<ElemT> k_proj;
    DevBuf<ElemT> v_proj;
    DevBuf<ElemT> o_proj;
    DevBuf<ElemT> q_norm;
    DevBuf<ElemT> k_norm;
    DevBuf<ElemT> post_attention_layernorm;
    DevBuf<ElemT> pre_feedforward_layernorm;
    DevBuf<ElemT> gate_proj;
    DevBuf<ElemT> up_proj;
    DevBuf<ElemT> down_proj;
    DevBuf<ElemT> post_feedforward_layernorm;
  };

  template <typename ElemT>
  struct VisionWeightBank {
    DevBuf<ElemT> patch_input_proj;
    DevBuf<ElemT> position_embedding_table;
    std::vector<VisionLayerDeviceWeights<ElemT>> layers;
    DevBuf<ElemT> embed_vision_projection;
    std::uint64_t total_bytes = 0;

    // Reusable vision scratch buffers
    DevBuf<float> pv_f32;
    DevBuf<ElemT> pv_elem;
    DevBuf<ElemT> hidden;
    DevBuf<ElemT> norm_buf;
    DevBuf<ElemT> sub_out;
    DevBuf<ElemT> q_buf;
    DevBuf<ElemT> k_buf;
    DevBuf<ElemT> v_buf;
    DevBuf<ElemT> q_hnd;
    DevBuf<ElemT> k_hnd;
    DevBuf<ElemT> v_hdn;
    DevBuf<ElemT> o_hnd;
    DevBuf<float> attn_scores_f32;
    DevBuf<ElemT> attn_probs;
    DevBuf<ElemT> attn_ctx;
    DevBuf<ElemT> gate_buf;
    DevBuf<ElemT> up_buf;
    DevBuf<ElemT> pooled_normed;
    DevBuf<ElemT> soft_out;  // contiguous [total_soft_tokens, text_H] across all images in batch
  };

  template <typename ElemT>
  struct AudioClippedLinearDev {
    DevBuf<ElemT> weight;
    float input_min = -1e30f;
    float input_max = 1e30f;
    float output_min = -1e30f;
    float output_max = 1e30f;

    template <typename HostClip>
    std::uint64_t upload(const HostClip& h, cudaStream_t stream) {
      weight.upload(h.weight, stream);
      input_min = h.input_min;
      input_max = h.input_max;
      output_min = h.output_min;
      output_max = h.output_max;
      return weight.bytes();
    }
  };

  template <typename ElemT>
  struct AudioFfwDev {
    DevBuf<ElemT> pre_layer_norm;
    AudioClippedLinearDev<ElemT> ffw_layer_1;
    AudioClippedLinearDev<ElemT> ffw_layer_2;
    DevBuf<ElemT> post_layer_norm;
  };

  template <typename ElemT>
  struct AudioLayerDeviceWeights {
    AudioFfwDev<ElemT> feed_forward1;
    DevBuf<ElemT> norm_pre_attn;
    AudioClippedLinearDev<ElemT> q_proj;
    AudioClippedLinearDev<ElemT> k_proj;
    AudioClippedLinearDev<ElemT> v_proj;
    DevBuf<ElemT> relative_k_proj;
    DevBuf<float> per_dim_scale;
    AudioClippedLinearDev<ElemT> post;
    DevBuf<ElemT> norm_post_attn;
    DevBuf<ElemT> lconv_pre_layer_norm;
    AudioClippedLinearDev<ElemT> lconv_linear_start;
    DevBuf<ElemT> lconv_depthwise_conv1d;
    DevBuf<ElemT> lconv_conv_norm;
    AudioClippedLinearDev<ElemT> lconv_linear_end;
    AudioFfwDev<ElemT> feed_forward2;
    DevBuf<ElemT> norm_out;
  };

  template <typename ElemT>
  struct AudioWeightBank {
    DevBuf<ElemT> sscp_layer0_conv;
    DevBuf<ElemT> sscp_layer0_norm;
    DevBuf<ElemT> sscp_layer1_conv;
    DevBuf<ElemT> sscp_layer1_norm;
    DevBuf<ElemT> sscp_input_proj_linear;
    DevBuf<ElemT> rel_pos_embed;  // [13, AH]
    std::vector<AudioLayerDeviceWeights<ElemT>> layers;
    DevBuf<ElemT> output_proj_weight;
    DevBuf<ElemT> output_proj_bias;
    DevBuf<ElemT> embed_audio_projection;
    std::uint64_t total_bytes = 0;

    // Reusable audio scratch buffers
    DevBuf<float> mel_f32;
    DevBuf<ElemT> mel_elem;
    DevBuf<ElemT> s0_buf;
    DevBuf<ElemT> s1_buf;
    DevBuf<ElemT> s1_tfc;
    DevBuf<ElemT> hidden;
    DevBuf<ElemT> norm_buf;
    DevBuf<ElemT> clip_buf;
    DevBuf<ElemT> ffw_mid;
    DevBuf<ElemT> sub_out;
    DevBuf<ElemT> q_buf;
    DevBuf<ElemT> k_buf;
    DevBuf<ElemT> v_buf;
    DevBuf<ElemT> rel_k;
    DevBuf<ElemT> attn_ctx;
    DevBuf<ElemT> lconv_start;
    DevBuf<ElemT> lconv_glu;
    DevBuf<ElemT> lconv_dw;
    DevBuf<ElemT> aud_proj;
    DevBuf<ElemT> aud_valid_proj;
    DevBuf<ElemT> soft_out;  // contiguous [total_audio_soft_tokens, text_H]
  };

  int device = 0;
  cudaStream_t stream = nullptr;
  cublasLtHandle_t lt = nullptr;
  DevBuf<std::uint8_t> lt_workspace;
  bool use_bf16 = false;
  std::mutex mu;

  WeightBank<std::uint16_t> w_bf16;
  WeightBank<float> w_f32;
  VisionWeightBank<std::uint16_t> v_bf16;
  VisionWeightBank<float> v_f32;
  AudioWeightBank<std::uint16_t> a_bf16;
  AudioWeightBank<float> a_f32;

  DevBuf<std::int32_t> d_tokens;
  DevBuf<std::int32_t> d_offsets;
  DevBuf<std::int32_t> d_seq_of_tok;
  DevBuf<std::int32_t> d_pos_of_tok;
  DevBuf<std::int32_t> d_vis_pos_x;
  DevBuf<std::int32_t> d_vis_pos_y;
  DevBuf<std::int32_t> d_vis_offsets;
  DevBuf<std::int32_t> d_vis_seq_of_patch;
  DevBuf<std::int32_t> d_vis_scatter_dst;
  DevBuf<std::uint8_t> d_aud_mel_mask;
  DevBuf<std::uint8_t> d_aud_l0_mask;
  DevBuf<std::uint8_t> d_aud_valid_mask;
  DevBuf<std::int32_t> d_aud_valid_indices;
  DevBuf<std::int32_t> d_aud_scatter_dst;
  DevBuf<float> d_pooled;
  DevBuf<float> d_projected;
  DevBuf<float> d_final;

  std::shared_ptr<const PlatosKeyGpu> platos_key_gpu;

  Impl() {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublas_check(cublasLtCreate(&lt), "create");
    lt_workspace.ensure(32ull * 1024ull * 1024ull);
  }

  ~Impl() {
    if (lt) cublasLtDestroy(lt);
    if (stream) cudaStreamDestroy(stream);
  }

  // Strided-batched row-major GEMM: Y[b][M, N] = alpha * X[b][M, K] @ W[b][N, K]^T.
  // Input dtype is BF16 (sm_80+) or FP32 (sm_70); output is FP32 when out_f32=true, else matches input.
  void gemm_strided_batched_rowmajor(const void* x, const void* w, void* y, int M, int N, int K, int batch_count,
                                     std::int64_t x_stride, std::int64_t w_stride, std::int64_t y_stride,
                                     bool out_f32 = false, float alpha = 1.0f) {
    const cudaDataType_t in_dtype = use_bf16 ? CUDA_R_16BF : CUDA_R_32F;
    const cudaDataType_t out_dtype = out_f32 ? CUDA_R_32F : in_dtype;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&] {
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
      if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
      if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
      if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    };
    try {
      cublas_check(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
      const cublasOperation_t trans_a = CUBLAS_OP_T;
      const cublasOperation_t trans_b = CUBLAS_OP_N;
      cublas_check(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
                   "SetTransA");
      cublas_check(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
                   "SetTransB");
      // Col-major view: A = W [K, N], B = X [K, M], C = Y [N, M]
      cublas_check(cublasLtMatrixLayoutCreate(&a_desc, in_dtype, static_cast<std::uint64_t>(K),
                                              static_cast<std::uint64_t>(N), static_cast<std::int64_t>(K)),
                   "LayoutA");
      cublas_check(cublasLtMatrixLayoutCreate(&b_desc, in_dtype, static_cast<std::uint64_t>(K),
                                              static_cast<std::uint64_t>(M), static_cast<std::int64_t>(K)),
                   "LayoutB");
      cublas_check(cublasLtMatrixLayoutCreate(&c_desc, out_dtype, static_cast<std::uint64_t>(N),
                                              static_cast<std::uint64_t>(M), static_cast<std::int64_t>(N)),
                   "LayoutC");
      if (batch_count > 1) {
        const std::int32_t bc = batch_count;
        cublas_check(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)),
                     "BatchA");
        cublas_check(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)),
                     "BatchB");
        cublas_check(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)),
                     "BatchC");
        cublas_check(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &w_stride,
                                                      sizeof(w_stride)),
                     "StrideA");
        cublas_check(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &x_stride,
                                                      sizeof(x_stride)),
                     "StrideB");
        cublas_check(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &y_stride,
                                                      sizeof(y_stride)),
                     "StrideC");
      }
      cublas_check(cublasLtMatmulPreferenceCreate(&pref), "PrefCreate");
      const std::size_t ws_bytes = lt_workspace.count;
      cublas_check(
          cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)),
          "PrefWorkspace");

      cublasLtMatmulHeuristicResult_t heur{};
      int returned = 0;
      cublas_check(cublasLtMatmulAlgoGetHeuristic(lt, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heur, &returned),
                   "AlgoGetHeuristic");
      if (returned <= 0) {
        throw std::runtime_error("Eg2Cuda cuBLASLt found no valid heuristic for M=" + std::to_string(M) +
                                 " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                 " B=" + std::to_string(batch_count));
      }
      const float beta = 0.0f;
      cublas_check(cublasLtMatmul(lt, op_desc, &alpha, w, a_desc, x, b_desc, &beta, y, c_desc, y, c_desc, &heur.algo,
                                  lt_workspace.ptr, ws_bytes, stream),
                   "Matmul");
      cleanup();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  // Row-major GEMM: Y[M, N] = alpha * X[M, K] @ W[N, K]^T (zero FP16 — strictly BF16 or FP32).
  void gemm_rowmajor(const void* x, const void* w, void* y, int M, int N, int K, float alpha = 1.0f) {
    gemm_strided_batched_rowmajor(x, w, y, M, N, K, /*batch_count=*/1, 0, 0, 0, /*out_f32=*/false, alpha);
  }

  template <typename ElemT>
  void encode_vision_batch_typed(const Eg2Config& cfg, VisionWeightBank<ElemT>& vw,
                                 const std::vector<Eg2PreprocessedImage>& preps, int total_soft) {
    if (preps.empty()) return;
    const Eg2VisionConfig& vc = cfg.vision;
    const int VH = vc.hidden_size;
    const int VI = vc.intermediate_size;
    const int TH = cfg.hidden_size;
    const int n_heads = vc.num_attention_heads;
    const int n_kv = vc.num_key_value_heads;
    const int hd = vc.head_dim;
    const int q_dim = n_heads * hd;
    const int kv_dim = n_kv * hd;
    const int patch_in = 3 * vc.patch_size * vc.patch_size;

    vw.soft_out.ensure(static_cast<std::size_t>(total_soft) * static_cast<std::size_t>(TH));
    int soft_offset = 0;

    for (const Eg2PreprocessedImage& prep : preps) {
      const int N = prep.size.num_patches;
      const int n_soft = prep.size.num_soft_tokens;
      const std::size_t pv_numel = static_cast<std::size_t>(N) * static_cast<std::size_t>(patch_in);

      vw.pv_f32.upload(prep.patches_f32, stream);
      vw.pv_elem.ensure(pv_numel);
      vw.hidden.ensure(static_cast<std::size_t>(N) * VH);
      vw.norm_buf.ensure(static_cast<std::size_t>(N) * VH);
      vw.sub_out.ensure(static_cast<std::size_t>(N) * VH);
      vw.q_buf.ensure(static_cast<std::size_t>(N) * q_dim);
      vw.k_buf.ensure(static_cast<std::size_t>(N) * kv_dim);
      vw.v_buf.ensure(static_cast<std::size_t>(N) * kv_dim);
      vw.attn_ctx.ensure(static_cast<std::size_t>(N) * q_dim);
      const bool use_gemm_attn = (n_heads == n_kv);
      const int q_tile = use_gemm_attn
                             ? std::max(64, std::min(N, static_cast<int>((16ull * 1024ull * 1024ull) /
                                                                         (static_cast<std::size_t>(n_heads) *
                                                                          static_cast<std::size_t>(N)))))
                             : 0;
      if (use_gemm_attn) {
        vw.q_hnd.ensure(static_cast<std::size_t>(N) * q_dim);
        vw.k_hnd.ensure(static_cast<std::size_t>(N) * kv_dim);
        vw.v_hdn.ensure(static_cast<std::size_t>(N) * kv_dim);
        vw.o_hnd.ensure(static_cast<std::size_t>(N) * q_dim);
        const std::size_t tile_elems =
            static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(q_tile) * static_cast<std::size_t>(N);
        vw.attn_scores_f32.ensure(tile_elems);
        vw.attn_probs.ensure(tile_elems);
      }
      vw.gate_buf.ensure(static_cast<std::size_t>(N) * VI);
      vw.up_buf.ensure(static_cast<std::size_t>(N) * VI);
      vw.pooled_normed.ensure(static_cast<std::size_t>(n_soft) * VH);

      d_vis_pos_x.upload(prep.pos_x, stream);
      d_vis_pos_y.upload(prep.pos_y, stream);
      const std::vector<std::int32_t> single_off = {0, N};
      const std::vector<std::int32_t> single_seq(static_cast<std::size_t>(N), 0);
      d_vis_offsets.upload(single_off, stream);
      d_vis_seq_of_patch.upload(single_seq, stream);

      // 1. PatchEmbedder: 2*(pv - 0.5) -> Linear -> + pos_table[0, x] + pos_table[1, y]
      scale_patches_to_elem_kernel<ElemT>
          <<<static_cast<unsigned>((pv_numel + 255) / 256), 256, 0, stream>>>(vw.pv_f32.ptr, vw.pv_elem.ptr, pv_numel);
      gemm_rowmajor(vw.pv_elem.ptr, vw.patch_input_proj.ptr, vw.hidden.ptr, N, VH, patch_in);
      {
        const dim3 grid(static_cast<unsigned>((VH + 255) / 256), static_cast<unsigned>(N));
        add_2d_pos_embed_kernel<ElemT><<<grid, 256, 0, stream>>>(
            vw.hidden.ptr, vw.position_embedding_table.ptr, d_vis_pos_x.ptr, d_vis_pos_y.ptr, N,
            vc.position_embedding_size, VH);
      }

      // 2. 16 bidirectional ViT layers (2D axial RoPE, full attention, scaling = 1.0, GeGLU)
      for (int li = 0; li < vc.num_hidden_layers; ++li) {
        const auto& lw = vw.layers[static_cast<std::size_t>(li)];
        rms_norm_kernel<ElemT><<<static_cast<unsigned>(N), 256, 0, stream>>>(
            vw.hidden.ptr, lw.input_layernorm.ptr, N, VH, vc.rms_norm_eps, vw.norm_buf.ptr);
        gemm_rowmajor(vw.norm_buf.ptr, lw.q_proj.ptr, vw.q_buf.ptr, N, q_dim, VH);
        gemm_rowmajor(vw.norm_buf.ptr, lw.k_proj.ptr, vw.k_buf.ptr, N, kv_dim, VH);
        gemm_rowmajor(vw.norm_buf.ptr, lw.v_proj.ptr, vw.v_buf.ptr, N, kv_dim, VH);

        head_rmsnorm_axial_2d_rope_kernel<ElemT><<<static_cast<unsigned>(N * n_heads), 64, 0, stream>>>(
            vw.q_buf.ptr, lw.q_norm.ptr, d_vis_pos_x.ptr, d_vis_pos_y.ptr, N, n_heads, hd, vc.rms_norm_eps,
            vc.rope_theta);
        head_rmsnorm_axial_2d_rope_kernel<ElemT><<<static_cast<unsigned>(N * n_kv), 64, 0, stream>>>(
            vw.k_buf.ptr, lw.k_norm.ptr, d_vis_pos_x.ptr, d_vis_pos_y.ptr, N, n_kv, hd, vc.rms_norm_eps, vc.rope_theta);
        rms_norm_kernel<ElemT><<<static_cast<unsigned>(N * n_kv), 64, 0, stream>>>(
            vw.v_buf.ptr, nullptr, N * n_kv, hd, vc.rms_norm_eps, vw.v_buf.ptr);

        if (use_gemm_attn) {
          const std::size_t total_q = static_cast<std::size_t>(N) * static_cast<std::size_t>(q_dim);
          const unsigned tb = static_cast<unsigned>((total_q + 255) / 256);
          transpose_nhd_to_hnd_kernel<ElemT><<<tb, 256, 0, stream>>>(vw.q_buf.ptr, vw.q_hnd.ptr, N, n_heads, hd);
          transpose_nhd_to_hnd_kernel<ElemT><<<tb, 256, 0, stream>>>(vw.k_buf.ptr, vw.k_hnd.ptr, N, n_heads, hd);
          transpose_nhd_to_hdn_kernel<ElemT><<<tb, 256, 0, stream>>>(vw.v_buf.ptr, vw.v_hdn.ptr, N, n_heads, hd);
          const std::int64_t head_stride = static_cast<std::int64_t>(N) * static_cast<std::int64_t>(hd);
          for (int q0 = 0; q0 < N; q0 += q_tile) {
            const int qm = std::min(q_tile, N - q0);
            const std::int64_t score_stride = static_cast<std::int64_t>(qm) * static_cast<std::int64_t>(N);
            const ElemT* q_tile_ptr = vw.q_hnd.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(hd);
            ElemT* o_tile_ptr = vw.o_hnd.ptr + static_cast<std::size_t>(q0) * static_cast<std::size_t>(hd);
            // Q_h[q0..q0+qm, D] @ K_h[N, D]^T -> S_f32[qm, N] (exact FP32 logits before softmax)
            gemm_strided_batched_rowmajor(q_tile_ptr, vw.k_hnd.ptr, vw.attn_scores_f32.ptr, qm, N, hd, n_heads,
                                          head_stride, head_stride, score_stride, /*out_f32=*/true);
            softmax_rows_f32_to_elem_kernel<ElemT><<<static_cast<unsigned>(n_heads * qm), 256, 0, stream>>>(
                vw.attn_scores_f32.ptr, vw.attn_probs.ptr, n_heads * qm, N);
            // P[qm, N] @ V_ht[D, N]^T -> O_h[q0..q0+qm, D]
            gemm_strided_batched_rowmajor(vw.attn_probs.ptr, vw.v_hdn.ptr, o_tile_ptr, qm, hd, N, n_heads,
                                          score_stride, head_stride, head_stride, /*out_f32=*/false);
          }
          transpose_hnd_to_nhd_kernel<ElemT><<<tb, 256, 0, stream>>>(vw.o_hnd.ptr, vw.attn_ctx.ptr, N, n_heads, hd);
        } else {
          const dim3 attn_grid(static_cast<unsigned>(N), static_cast<unsigned>(n_heads));
          bidir_attention_kernel<ElemT><<<attn_grid, 128, 0, stream>>>(
              vw.q_buf.ptr, vw.k_buf.ptr, vw.v_buf.ptr, d_vis_offsets.ptr, d_vis_seq_of_patch.ptr, N, n_heads, n_kv, hd,
              0, /*is_sliding=*/false, vw.attn_ctx.ptr);
        }
        gemm_rowmajor(vw.attn_ctx.ptr, lw.o_proj.ptr, vw.sub_out.ptr, N, VH, q_dim);
        rms_norm_residual_add_kernel<ElemT><<<static_cast<unsigned>(N), 256, 0, stream>>>(
            vw.sub_out.ptr, lw.post_attention_layernorm.ptr, N, VH, vc.rms_norm_eps, vw.hidden.ptr);

        rms_norm_kernel<ElemT><<<static_cast<unsigned>(N), 256, 0, stream>>>(
            vw.hidden.ptr, lw.pre_feedforward_layernorm.ptr, N, VH, vc.rms_norm_eps, vw.norm_buf.ptr);
        gemm_rowmajor(vw.norm_buf.ptr, lw.gate_proj.ptr, vw.gate_buf.ptr, N, VI, VH);
        gemm_rowmajor(vw.norm_buf.ptr, lw.up_proj.ptr, vw.up_buf.ptr, N, VI, VH);
        {
          const std::size_t n_mlp = static_cast<std::size_t>(N) * static_cast<std::size_t>(VI);
          geglu_mul_kernel<ElemT><<<static_cast<unsigned>((n_mlp + 255) / 256), 256, 0, stream>>>(
              vw.gate_buf.ptr, vw.up_buf.ptr, n_mlp);
        }
        gemm_rowmajor(vw.gate_buf.ptr, lw.down_proj.ptr, vw.sub_out.ptr, N, VH, VI);
        rms_norm_residual_add_kernel<ElemT><<<static_cast<unsigned>(N), 256, 0, stream>>>(
            vw.sub_out.ptr, lw.post_feedforward_layernorm.ptr, N, VH, vc.rms_norm_eps, vw.hidden.ptr);
      }

      // 3. 3x3 spatial pool -> * sqrt(VH) -> unscaled RMSNorm -> embed_vision.embedding_projection -> soft_out slice
      vision_spatial_pool_norm_kernel<ElemT><<<static_cast<unsigned>(n_soft), 256, 0, stream>>>(
          vw.hidden.ptr, prep.size.patch_height, prep.size.patch_width, vc.pooling_kernel_size, VH, vc.rms_norm_eps,
          vw.pooled_normed.ptr);
      ElemT* dst_slice = vw.soft_out.ptr + static_cast<std::size_t>(soft_offset) * static_cast<std::size_t>(TH);
      gemm_rowmajor(vw.pooled_normed.ptr, vw.embed_vision_projection.ptr, dst_slice, n_soft, TH, VH);
      soft_offset += n_soft;
    }
  }

  template <typename ElemT>
  void audio_clipped_gemm(const ElemT* x, const AudioClippedLinearDev<ElemT>& lin, bool use_clip, int rows, int in_dim,
                          int out_dim, ElemT* out, DevBuf<ElemT>& clip_buf) {
    const ElemT* src = x;
    if (use_clip) {
      const std::size_t n_in = static_cast<std::size_t>(rows) * static_cast<std::size_t>(in_dim);
      clip_buf.ensure(n_in);
      audio_clamp_copy_kernel<ElemT><<<static_cast<unsigned>((n_in + 255) / 256), 256, 0, stream>>>(
          x, clip_buf.ptr, n_in, lin.input_min, lin.input_max);
      src = clip_buf.ptr;
    }
    gemm_rowmajor(src, lin.weight.ptr, out, rows, out_dim, in_dim);
    if (use_clip) {
      const std::size_t n_out = static_cast<std::size_t>(rows) * static_cast<std::size_t>(out_dim);
      audio_clamp_copy_kernel<ElemT><<<static_cast<unsigned>((n_out + 255) / 256), 256, 0, stream>>>(
          out, out, n_out, lin.output_min, lin.output_max);
    }
  }

  template <typename ElemT>
  void encode_audio_batch_typed(const Eg2Config& cfg, AudioWeightBank<ElemT>& aw,
                                const std::vector<Eg2PreprocessedAudio>& preps, int total_soft) {
    if (preps.empty()) return;
    const Eg2AudioConfig& ac = cfg.audio;
    const int AH = ac.hidden_size;             // 1024
    const int n_heads = ac.num_attention_heads;  // 8
    const int hd = AH / n_heads;               // 128
    const int max_past = ac.attention_context_left - 1;  // 12
    const int ctx_size = ac.attention_chunk_size + max_past + ac.attention_context_right;
    const int pos_len = ctx_size / 2 + 1;      // 13
    const int C0 = ac.sscp_conv_channels_0;    // 128
    const int C1 = ac.sscp_conv_channels_1;    // 32
    const int OP = ac.output_proj_dims;        // 1536
    const int TH = cfg.hidden_size;            // 512
    const int K = ac.conv_kernel_size;         // 5
    const float eps = ac.rms_norm_eps;
    const float grad_clip = use_bf16 ? std::min(ac.gradient_clipping, 3.38953139e38f) : ac.gradient_clipping;
    const float res_w = ac.residual_weight;
    const float softcap = ac.attention_logit_cap;
    const bool use_clip = ac.use_clipped_linears;
    const float q_scale = static_cast<float>((1.0 / std::sqrt(static_cast<double>(hd))) / std::log(2.0));
    const float k_scale = static_cast<float>(std::log(1.0 + std::exp(1.0)) / std::log(2.0));

    aw.soft_out.ensure(static_cast<std::size_t>(total_soft) * static_cast<std::size_t>(TH));
    int soft_offset = 0;

    for (const Eg2PreprocessedAudio& prep : preps) {
      const int T_mel = prep.mel_frames;
      const int F_mel = prep.num_mels;  // 128
      const int T1 = (T_mel + 2 - 3) / 2 + 1;
      const int F1 = (F_mel + 2 - 3) / 2 + 1;  // 64
      const int T_seq = (T1 + 2 - 3) / 2 + 1;
      const int F2 = (F1 + 2 - 3) / 2 + 1;     // 32
      const int sscp_flat_dim = F2 * C1;       // 1024
      const int n_valid = prep.num_valid_tokens;

      std::vector<std::uint8_t> h_l0_mask(static_cast<std::size_t>(T1), 0);
      for (int t = 0; t < T1; ++t) h_l0_mask[static_cast<std::size_t>(t)] = prep.mel_mask[static_cast<std::size_t>(t * 2)];
      std::vector<std::int32_t> h_valid_idx;
      h_valid_idx.reserve(static_cast<std::size_t>(n_valid));
      for (int t = 0; t < T_seq; ++t) {
        if (prep.valid_mask[static_cast<std::size_t>(t)] != 0) h_valid_idx.push_back(t);
      }

      const std::size_t mel_numel = static_cast<std::size_t>(T_mel) * static_cast<std::size_t>(F_mel);
      aw.mel_f32.upload(prep.mel_f32, stream);
      d_aud_mel_mask.upload(prep.mel_mask, stream);
      d_aud_l0_mask.upload(h_l0_mask, stream);
      d_aud_valid_mask.upload(prep.valid_mask, stream);
      d_aud_valid_indices.upload(h_valid_idx, stream);

      aw.mel_elem.ensure(mel_numel);
      aw.s0_buf.ensure(static_cast<std::size_t>(C0) * static_cast<std::size_t>(T1) * static_cast<std::size_t>(F1));
      aw.s1_buf.ensure(static_cast<std::size_t>(C1) * static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(F2));
      aw.s1_tfc.ensure(static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(sscp_flat_dim));
      aw.hidden.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.norm_buf.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.ffw_mid.ensure(static_cast<std::size_t>(T_seq) * (4 * AH));
      aw.sub_out.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.q_buf.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.k_buf.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.v_buf.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.rel_k.ensure(static_cast<std::size_t>(pos_len) * AH);
      aw.attn_ctx.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.lconv_start.ensure(static_cast<std::size_t>(T_seq) * (2 * AH));
      aw.lconv_glu.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.lconv_dw.ensure(static_cast<std::size_t>(T_seq) * AH);
      aw.aud_proj.ensure(static_cast<std::size_t>(T_seq) * OP);
      aw.aud_valid_proj.ensure(static_cast<std::size_t>(n_valid) * OP);

      // 1. SSCP Conv2d layer0 + layer1 + permute + input_proj_linear
      audio_copy_f32_to_elem_kernel<ElemT><<<static_cast<unsigned>((mel_numel + 255) / 256), 256, 0, stream>>>(
          aw.mel_f32.ptr, aw.mel_elem.ptr, mel_numel);
      audio_sscp_conv_ln_relu_kernel<ElemT><<<static_cast<unsigned>(T1 * F1), 128, 0, stream>>>(
          aw.mel_elem.ptr, d_aud_mel_mask.ptr, aw.sscp_layer0_conv.ptr, aw.sscp_layer0_norm.ptr, 1, T_mel, F_mel, C0,
          eps, aw.s0_buf.ptr);
      audio_sscp_conv_ln_relu_kernel<ElemT><<<static_cast<unsigned>(T_seq * F2), 128, 0, stream>>>(
          aw.s0_buf.ptr, d_aud_l0_mask.ptr, aw.sscp_layer1_conv.ptr, aw.sscp_layer1_norm.ptr, C0, T1, F1, C1, eps,
          aw.s1_buf.ptr);
      {
        const std::size_t n_tfc = static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(sscp_flat_dim);
        audio_sscp_permute_tfc_kernel<ElemT><<<static_cast<unsigned>((n_tfc + 255) / 256), 256, 0, stream>>>(
            aw.s1_buf.ptr, C1, T_seq, F2, aw.s1_tfc.ptr);
      }
      gemm_rowmajor(aw.s1_tfc.ptr, aw.sscp_input_proj_linear.ptr, aw.hidden.ptr, T_seq, AH, sscp_flat_dim);

      auto run_ffw = [&](const AudioFfwDev<ElemT>& ffw) {
        audio_clamp_rmsnorm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.hidden.ptr, ffw.pre_layer_norm.ptr, T_seq, AH, eps, grad_clip, aw.norm_buf.ptr);
        audio_clipped_gemm(aw.norm_buf.ptr, ffw.ffw_layer_1, use_clip, T_seq, AH, 4 * AH, aw.ffw_mid.ptr, aw.clip_buf);
        {
          const std::size_t n_mid = static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(4 * AH);
          audio_silu_inplace_kernel<ElemT><<<static_cast<unsigned>((n_mid + 255) / 256), 256, 0, stream>>>(
              aw.ffw_mid.ptr, n_mid);
        }
        audio_clipped_gemm(aw.ffw_mid.ptr, ffw.ffw_layer_2, use_clip, T_seq, 4 * AH, AH, aw.sub_out.ptr, aw.clip_buf);
        audio_clamp_rmsnorm_residual_scale_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.sub_out.ptr, ffw.post_layer_norm.ptr, T_seq, AH, eps, grad_clip, res_w, aw.hidden.ptr);
      };

      const std::size_t n_hid = static_cast<std::size_t>(T_seq) * static_cast<std::size_t>(AH);

      // 2. 12 Conformer layers
      for (int li = 0; li < ac.num_hidden_layers; ++li) {
        const auto& lw = aw.layers[static_cast<std::size_t>(li)];

        run_ffw(lw.feed_forward1);

        // Self-Attention
        audio_clamp_rmsnorm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.hidden.ptr, lw.norm_pre_attn.ptr, T_seq, AH, eps, grad_clip, aw.norm_buf.ptr);
        audio_clipped_gemm(aw.norm_buf.ptr, lw.q_proj, use_clip, T_seq, AH, AH, aw.q_buf.ptr, aw.clip_buf);
        audio_clipped_gemm(aw.norm_buf.ptr, lw.k_proj, use_clip, T_seq, AH, AH, aw.k_buf.ptr, aw.clip_buf);
        audio_clipped_gemm(aw.norm_buf.ptr, lw.v_proj, use_clip, T_seq, AH, AH, aw.v_buf.ptr, aw.clip_buf);
        gemm_rowmajor(aw.rel_pos_embed.ptr, lw.relative_k_proj.ptr, aw.rel_k.ptr, pos_len, AH, AH);
        {
          const dim3 grid(static_cast<unsigned>(T_seq), static_cast<unsigned>(n_heads));
          audio_local_rel_attn_kernel<ElemT><<<grid, 128, 0, stream>>>(
              aw.q_buf.ptr, aw.k_buf.ptr, aw.v_buf.ptr, aw.rel_k.ptr, lw.per_dim_scale.ptr, d_aud_valid_mask.ptr,
              T_seq, n_heads, hd, max_past, q_scale, k_scale, softcap, aw.attn_ctx.ptr);
        }
        audio_clipped_gemm(aw.attn_ctx.ptr, lw.post, use_clip, T_seq, AH, AH, aw.sub_out.ptr, aw.clip_buf);
        audio_clamp_rmsnorm_residual_scale_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.sub_out.ptr, lw.norm_post_attn.ptr, T_seq, AH, eps, grad_clip, 1.0f, aw.hidden.ptr);

        // LightConv1d
        rms_norm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.hidden.ptr, lw.lconv_pre_layer_norm.ptr, T_seq, AH, eps, aw.norm_buf.ptr);
        audio_clipped_gemm(aw.norm_buf.ptr, lw.lconv_linear_start, use_clip, T_seq, AH, 2 * AH, aw.lconv_start.ptr,
                           aw.clip_buf);
        {
          const dim3 grid(static_cast<unsigned>((AH + 255) / 256), static_cast<unsigned>(T_seq));
          audio_glu_kernel<ElemT><<<grid, 256, 0, stream>>>(aw.lconv_start.ptr, aw.lconv_glu.ptr, T_seq, AH);
          audio_causal_dw_conv1d_kernel<ElemT><<<grid, 256, 0, stream>>>(
              aw.lconv_glu.ptr, lw.lconv_depthwise_conv1d.ptr, T_seq, AH, K, grad_clip, aw.lconv_dw.ptr);
        }
        rms_norm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.lconv_dw.ptr, lw.lconv_conv_norm.ptr, T_seq, AH, eps, aw.lconv_dw.ptr);
        audio_silu_inplace_kernel<ElemT><<<static_cast<unsigned>((n_hid + 255) / 256), 256, 0, stream>>>(
            aw.lconv_dw.ptr, n_hid);
        audio_clipped_gemm(aw.lconv_dw.ptr, lw.lconv_linear_end, use_clip, T_seq, AH, AH, aw.sub_out.ptr, aw.clip_buf);
        audio_residual_add_kernel<ElemT><<<static_cast<unsigned>((n_hid + 255) / 256), 256, 0, stream>>>(
            aw.hidden.ptr, aw.sub_out.ptr, n_hid);

        run_ffw(lw.feed_forward2);

        audio_clamp_rmsnorm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
            aw.hidden.ptr, lw.norm_out.ptr, T_seq, AH, eps, grad_clip, aw.hidden.ptr);
      }

      // 3. output_proj + bias + unscaled RMSNorm -> compact valid rows -> embed_audio.embedding_projection
      gemm_rowmajor(aw.hidden.ptr, aw.output_proj_weight.ptr, aw.aud_proj.ptr, T_seq, OP, AH);
      audio_bias_unscaled_rmsnorm_kernel<ElemT><<<static_cast<unsigned>(T_seq), 256, 0, stream>>>(
          aw.aud_proj.ptr, aw.output_proj_bias.ptr, T_seq, OP, eps);
      {
        const dim3 grid(static_cast<unsigned>((OP + 255) / 256), static_cast<unsigned>(n_valid));
        audio_compact_rows_kernel<ElemT><<<grid, 256, 0, stream>>>(aw.aud_proj.ptr, d_aud_valid_indices.ptr, n_valid,
                                                                   OP, aw.aud_valid_proj.ptr);
      }
      ElemT* dst_slice = aw.soft_out.ptr + static_cast<std::size_t>(soft_offset) * static_cast<std::size_t>(TH);
      gemm_rowmajor(aw.aud_valid_proj.ptr, aw.embed_audio_projection.ptr, dst_slice, n_valid, TH, OP);
      soft_offset += n_valid;
    }
  }

  template <typename ElemT>
  void run_forward_typed(const Eg2Config& cfg, WeightBank<ElemT>& wb, int B, int T,
                         const model::EmbeddingRequest& req, float* d_pooled_out,
                         const ElemT* d_vision_soft = nullptr, int n_vision_soft = 0,
                         const ElemT* d_audio_soft = nullptr, int n_audio_soft = 0) {
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int L = cfg.num_hidden_layers;
    const int stop_layer = req.tap_layer.value_or(L - 1);

    int max_q_dim = 0;
    int max_kv_dim = 0;
    for (int li = 0; li <= stop_layer; ++li) {
      const LayerSpec& ls = cfg.layers[static_cast<std::size_t>(li)];
      max_q_dim = std::max(max_q_dim, ls.num_attention_heads * ls.head_dim);
      max_kv_dim = std::max(max_kv_dim, ls.num_key_value_heads * ls.head_dim);
    }

    wb.hidden.ensure(static_cast<std::size_t>(T) * H);
    wb.ple.ensure(static_cast<std::size_t>(T) * L * H);
    wb.norm_buf.ensure(static_cast<std::size_t>(T) * H);
    wb.sub_out.ensure(static_cast<std::size_t>(T) * H);
    wb.q_buf.ensure(static_cast<std::size_t>(T) * max_q_dim);
    wb.k_buf.ensure(static_cast<std::size_t>(T) * max_kv_dim);
    wb.v_buf.ensure(static_cast<std::size_t>(T) * max_kv_dim);
    wb.attn_ctx.ensure(static_cast<std::size_t>(T) * max_q_dim);
    wb.gate_buf.ensure(static_cast<std::size_t>(T) * I);
    wb.up_buf.ensure(static_cast<std::size_t>(T) * I);
    wb.ple_gate.ensure(static_cast<std::size_t>(T) * H);

    // 1. Scaled word embeddings
    const float embed_scale =
        use_bf16 ? act::host_bf16_bits_to_float(act::host_float_to_bf16_bits(std::sqrt(static_cast<float>(H))))
                 : std::sqrt(static_cast<float>(H));
    {
      const dim3 block(256);
      const dim3 grid(static_cast<unsigned>((H + 255) / 256), static_cast<unsigned>(T));
      embed_lookup_scaled_kernel<ElemT>
          <<<grid, block, 0, stream>>>(wb.embed_tokens.ptr, d_tokens.ptr, T, H, embed_scale, wb.hidden.ptr);
    }

    // 1b. Scatter multimodal vision & audio soft tokens onto placeholders BEFORE PLE.
    if (d_vision_soft != nullptr && n_vision_soft > 0) {
      const dim3 block(256);
      const dim3 grid(static_cast<unsigned>((H + 255) / 256), static_cast<unsigned>(n_vision_soft));
      scatter_rows_kernel<ElemT><<<grid, block, 0, stream>>>(wb.hidden.ptr, d_vision_soft, d_vis_scatter_dst.ptr,
                                                             n_vision_soft, H);
    }
    if (d_audio_soft != nullptr && n_audio_soft > 0) {
      const dim3 block(256);
      const dim3 grid(static_cast<unsigned>((H + 255) / 256), static_cast<unsigned>(n_audio_soft));
      scatter_rows_kernel<ElemT><<<grid, block, 0, stream>>>(wb.hidden.ptr, d_audio_soft, d_aud_scatter_dst.ptr,
                                                             n_audio_soft, H);
    }

    // 2. Projection-only PLE: [T, H] -> [T, L * H] scaled by H^-0.5 -> RMSNorm per [H]
    const float ple_scale = 1.0f / std::sqrt(static_cast<float>(H));
    gemm_rowmajor(wb.hidden.ptr, wb.per_layer_model_projection.ptr, wb.ple.ptr, T, L * H, H, ple_scale);
    rms_norm_kernel<ElemT>
        <<<static_cast<unsigned>(T * L), 256, 0, stream>>>(wb.ple.ptr, wb.per_layer_projection_norm.ptr, T * L, H,
                                                           cfg.rms_norm_eps, wb.ple.ptr);

    // 3. 24 bidirectional encoder layers
    for (int li = 0; li <= stop_layer; ++li) {
      const LayerSpec& ls = cfg.layers[static_cast<std::size_t>(li)];
      const auto& lw = wb.layers[static_cast<std::size_t>(li)];
      const int n_heads = ls.num_attention_heads;
      const int n_kv = ls.num_key_value_heads;
      const int hd = ls.head_dim;
      const int q_dim = n_heads * hd;
      const int kv_dim = n_kv * hd;
      const bool is_sliding = (ls.kind == LayerAttnKind::Sliding);

      // Attention sub-block
      rms_norm_kernel<ElemT>
          <<<static_cast<unsigned>(T), 256, 0, stream>>>(wb.hidden.ptr, lw.input_layernorm.ptr, T, H,
                                                         cfg.rms_norm_eps, wb.norm_buf.ptr);
      gemm_rowmajor(wb.norm_buf.ptr, lw.q_proj.ptr, wb.q_buf.ptr, T, q_dim, H);
      gemm_rowmajor(wb.norm_buf.ptr, lw.k_proj.ptr, wb.k_buf.ptr, T, kv_dim, H);
      gemm_rowmajor(wb.norm_buf.ptr, lw.v_proj.ptr, wb.v_buf.ptr, T, kv_dim, H);

      head_rmsnorm_rope_kernel<ElemT><<<static_cast<unsigned>(T * n_heads), 256, 0, stream>>>(
          wb.q_buf.ptr, lw.q_norm.ptr, d_pos_of_tok.ptr, T, n_heads, hd, cfg.rms_norm_eps, ls.rope_theta);
      head_rmsnorm_rope_kernel<ElemT><<<static_cast<unsigned>(T * n_kv), 256, 0, stream>>>(
          wb.k_buf.ptr, lw.k_norm.ptr, d_pos_of_tok.ptr, T, n_kv, hd, cfg.rms_norm_eps, ls.rope_theta);
      rms_norm_kernel<ElemT><<<static_cast<unsigned>(T * n_kv), 256, 0, stream>>>(
          wb.v_buf.ptr, nullptr, T * n_kv, hd, cfg.rms_norm_eps, wb.v_buf.ptr);

      {
        const dim3 attn_grid(static_cast<unsigned>(T), static_cast<unsigned>(n_heads));
        bidir_attention_kernel<ElemT><<<attn_grid, 128, 0, stream>>>(
            wb.q_buf.ptr, wb.k_buf.ptr, wb.v_buf.ptr, d_offsets.ptr, d_seq_of_tok.ptr, T, n_heads, n_kv, hd,
            cfg.sliding_window, is_sliding, wb.attn_ctx.ptr);
      }

      gemm_rowmajor(wb.attn_ctx.ptr, lw.o_proj.ptr, wb.sub_out.ptr, T, H, q_dim);
      rms_norm_residual_add_kernel<ElemT><<<static_cast<unsigned>(T), 256, 0, stream>>>(
          wb.sub_out.ptr, lw.post_attention_layernorm.ptr, T, H, cfg.rms_norm_eps, wb.hidden.ptr);

      // GeGLU MLP sub-block
      rms_norm_kernel<ElemT><<<static_cast<unsigned>(T), 256, 0, stream>>>(
          wb.hidden.ptr, lw.pre_feedforward_layernorm.ptr, T, H, cfg.rms_norm_eps, wb.norm_buf.ptr);
      gemm_rowmajor(wb.norm_buf.ptr, lw.gate_proj.ptr, wb.gate_buf.ptr, T, I, H);
      gemm_rowmajor(wb.norm_buf.ptr, lw.up_proj.ptr, wb.up_buf.ptr, T, I, H);
      {
        const std::size_t n_mlp = static_cast<std::size_t>(T) * static_cast<std::size_t>(I);
        const unsigned blocks = static_cast<unsigned>((n_mlp + 255) / 256);
        geglu_mul_kernel<ElemT><<<blocks, 256, 0, stream>>>(wb.gate_buf.ptr, wb.up_buf.ptr, n_mlp);
      }
      gemm_rowmajor(wb.gate_buf.ptr, lw.down_proj.ptr, wb.sub_out.ptr, T, H, I);
      rms_norm_residual_add_kernel<ElemT><<<static_cast<unsigned>(T), 256, 0, stream>>>(
          wb.sub_out.ptr, lw.post_feedforward_layernorm.ptr, T, H, cfg.rms_norm_eps, wb.hidden.ptr);

      // PLE residual sub-block + layer_scalar
      gemm_rowmajor(wb.hidden.ptr, lw.per_layer_input_gate.ptr, wb.ple_gate.ptr, T, H, H);
      {
        const std::size_t n_ple = static_cast<std::size_t>(T) * static_cast<std::size_t>(H);
        const unsigned blocks = static_cast<unsigned>((n_ple + 255) / 256);
        ple_gate_mul_kernel<ElemT><<<blocks, 256, 0, stream>>>(wb.ple_gate.ptr, wb.ple.ptr, T, L, li, H);
      }
      gemm_rowmajor(wb.ple_gate.ptr, lw.per_layer_projection.ptr, wb.sub_out.ptr, T, H, H);
      rms_norm_residual_scale_kernel<ElemT><<<static_cast<unsigned>(T), 256, 0, stream>>>(
          wb.sub_out.ptr, lw.post_per_layer_input_norm.ptr, T, H, cfg.rms_norm_eps, lw.layer_scalar, wb.hidden.ptr);
    }

    // 4. Final RMSNorm only at full depth
    if (!req.tap_layer.has_value()) {
      rms_norm_kernel<ElemT><<<static_cast<unsigned>(T), 256, 0, stream>>>(wb.hidden.ptr, wb.norm.ptr, T, H,
                                                                           cfg.rms_norm_eps, wb.hidden.ptr);
    }

    // 5. Pool [T, H] -> [B, H] FP32
    {
      const int pmode = (req.pooling == model::EmbeddingPooling::LastToken) ? 1
                        : (req.pooling == model::EmbeddingPooling::Cls)     ? 2
                                                                            : 0;
      const dim3 block(256);
      const dim3 grid(static_cast<unsigned>((H + 255) / 256), static_cast<unsigned>(B));
      pool_to_f32_kernel<ElemT><<<grid, block, 0, stream>>>(wb.hidden.ptr, d_offsets.ptr, B, H, pmode, d_pooled_out);
    }
  }
};

Eg2CudaBackend::Eg2CudaBackend(const ingine::model::RuntimeOptions& options)
    : impl_(std::make_unique<Impl>()),
      max_ctx_(options.max_ctx > 0 ? std::min(options.max_ctx, 8192) : 8192) {
  if (!options.model_dir.filename().empty()) {
    model_id_ = options.model_dir.filename().string();
  }
  // Strict precision guard: NEVER use FP16 activations for EmbeddingGemma 2.
  // Use native BF16 GEMMs only on sm_80+ when not compiled with INGINE_ACT_FP16; otherwise native FP32.
  const int sm_major = current_sm_major();
  use_bf16_ = (sm_major >= 8) && !kActIsFP16;
  impl_->use_bf16 = use_bf16_;

  if (use_bf16_) {
    Eg2TextWeightsBf16 w = load_eg2_text_weights_bf16(options.model_dir);
    config_ = w.config;
    auto& wb = impl_->w_bf16;
    wb.embed_tokens.upload(w.embed_tokens, impl_->stream);
    wb.per_layer_model_projection.upload(w.per_layer_model_projection, impl_->stream);
    wb.per_layer_projection_norm.upload(w.per_layer_projection_norm, impl_->stream);
    wb.norm.upload(w.norm, impl_->stream);
    wb.embedding_projection.upload(w.embedding_projection, impl_->stream);
    wb.total_bytes = wb.embed_tokens.bytes() + wb.per_layer_model_projection.bytes() +
                     wb.per_layer_projection_norm.bytes() + wb.norm.bytes() + wb.embedding_projection.bytes();
    wb.layers.resize(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
      auto& dl = wb.layers[i];
      const auto& hl = w.layers[i];
      dl.input_layernorm.upload(hl.input_layernorm, impl_->stream);
      dl.q_proj.upload(hl.q_proj, impl_->stream);
      dl.k_proj.upload(hl.k_proj, impl_->stream);
      dl.v_proj.upload(hl.v_proj, impl_->stream);
      dl.o_proj.upload(hl.o_proj, impl_->stream);
      dl.q_norm.upload(hl.q_norm, impl_->stream);
      dl.k_norm.upload(hl.k_norm, impl_->stream);
      dl.post_attention_layernorm.upload(hl.post_attention_layernorm, impl_->stream);
      dl.pre_feedforward_layernorm.upload(hl.pre_feedforward_layernorm, impl_->stream);
      dl.gate_proj.upload(hl.gate_proj, impl_->stream);
      dl.up_proj.upload(hl.up_proj, impl_->stream);
      dl.down_proj.upload(hl.down_proj, impl_->stream);
      dl.post_feedforward_layernorm.upload(hl.post_feedforward_layernorm, impl_->stream);
      dl.per_layer_input_gate.upload(hl.per_layer_input_gate, impl_->stream);
      dl.per_layer_projection.upload(hl.per_layer_projection, impl_->stream);
      dl.post_per_layer_input_norm.upload(hl.post_per_layer_input_norm, impl_->stream);
      dl.layer_scalar = hl.layer_scalar;
      wb.total_bytes += dl.input_layernorm.bytes() + dl.q_proj.bytes() + dl.k_proj.bytes() + dl.v_proj.bytes() +
                        dl.o_proj.bytes() + dl.q_norm.bytes() + dl.k_norm.bytes() +
                        dl.post_attention_layernorm.bytes() + dl.pre_feedforward_layernorm.bytes() +
                        dl.gate_proj.bytes() + dl.up_proj.bytes() + dl.down_proj.bytes() +
                        dl.post_feedforward_layernorm.bytes() + dl.per_layer_input_gate.bytes() +
                        dl.per_layer_projection.bytes() + dl.post_per_layer_input_norm.bytes();
    }
  } else {
    Eg2TextWeightsF32 w = load_eg2_text_weights_f32(options.model_dir);
    config_ = w.config;
    auto& wb = impl_->w_f32;
    wb.embed_tokens.upload(w.embed_tokens, impl_->stream);
    wb.per_layer_model_projection.upload(w.per_layer_model_projection, impl_->stream);
    wb.per_layer_projection_norm.upload(w.per_layer_projection_norm, impl_->stream);
    wb.norm.upload(w.norm, impl_->stream);
    wb.embedding_projection.upload(w.embedding_projection, impl_->stream);
    wb.total_bytes = wb.embed_tokens.bytes() + wb.per_layer_model_projection.bytes() +
                     wb.per_layer_projection_norm.bytes() + wb.norm.bytes() + wb.embedding_projection.bytes();
    wb.layers.resize(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
      auto& dl = wb.layers[i];
      const auto& hl = w.layers[i];
      dl.input_layernorm.upload(hl.input_layernorm, impl_->stream);
      dl.q_proj.upload(hl.q_proj, impl_->stream);
      dl.k_proj.upload(hl.k_proj, impl_->stream);
      dl.v_proj.upload(hl.v_proj, impl_->stream);
      dl.o_proj.upload(hl.o_proj, impl_->stream);
      dl.q_norm.upload(hl.q_norm, impl_->stream);
      dl.k_norm.upload(hl.k_norm, impl_->stream);
      dl.post_attention_layernorm.upload(hl.post_attention_layernorm, impl_->stream);
      dl.pre_feedforward_layernorm.upload(hl.pre_feedforward_layernorm, impl_->stream);
      dl.gate_proj.upload(hl.gate_proj, impl_->stream);
      dl.up_proj.upload(hl.up_proj, impl_->stream);
      dl.down_proj.upload(hl.down_proj, impl_->stream);
      dl.post_feedforward_layernorm.upload(hl.post_feedforward_layernorm, impl_->stream);
      dl.per_layer_input_gate.upload(hl.per_layer_input_gate, impl_->stream);
      dl.per_layer_projection.upload(hl.per_layer_projection, impl_->stream);
      dl.post_per_layer_input_norm.upload(hl.post_per_layer_input_norm, impl_->stream);
      dl.layer_scalar = hl.layer_scalar;
      wb.total_bytes += dl.input_layernorm.bytes() + dl.q_proj.bytes() + dl.k_proj.bytes() + dl.v_proj.bytes() +
                        dl.o_proj.bytes() + dl.q_norm.bytes() + dl.k_norm.bytes() +
                        dl.post_attention_layernorm.bytes() + dl.pre_feedforward_layernorm.bytes() +
                        dl.gate_proj.bytes() + dl.up_proj.bytes() + dl.down_proj.bytes() +
                        dl.post_feedforward_layernorm.bytes() + dl.per_layer_input_gate.bytes() +
                        dl.per_layer_projection.bytes() + dl.post_per_layer_input_norm.bytes();
    }
  }
  if (options.vision && config_.has_vision_config) {
    if (use_bf16_) {
      Eg2VisionWeightsBf16 vw_host = load_eg2_vision_weights_bf16(options.model_dir);
      auto& vw = impl_->v_bf16;
      vw.patch_input_proj.upload(vw_host.patch_input_proj, impl_->stream);
      vw.position_embedding_table.upload(vw_host.position_embedding_table, impl_->stream);
      vw.embed_vision_projection.upload(vw_host.embed_vision_projection, impl_->stream);
      vw.total_bytes = vw.patch_input_proj.bytes() + vw.position_embedding_table.bytes() +
                       vw.embed_vision_projection.bytes();
      vw.layers.resize(vw_host.layers.size());
      for (std::size_t i = 0; i < vw_host.layers.size(); ++i) {
        auto& dl = vw.layers[i];
        const auto& hl = vw_host.layers[i];
        dl.input_layernorm.upload(hl.input_layernorm, impl_->stream);
        dl.q_proj.upload(hl.q_proj, impl_->stream);
        dl.k_proj.upload(hl.k_proj, impl_->stream);
        dl.v_proj.upload(hl.v_proj, impl_->stream);
        dl.o_proj.upload(hl.o_proj, impl_->stream);
        dl.q_norm.upload(hl.q_norm, impl_->stream);
        dl.k_norm.upload(hl.k_norm, impl_->stream);
        dl.post_attention_layernorm.upload(hl.post_attention_layernorm, impl_->stream);
        dl.pre_feedforward_layernorm.upload(hl.pre_feedforward_layernorm, impl_->stream);
        dl.gate_proj.upload(hl.gate_proj, impl_->stream);
        dl.up_proj.upload(hl.up_proj, impl_->stream);
        dl.down_proj.upload(hl.down_proj, impl_->stream);
        dl.post_feedforward_layernorm.upload(hl.post_feedforward_layernorm, impl_->stream);
        vw.total_bytes += dl.input_layernorm.bytes() + dl.q_proj.bytes() + dl.k_proj.bytes() + dl.v_proj.bytes() +
                          dl.o_proj.bytes() + dl.q_norm.bytes() + dl.k_norm.bytes() +
                          dl.post_attention_layernorm.bytes() + dl.pre_feedforward_layernorm.bytes() +
                          dl.gate_proj.bytes() + dl.up_proj.bytes() + dl.down_proj.bytes() +
                          dl.post_feedforward_layernorm.bytes();
      }
    } else {
      Eg2VisionWeightsF32 vw_host = load_eg2_vision_weights_f32(options.model_dir);
      auto& vw = impl_->v_f32;
      vw.patch_input_proj.upload(vw_host.patch_input_proj, impl_->stream);
      vw.position_embedding_table.upload(vw_host.position_embedding_table, impl_->stream);
      vw.embed_vision_projection.upload(vw_host.embed_vision_projection, impl_->stream);
      vw.total_bytes = vw.patch_input_proj.bytes() + vw.position_embedding_table.bytes() +
                       vw.embed_vision_projection.bytes();
      vw.layers.resize(vw_host.layers.size());
      for (std::size_t i = 0; i < vw_host.layers.size(); ++i) {
        auto& dl = vw.layers[i];
        const auto& hl = vw_host.layers[i];
        dl.input_layernorm.upload(hl.input_layernorm, impl_->stream);
        dl.q_proj.upload(hl.q_proj, impl_->stream);
        dl.k_proj.upload(hl.k_proj, impl_->stream);
        dl.v_proj.upload(hl.v_proj, impl_->stream);
        dl.o_proj.upload(hl.o_proj, impl_->stream);
        dl.q_norm.upload(hl.q_norm, impl_->stream);
        dl.k_norm.upload(hl.k_norm, impl_->stream);
        dl.post_attention_layernorm.upload(hl.post_attention_layernorm, impl_->stream);
        dl.pre_feedforward_layernorm.upload(hl.pre_feedforward_layernorm, impl_->stream);
        dl.gate_proj.upload(hl.gate_proj, impl_->stream);
        dl.up_proj.upload(hl.up_proj, impl_->stream);
        dl.down_proj.upload(hl.down_proj, impl_->stream);
        dl.post_feedforward_layernorm.upload(hl.post_feedforward_layernorm, impl_->stream);
        vw.total_bytes += dl.input_layernorm.bytes() + dl.q_proj.bytes() + dl.k_proj.bytes() + dl.v_proj.bytes() +
                          dl.o_proj.bytes() + dl.q_norm.bytes() + dl.k_norm.bytes() +
                          dl.post_attention_layernorm.bytes() + dl.pre_feedforward_layernorm.bytes() +
                          dl.gate_proj.bytes() + dl.up_proj.bytes() + dl.down_proj.bytes() +
                          dl.post_feedforward_layernorm.bytes();
      }
    }
    has_vision_ = true;
  }
  if (options.audio && config_.has_audio_config) {
    const std::vector<float> rel_pos_f32 = eg2_build_rel_pos_embed_f32(config_.audio);
    auto upload_audio_bank = [&](auto& aw, const auto& aw_host) {
      aw.sscp_layer0_conv.upload(aw_host.sscp_layer0_conv, impl_->stream);
      aw.sscp_layer0_norm.upload(aw_host.sscp_layer0_norm, impl_->stream);
      aw.sscp_layer1_conv.upload(aw_host.sscp_layer1_conv, impl_->stream);
      aw.sscp_layer1_norm.upload(aw_host.sscp_layer1_norm, impl_->stream);
      aw.sscp_input_proj_linear.upload(aw_host.sscp_input_proj_linear, impl_->stream);
      aw.output_proj_weight.upload(aw_host.output_proj_weight, impl_->stream);
      aw.output_proj_bias.upload(aw_host.output_proj_bias, impl_->stream);
      aw.embed_audio_projection.upload(aw_host.embed_audio_projection, impl_->stream);
      aw.total_bytes = aw.sscp_layer0_conv.bytes() + aw.sscp_layer0_norm.bytes() + aw.sscp_layer1_conv.bytes() +
                       aw.sscp_layer1_norm.bytes() + aw.sscp_input_proj_linear.bytes() +
                       aw.output_proj_weight.bytes() + aw.output_proj_bias.bytes() +
                       aw.embed_audio_projection.bytes();
      aw.layers.resize(aw_host.layers.size());
      for (std::size_t i = 0; i < aw_host.layers.size(); ++i) {
        auto& dl = aw.layers[i];
        const auto& hl = aw_host.layers[i];
        dl.feed_forward1.pre_layer_norm.upload(hl.feed_forward1.pre_layer_norm, impl_->stream);
        aw.total_bytes += dl.feed_forward1.pre_layer_norm.bytes();
        aw.total_bytes += dl.feed_forward1.ffw_layer_1.upload(hl.feed_forward1.ffw_layer_1, impl_->stream);
        aw.total_bytes += dl.feed_forward1.ffw_layer_2.upload(hl.feed_forward1.ffw_layer_2, impl_->stream);
        dl.feed_forward1.post_layer_norm.upload(hl.feed_forward1.post_layer_norm, impl_->stream);
        aw.total_bytes += dl.feed_forward1.post_layer_norm.bytes();

        dl.norm_pre_attn.upload(hl.norm_pre_attn, impl_->stream);
        aw.total_bytes += dl.norm_pre_attn.bytes();
        aw.total_bytes += dl.q_proj.upload(hl.q_proj, impl_->stream);
        aw.total_bytes += dl.k_proj.upload(hl.k_proj, impl_->stream);
        aw.total_bytes += dl.v_proj.upload(hl.v_proj, impl_->stream);
        dl.relative_k_proj.upload(hl.relative_k_proj, impl_->stream);
        aw.total_bytes += dl.relative_k_proj.bytes();
        dl.per_dim_scale.upload(hl.per_dim_scale, impl_->stream);
        aw.total_bytes += dl.per_dim_scale.bytes();
        aw.total_bytes += dl.post.upload(hl.post, impl_->stream);
        dl.norm_post_attn.upload(hl.norm_post_attn, impl_->stream);
        aw.total_bytes += dl.norm_post_attn.bytes();

        dl.lconv_pre_layer_norm.upload(hl.lconv_pre_layer_norm, impl_->stream);
        aw.total_bytes += dl.lconv_pre_layer_norm.bytes();
        aw.total_bytes += dl.lconv_linear_start.upload(hl.lconv_linear_start, impl_->stream);
        dl.lconv_depthwise_conv1d.upload(hl.lconv_depthwise_conv1d, impl_->stream);
        aw.total_bytes += dl.lconv_depthwise_conv1d.bytes();
        dl.lconv_conv_norm.upload(hl.lconv_conv_norm, impl_->stream);
        aw.total_bytes += dl.lconv_conv_norm.bytes();
        aw.total_bytes += dl.lconv_linear_end.upload(hl.lconv_linear_end, impl_->stream);

        dl.feed_forward2.pre_layer_norm.upload(hl.feed_forward2.pre_layer_norm, impl_->stream);
        aw.total_bytes += dl.feed_forward2.pre_layer_norm.bytes();
        aw.total_bytes += dl.feed_forward2.ffw_layer_1.upload(hl.feed_forward2.ffw_layer_1, impl_->stream);
        aw.total_bytes += dl.feed_forward2.ffw_layer_2.upload(hl.feed_forward2.ffw_layer_2, impl_->stream);
        dl.feed_forward2.post_layer_norm.upload(hl.feed_forward2.post_layer_norm, impl_->stream);
        aw.total_bytes += dl.feed_forward2.post_layer_norm.bytes();

        dl.norm_out.upload(hl.norm_out, impl_->stream);
        aw.total_bytes += dl.norm_out.bytes();
      }
    };
    if (use_bf16_) {
      Eg2AudioWeightsBf16 aw_host = load_eg2_audio_weights_bf16(options.model_dir);
      upload_audio_bank(impl_->a_bf16, aw_host);
      std::vector<std::uint16_t> rel_bf16(rel_pos_f32.size());
      for (std::size_t i = 0; i < rel_pos_f32.size(); ++i) {
        rel_bf16[i] = act::host_float_to_bf16_bits(rel_pos_f32[i]);
      }
      impl_->a_bf16.rel_pos_embed.upload(rel_bf16, impl_->stream);
    } else {
      Eg2AudioWeightsF32 aw_host = load_eg2_audio_weights_f32(options.model_dir);
      upload_audio_bank(impl_->a_f32, aw_host);
      impl_->a_f32.rel_pos_embed.upload(rel_pos_f32, impl_->stream);
    }
    has_audio_ = true;
  }
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  if (!options.platos_key.empty()) {
    platos_key_ = PlatosKey::load(options.platos_key, options.platos_key_metadata, 0);
    platos_key_->validate_binding(model_id_, model_family(), config_.embedding_dim, config_.num_hidden_layers,
                                  config_.hidden_size);
    const PlatosKeyPrecision pk_prec = use_bf16_ ? PlatosKeyPrecision::Bf16 : PlatosKeyPrecision::Fp32;
    impl_->platos_key_gpu = PlatosKeyGpu::create(platos_key_, impl_->stream, pk_prec);
  }

  caps_.family = model::ModelFamily::EmbeddingGemma2;
  caps_.backend_name = use_bf16_ ? "embeddinggemma2-cuda-bf16" : "embeddinggemma2-cuda-f32";
  caps_.architecture = !config_.architectures.empty() ? config_.architectures.front() : "EmbeddingGemma2Model";
  caps_.supports_text = true;
  caps_.supports_images = has_vision_;
  caps_.supports_audio = has_audio_;
  caps_.max_supported_batch = 64;
  caps_.supported_gears = {use_bf16_ ? "encoder_bf16" : "encoder_f32"};
  caps_.supported_kv_cache_modes = {"none"};
  caps_.supported_quant_layouts = {use_bf16_ ? "bf16" : "bf16_to_f32"};
  caps_.media_token_ids = {config_.boi_token_id, config_.boa_token_id, config_.image_token_id,
                           config_.audio_token_id, config_.eoi_token_id, config_.eoa_token_id, config_.video_token_id};
  caps_.vocab_size = config_.vocab_size;
  caps_.max_position_embeddings = max_ctx_;

  profile_.family = caps_.family;
  profile_.model_id = model_id_;
  profile_.gear = caps_.supported_gears.front();
  profile_.kv_cache_mode = "none";
  profile_.tokenizer_format = "gemma_bpe";
  profile_.chat_template_family = "embedding_gemma2";
  profile_.max_ctx = max_ctx_;
  profile_.max_batch = 64;
  profile_.capabilities = caps_;

  identity_.family = caps_.family;
  identity_.backend_name = caps_.backend_name;
  identity_.state_layout_id = "eg2_stateless_encoder_v1";
  identity_.act_dtype = use_bf16_ ? model::ActivationDtype::Bf16 : model::ActivationDtype::Fp32;
  identity_.numeric_abi = use_bf16_ ? "act=bf16;gemm_acc=f32;attn=bidir_swa_mqa_f32"
                                    : "act=f32;weights=bf16_to_f32;attn=bidir_swa_mqa_f32";
  identity_.weight_precision = use_bf16_ ? "bf16" : "f32";
  identity_.kv_format = "none";
  identity_.kv_cache_mode = "none";
  identity_.gear = profile_.gear;
  identity_.max_ctx = max_ctx_;
  identity_.prefill_chunk = max_ctx_;
}

Eg2CudaBackend::~Eg2CudaBackend() = default;

ingine::model::GenerateStats Eg2CudaBackend::generate(std::span<const std::int32_t>, int,
                                                      const ingine::model::Sampling&, std::span<const std::int32_t>,
                                                      const std::function<bool(std::span<const std::int32_t>)>&) {
  throw std::invalid_argument("EmbeddingGemma2 is an embeddings-only encoder; use POST /v1/embeddings");
}

std::optional<PlatosKeyPrecision> Eg2CudaBackend::platos_key_precision() const noexcept {
  if (!impl_ || !impl_->platos_key_gpu) return std::nullopt;
  return impl_->platos_key_gpu->precision();
}

ingine::model::ResidencyReport Eg2CudaBackend::residency_report() const {
  model::ResidencyReport r;
  const std::uint64_t text_bytes = use_bf16_ ? impl_->w_bf16.total_bytes : impl_->w_f32.total_bytes;
  const std::uint64_t vis_bytes = has_vision_ ? (use_bf16_ ? impl_->v_bf16.total_bytes : impl_->v_f32.total_bytes) : 0;
  const std::uint64_t aud_bytes = has_audio_ ? (use_bf16_ ? impl_->a_bf16.total_bytes : impl_->a_f32.total_bytes) : 0;
  const std::uint64_t pk_bytes = impl_->platos_key_gpu ? impl_->platos_key_gpu->device_bytes() : 0;
  const std::uint64_t wbytes = text_bytes + vis_bytes + aud_bytes;
  r.weights_device_bytes = wbytes;
  r.device_bytes = wbytes + pk_bytes + impl_->lt_workspace.bytes();
  return r;
}

ingine::model::EmbeddingResult Eg2CudaBackend::embed(const ingine::model::EmbeddingRequest& req) {
  if (req.inputs.empty()) throw std::invalid_argument("Eg2CudaBackend::embed: inputs must not be empty");
  if (!req.images.empty() && req.images.size() != req.inputs.size()) {
    throw std::invalid_argument("Eg2CudaBackend::embed: images batch size must match inputs batch size");
  }
  if (!req.audios.empty() && req.audios.size() != req.inputs.size()) {
    throw std::invalid_argument("Eg2CudaBackend::embed: audios batch size must match inputs batch size");
  }
  if (req.tap_layer.has_value() && (*req.tap_layer < 0 || *req.tap_layer >= config_.num_hidden_layers)) {
    throw std::invalid_argument("Eg2CudaBackend::embed: tap_layer out of range");
  }
  if (req.use_platos_key && !impl_->platos_key_gpu) {
    throw std::invalid_argument("Eg2CudaBackend::embed: platos_key requested but none loaded");
  }

  const int B = static_cast<int>(req.inputs.size());
  const int H = config_.hidden_size;
  const int E = config_.embedding_dim;
  const bool full_depth = !req.tap_layer.has_value();
  const int cur_dim = full_depth ? E : H;

  if (req.use_platos_key) {
    if (req.dimensions.has_value()) {
      throw std::invalid_argument("EmbeddingGemma2: 'dimensions' cannot be combined with 'platos_key'");
    }
    platos_key_->validate_request_tap(req.tap_layer);
    platos_key_->validate_request_identity(req.cartridge_id.value_or(""), req.cartridge_sha256.value_or(""));
    if (platos_key_->in_dim() != cur_dim) {
      throw std::invalid_argument("EmbeddingGemma2: platos_key d_in (" + std::to_string(platos_key_->in_dim()) +
                                  ") does not match embedding input dim (" + std::to_string(cur_dim) + ")");
    }
  } else {
    validate_dimensions(config_, req.dimensions, cur_dim);
  }

  std::vector<std::int32_t> h_tokens;
  std::vector<std::int32_t> h_offsets(static_cast<std::size_t>(B + 1), 0);
  std::vector<std::int32_t> h_seq_of_tok;
  std::vector<std::int32_t> h_pos_of_tok;
  std::vector<std::int32_t> h_vis_scatter_dst;
  std::vector<std::int32_t> h_aud_scatter_dst;
  std::vector<Eg2PreprocessedImage> all_preps;
  std::vector<Eg2PreprocessedAudio> all_audio_preps;
  int total_vis_soft = 0;
  int total_aud_soft = 0;

  for (int b = 0; b < B; ++b) {
    const auto& seq = req.inputs[static_cast<std::size_t>(b)];
    if (seq.empty()) throw std::invalid_argument("Eg2CudaBackend::embed: input sequence must not be empty");
    if (static_cast<int>(seq.size()) > max_ctx_) {
      throw std::invalid_argument("EmbeddingGemma2: input length " + std::to_string(seq.size()) +
                                  " exceeds max_ctx " + std::to_string(max_ctx_));
    }
    int item_vis_placeholders = 0;
    int item_aud_placeholders = 0;
    for (int p = 0; p < static_cast<int>(seq.size()); ++p) {
      const std::int32_t tid = seq[static_cast<std::size_t>(p)];
      if (tid < 0 || tid >= config_.vocab_size) {
        throw std::invalid_argument("Eg2CudaBackend::embed: token_id " + std::to_string(tid) +
                                    " out of vocab range [0, " + std::to_string(config_.vocab_size - 1) + "]");
      }
      if (tid == config_.image_token_id || tid == config_.video_token_id) {
        h_vis_scatter_dst.push_back(static_cast<std::int32_t>(h_tokens.size()));
        h_tokens.push_back(config_.pad_token_id);  // placeholder replaced before PLE
        ++item_vis_placeholders;
      } else if (tid == config_.audio_token_id) {
        h_aud_scatter_dst.push_back(static_cast<std::int32_t>(h_tokens.size()));
        h_tokens.push_back(config_.pad_token_id);  // placeholder replaced before PLE
        ++item_aud_placeholders;
      } else {
        h_tokens.push_back(tid);
      }
      h_seq_of_tok.push_back(b);
      h_pos_of_tok.push_back(p);
    }
    int item_vis_soft = 0;
    if (!req.images.empty() && !req.images[static_cast<std::size_t>(b)].empty()) {
      if (!has_vision_) {
        throw std::invalid_argument("EmbeddingGemma2: image/video input rejected because vision tower is disabled (--no-vision)");
      }
      for (const auto& img : req.images[static_cast<std::size_t>(b)]) {
        Eg2PreprocessedImage prep = eg2_preprocess_image(img, config_.vision);
        item_vis_soft += prep.size.num_soft_tokens;
        all_preps.push_back(std::move(prep));
      }
    }
    if (item_vis_placeholders != item_vis_soft) {
      throw std::invalid_argument("Eg2CudaBackend::embed: item " + std::to_string(b) + " image/video placeholders (" +
                                  std::to_string(item_vis_placeholders) + ") != vision soft tokens (" +
                                  std::to_string(item_vis_soft) + ")");
    }
    int item_aud_soft = 0;
    if (!req.audios.empty() && !req.audios[static_cast<std::size_t>(b)].empty()) {
      if (!has_audio_) {
        throw std::invalid_argument("EmbeddingGemma2: audio input rejected because audio tower is disabled (--no-audio)");
      }
      for (const auto& aud : req.audios[static_cast<std::size_t>(b)]) {
        Eg2PreprocessedAudio prep = eg2_preprocess_audio(aud);
        item_aud_soft += prep.num_valid_tokens;
        all_audio_preps.push_back(std::move(prep));
      }
    }
    if (item_aud_placeholders != item_aud_soft) {
      throw std::invalid_argument("Eg2CudaBackend::embed: item " + std::to_string(b) + " audio placeholders (" +
                                  std::to_string(item_aud_placeholders) + ") != audio soft tokens (" +
                                  std::to_string(item_aud_soft) + ")");
    }
    total_vis_soft += item_vis_soft;
    total_aud_soft += item_aud_soft;
    h_offsets[static_cast<std::size_t>(b + 1)] = static_cast<std::int32_t>(h_tokens.size());
  }
  const int T = static_cast<int>(h_tokens.size());

  std::lock_guard lock(impl_->mu);
  CUDA_CHECK(cudaSetDevice(impl_->device));

  const auto t0 = std::chrono::steady_clock::now();
  impl_->d_tokens.upload(h_tokens, impl_->stream);
  impl_->d_offsets.upload(h_offsets, impl_->stream);
  impl_->d_seq_of_tok.upload(h_seq_of_tok, impl_->stream);
  impl_->d_pos_of_tok.upload(h_pos_of_tok, impl_->stream);
  if (total_vis_soft > 0) {
    impl_->d_vis_scatter_dst.upload(h_vis_scatter_dst, impl_->stream);
  }
  if (total_aud_soft > 0) {
    impl_->d_aud_scatter_dst.upload(h_aud_scatter_dst, impl_->stream);
  }
  impl_->d_pooled.ensure(static_cast<std::size_t>(B) * H);

  if (use_bf16_) {
    if (total_vis_soft > 0) {
      impl_->encode_vision_batch_typed<std::uint16_t>(config_, impl_->v_bf16, all_preps, total_vis_soft);
    }
    if (total_aud_soft > 0) {
      impl_->encode_audio_batch_typed<std::uint16_t>(config_, impl_->a_bf16, all_audio_preps, total_aud_soft);
    }
    impl_->run_forward_typed<std::uint16_t>(config_, impl_->w_bf16, B, T, req, impl_->d_pooled.ptr,
                                            total_vis_soft > 0 ? impl_->v_bf16.soft_out.ptr : nullptr, total_vis_soft,
                                            total_aud_soft > 0 ? impl_->a_bf16.soft_out.ptr : nullptr, total_aud_soft);
  } else {
    if (total_vis_soft > 0) {
      impl_->encode_vision_batch_typed<float>(config_, impl_->v_f32, all_preps, total_vis_soft);
    }
    if (total_aud_soft > 0) {
      impl_->encode_audio_batch_typed<float>(config_, impl_->a_f32, all_audio_preps, total_aud_soft);
    }
    impl_->run_forward_typed<float>(config_, impl_->w_f32, B, T, req, impl_->d_pooled.ptr,
                                    total_vis_soft > 0 ? impl_->v_f32.soft_out.ptr : nullptr, total_vis_soft,
                                    total_aud_soft > 0 ? impl_->a_f32.soft_out.ptr : nullptr, total_aud_soft);
  }

  float* d_cur = impl_->d_pooled.ptr;
  if (full_depth) {
    impl_->d_projected.ensure(static_cast<std::size_t>(B) * E);
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((E + 255) / 256), static_cast<unsigned>(B));
    if (use_bf16_) {
      project_pooled_to_f32_kernel<std::uint16_t><<<grid, block, 0, impl_->stream>>>(
          impl_->d_pooled.ptr, impl_->w_bf16.embedding_projection.ptr, B, H, E, impl_->d_projected.ptr);
    } else {
      project_pooled_to_f32_kernel<float><<<grid, block, 0, impl_->stream>>>(
          impl_->d_pooled.ptr, impl_->w_f32.embedding_projection.ptr, B, H, E, impl_->d_projected.ptr);
    }
    d_cur = impl_->d_projected.ptr;
  }
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  const auto t1 = std::chrono::steady_clock::now();

  double pk_seconds = 0.0;
  int out_dim = cur_dim;
  if (req.use_platos_key) {
    const auto tp0 = std::chrono::steady_clock::now();
    out_dim = platos_key_->out_dim();
    impl_->d_final.ensure(static_cast<std::size_t>(B) * out_dim);
    impl_->platos_key_gpu->project(d_cur, B, impl_->d_final.ptr, req.normalize, impl_->stream);
    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    pk_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
  } else {
    out_dim = req.dimensions.value_or(cur_dim);
    impl_->d_final.ensure(static_cast<std::size_t>(B) * out_dim);
    truncate_and_l2_norm_f32_kernel<<<static_cast<unsigned>(B), 256, 0, impl_->stream>>>(
        d_cur, B, cur_dim, out_dim, req.normalize, impl_->d_final.ptr);
    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  }

  std::vector<float> h_final(static_cast<std::size_t>(B) * static_cast<std::size_t>(out_dim));
  CUDA_CHECK(cudaMemcpyAsync(h_final.data(), impl_->d_final.ptr, h_final.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  model::EmbeddingResult res;
  res.embeddings.resize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    const float* row = h_final.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(out_dim);
    res.embeddings[static_cast<std::size_t>(b)].assign(row, row + out_dim);
  }
  res.stats.total_prompt_tokens = T;
  res.stats.prefill_seconds = std::chrono::duration<double>(t1 - t0).count();
  res.stats.platos_key_seconds = pk_seconds;
  res.stats.effective_tap_layer = req.tap_layer.value_or(-1);
  res.stats.output_dim = out_dim;
  res.stats.kv_slots_allocated = 0;
  res.stats.lm_head_skipped = true;
  return res;
}

void register_eg2_cuda_backend() {
  auto& reg = model::BackendRegistry::instance();
  if (reg.find(model::ModelFamily::EmbeddingGemma2).has_value()) return;
  reg.register_backend(model::BackendRegistration{
      model::ModelFamily::EmbeddingGemma2,
      "embeddinggemma2-cuda",
      [](const model::RuntimeOptions& options) -> std::unique_ptr<model::ModelBackend> {
        return std::make_unique<Eg2CudaBackend>(options);
      },
  });
}

}  // namespace ingine::backends::embedding_gemma2
