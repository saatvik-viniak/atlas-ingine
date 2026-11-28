#include "backends/embedding_gemma2/eg2_cuda.h"

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

  int device = 0;
  cudaStream_t stream = nullptr;
  cublasLtHandle_t lt = nullptr;
  DevBuf<std::uint8_t> lt_workspace;
  bool use_bf16 = false;
  std::mutex mu;

  WeightBank<std::uint16_t> w_bf16;
  WeightBank<float> w_f32;

  DevBuf<std::int32_t> d_tokens;
  DevBuf<std::int32_t> d_offsets;
  DevBuf<std::int32_t> d_seq_of_tok;
  DevBuf<std::int32_t> d_pos_of_tok;
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

  // Row-major GEMM: Y[M, N] = alpha * X[M, K] @ W[N, K]^T (zero FP16 — strictly BF16 or FP32).
  void gemm_rowmajor(const void* x, const void* w, void* y, int M, int N, int K, float alpha = 1.0f) {
    const cudaDataType_t dtype = use_bf16 ? CUDA_R_16BF : CUDA_R_32F;
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
      cublas_check(cublasLtMatrixLayoutCreate(&a_desc, dtype, static_cast<std::uint64_t>(K),
                                              static_cast<std::uint64_t>(N), static_cast<std::int64_t>(K)),
                   "LayoutA");
      cublas_check(cublasLtMatrixLayoutCreate(&b_desc, dtype, static_cast<std::uint64_t>(K),
                                              static_cast<std::uint64_t>(M), static_cast<std::int64_t>(K)),
                   "LayoutB");
      cublas_check(cublasLtMatrixLayoutCreate(&c_desc, dtype, static_cast<std::uint64_t>(N),
                                              static_cast<std::uint64_t>(M), static_cast<std::int64_t>(N)),
                   "LayoutC");
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
                                 " N=" + std::to_string(N) + " K=" + std::to_string(K));
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

  template <typename ElemT>
  void run_forward_typed(const Eg2Config& cfg, WeightBank<ElemT>& wb, int B, int T,
                         const model::EmbeddingRequest& req, float* d_pooled_out) {
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
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  if (!options.platos_key.empty()) {
    platos_key_ = PlatosKey::load(options.platos_key, options.platos_key_metadata, 0);
    impl_->platos_key_gpu = PlatosKeyGpu::create(platos_key_, impl_->stream);
  }

  caps_.family = model::ModelFamily::EmbeddingGemma2;
  caps_.backend_name = use_bf16_ ? "embeddinggemma2-cuda-bf16" : "embeddinggemma2-cuda-f32";
  caps_.architecture = !config_.architectures.empty() ? config_.architectures.front() : "EmbeddingGemma2Model";
  caps_.supports_text = true;
  caps_.supports_images = false;
  caps_.supports_audio = false;
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

ingine::model::ResidencyReport Eg2CudaBackend::residency_report() const {
  model::ResidencyReport r;
  const std::uint64_t wbytes = use_bf16_ ? impl_->w_bf16.total_bytes : impl_->w_f32.total_bytes;
  r.weights_device_bytes = wbytes;
  r.device_bytes = wbytes + impl_->lt_workspace.bytes();
  return r;
}

ingine::model::EmbeddingResult Eg2CudaBackend::embed(const ingine::model::EmbeddingRequest& req) {
  if (req.inputs.empty()) throw std::invalid_argument("Eg2CudaBackend::embed: inputs must not be empty");
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
  for (int b = 0; b < B; ++b) {
    const auto& seq = req.inputs[static_cast<std::size_t>(b)];
    if (seq.empty()) throw std::invalid_argument("Eg2CudaBackend::embed: input sequence must not be empty");
    if (static_cast<int>(seq.size()) > max_ctx_) {
      throw std::invalid_argument("EmbeddingGemma2: input length " + std::to_string(seq.size()) +
                                  " exceeds max_ctx " + std::to_string(max_ctx_));
    }
    for (int p = 0; p < static_cast<int>(seq.size()); ++p) {
      const std::int32_t tid = seq[static_cast<std::size_t>(p)];
      if (tid < 0 || tid >= config_.vocab_size) {
        throw std::invalid_argument("Eg2CudaBackend::embed: token_id " + std::to_string(tid) +
                                    " out of vocab range [0, " + std::to_string(config_.vocab_size - 1) + "]");
      }
      h_tokens.push_back(tid);
      h_seq_of_tok.push_back(b);
      h_pos_of_tok.push_back(p);
    }
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
  impl_->d_pooled.ensure(static_cast<std::size_t>(B) * H);

  if (use_bf16_) {
    impl_->run_forward_typed<std::uint16_t>(config_, impl_->w_bf16, B, T, req, impl_->d_pooled.ptr);
  } else {
    impl_->run_forward_typed<float>(config_, impl_->w_f32, B, T, req, impl_->d_pooled.ptr);
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
