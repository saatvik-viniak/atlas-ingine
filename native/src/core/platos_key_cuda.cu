#include "ingine/platos_key.h"

#include "core/act.h"
#include "core/device.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine {
namespace {

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("PlatosKeyGpu cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
  }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ float bf16_bits_to_f32_dev(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_bits_dev(float v) {
  const std::uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<std::uint16_t>((u >> 16) | 0x40u);
  }
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float fp16_bits_to_f32_dev(std::uint16_t b) {
  return __half2float(__ushort_as_half(b));
}

// Coalesced row-wise pooling for packed sequences `[T, dim]` -> `[batch, dim]` FP32.
// pooling_mode: 0 = Mean, 1 = LastToken, 2 = Cls.
__global__ void pool_f32_kernel(const float* __restrict__ hidden, const std::int32_t* __restrict__ offsets, int batch,
                                int dim, int pooling_mode, float* __restrict__ out) {
  const int b = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch || d >= dim) return;
  const int start = offsets[b];
  const int end = offsets[b + 1];
  const int len = end - start;
  float val = 0.0f;
  if (len > 0) {
    if (pooling_mode == 1) {
      val = hidden[static_cast<std::size_t>(end - 1) * static_cast<std::size_t>(dim) + d];
    } else if (pooling_mode == 2) {
      val = hidden[static_cast<std::size_t>(start) * static_cast<std::size_t>(dim) + d];
    } else {
      float acc = 0.0f;
      for (int t = start; t < end; ++t) {
        acc += hidden[static_cast<std::size_t>(t) * static_cast<std::size_t>(dim) + d];
      }
      val = acc / static_cast<float>(len);
    }
  }
  out[static_cast<std::size_t>(b) * static_cast<std::size_t>(dim) + d] = val;
}

__global__ void pool_act_kernel(const std::uint16_t* __restrict__ hidden, bool is_fp16,
                                const std::int32_t* __restrict__ offsets, int batch, int dim, int pooling_mode,
                                float* __restrict__ out) {
  const int b = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch || d >= dim) return;
  const int start = offsets[b];
  const int end = offsets[b + 1];
  const int len = end - start;
  float val = 0.0f;
  if (len > 0) {
    if (pooling_mode == 1) {
      const std::uint16_t bits = hidden[static_cast<std::size_t>(end - 1) * static_cast<std::size_t>(dim) + d];
      val = is_fp16 ? fp16_bits_to_f32_dev(bits) : bf16_bits_to_f32_dev(bits);
    } else if (pooling_mode == 2) {
      const std::uint16_t bits = hidden[static_cast<std::size_t>(start) * static_cast<std::size_t>(dim) + d];
      val = is_fp16 ? fp16_bits_to_f32_dev(bits) : bf16_bits_to_f32_dev(bits);
    } else {
      float acc = 0.0f;
      for (int t = start; t < end; ++t) {
        const std::uint16_t bits = hidden[static_cast<std::size_t>(t) * static_cast<std::size_t>(dim) + d];
        acc += is_fp16 ? fp16_bits_to_f32_dev(bits) : bf16_bits_to_f32_dev(bits);
      }
      val = acc / static_cast<float>(len);
    }
  }
  out[static_cast<std::size_t>(b) * static_cast<std::size_t>(dim) + d] = val;
}

__global__ void l2_normalize_rows_kernel(float* __restrict__ data, int batch, int dim) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= batch) return;
  float* row = data + static_cast<std::size_t>(b) * static_cast<std::size_t>(dim);
  float local_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = row[d];
    local_sq = fmaf(v, v, local_sq);
  }
  __shared__ float smem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float wsum = warp_reduce_sum(local_sq);
  if (lane == 0) smem[warp] = wsum;
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  float total = 0.0f;
  if (warp == 0) {
    total = (lane < nwarps) ? smem[lane] : 0.0f;
    total = warp_reduce_sum(total);
    if (lane == 0) smem[0] = total;
  }
  __syncthreads();
  const float norm = sqrtf(smem[0]);
  if (norm > 1e-12f) {
    const float inv = 1.0f / norm;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      row[d] *= inv;
    }
  }
}

// Coalesced warp-per-output-row GEMV/small-batch GEMM (B <= 8) with FP32 activations and FP16 weights R^T [d_out, d_in].
// Reads R^T once per CTA across all `batch` rows in registers for bandwidth-bound <0.07 ms latency on 5120x5120.
template <int MAX_B>
__global__ void __launch_bounds__(256)
    platos_project_small_batch_f16_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean,
                                          const __half* __restrict__ rt_f16, const float* __restrict__ target_mean,
                                          float scale, int batch, int d_in, int d_out, int mask_coord,
                                          float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int j = static_cast<int>(blockIdx.x) * 8 + warp_in_block;
  if (j >= d_out) return;

  const __half* w_row = rt_f16 + static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in);
  float acc[MAX_B];
#pragma unroll
  for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

  // Vectorized 8-element (16-byte) coalesced load along d_in when every row stride is 8-aligned.
  if ((d_in & 7) == 0) {
    for (int i = lane * 8; i < d_in; i += 32 * 8) {
      const uint4 w4 = *reinterpret_cast<const uint4*>(w_row + i);
      const __half2* wh = reinterpret_cast<const __half2*>(&w4);
      float wv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f2 = __half22float2(wh[k]);
        wv[2 * k] = f2.x;
        wv[2 * k + 1] = f2.y;
      }
      const float4 dm0 = *reinterpret_cast<const float4*>(donor_mean + i);
      const float4 dm1 = *reinterpret_cast<const float4*>(donor_mean + i + 4);
      const float dm[8] = {dm0.x, dm0.y, dm0.z, dm0.w, dm1.x, dm1.y, dm1.z, dm1.w};
      const bool has_mask = (mask_coord >= i && mask_coord < i + 8);
      const int mask_lane = has_mask ? (mask_coord - i) : -1;

      for (int b = 0; b < batch; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i;
        const float4 x0 = *reinterpret_cast<const float4*>(xb);
        const float4 x1 = *reinterpret_cast<const float4*>(xb + 4);
        float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        if (has_mask) xv[mask_lane] = 0.0f;
        float s = acc[b];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          s = fmaf(xv[k] - dm[k], wv[k], s);
        }
        acc[b] = s;
      }
    }
  } else {
    for (int i = lane; i < d_in; i += 32) {
      const float wv = __half2float(w_row[i]);
      const float dmv = donor_mean[i];
      for (int b = 0; b < batch; ++b) {
        const float xv = (i == mask_coord) ? 0.0f : x[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i];
        acc[b] = fmaf(xv - dmv, wv, acc[b]);
      }
    }
  }

  const float tm = target_mean[j];
  for (int b = 0; b < batch; ++b) {
    const float sum = warp_reduce_sum(acc[b]);
    if (lane == 0) {
      out[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_out) + j] = fmaf(scale, sum, tm);
    }
  }
}

// Coalesced warp-per-output-row small-batch GEMM (B <= 8) with FP32 activations and BF16 weights R^T [d_out, d_in].
// Strictly zero FP16 instructions or casts (used by sm_80+ EmbeddingGemma 2).
template <int MAX_B>
__global__ void __launch_bounds__(256)
    platos_project_small_batch_bf16_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean,
                                           const std::uint16_t* __restrict__ rt_bf16,
                                           const float* __restrict__ target_mean, float scale, int batch, int d_in,
                                           int d_out, int mask_coord, float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int j = static_cast<int>(blockIdx.x) * 8 + warp_in_block;
  if (j >= d_out) return;

  const std::uint16_t* w_row = rt_bf16 + static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in);
  float acc[MAX_B];
#pragma unroll
  for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

  if ((d_in & 7) == 0) {
    for (int i = lane * 8; i < d_in; i += 32 * 8) {
      const uint4 w4 = *reinterpret_cast<const uint4*>(w_row + i);
      const std::uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
      float wv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        wv[2 * k] = __uint_as_float(ws[k] << 16);
        wv[2 * k + 1] = __uint_as_float(ws[k] & 0xffff0000u);
      }
      const float4 dm0 = *reinterpret_cast<const float4*>(donor_mean + i);
      const float4 dm1 = *reinterpret_cast<const float4*>(donor_mean + i + 4);
      const float dm[8] = {dm0.x, dm0.y, dm0.z, dm0.w, dm1.x, dm1.y, dm1.z, dm1.w};
      const bool has_mask = (mask_coord >= i && mask_coord < i + 8);
      const int mask_lane = has_mask ? (mask_coord - i) : -1;

      for (int b = 0; b < batch; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i;
        const float4 x0 = *reinterpret_cast<const float4*>(xb);
        const float4 x1 = *reinterpret_cast<const float4*>(xb + 4);
        float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        if (has_mask) xv[mask_lane] = 0.0f;
        float s = acc[b];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          s = fmaf(xv[k] - dm[k], wv[k], s);
        }
        acc[b] = s;
      }
    }
  } else {
    for (int i = lane; i < d_in; i += 32) {
      const float wv = bf16_bits_to_f32_dev(w_row[i]);
      const float dmv = donor_mean[i];
      for (int b = 0; b < batch; ++b) {
        const float xv = (i == mask_coord) ? 0.0f : x[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i];
        acc[b] = fmaf(xv - dmv, wv, acc[b]);
      }
    }
  }

  const float tm = target_mean[j];
  for (int b = 0; b < batch; ++b) {
    const float sum = warp_reduce_sum(acc[b]);
    if (lane == 0) {
      out[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_out) + j] = fmaf(scale, sum, tm);
    }
  }
}

// Coalesced warp-per-output-row small-batch GEMM (B <= 8) with native FP32 activations and FP32 weights R^T [d_out, d_in].
// Strictly zero FP16 instructions or casts (used by sm_70 V100 EmbeddingGemma 2).
template <int MAX_B>
__global__ void __launch_bounds__(256)
    platos_project_small_batch_f32_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean,
                                          const float* __restrict__ rt_f32, const float* __restrict__ target_mean,
                                          float scale, int batch, int d_in, int d_out, int mask_coord,
                                          float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int j = static_cast<int>(blockIdx.x) * 8 + warp_in_block;
  if (j >= d_out) return;

  const float* w_row = rt_f32 + static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in);
  float acc[MAX_B];
#pragma unroll
  for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

  if ((d_in & 3) == 0) {
    for (int i = lane * 4; i < d_in; i += 32 * 4) {
      const float4 w0 = *reinterpret_cast<const float4*>(w_row + i);
      const float wv[4] = {w0.x, w0.y, w0.z, w0.w};
      const float4 dm0 = *reinterpret_cast<const float4*>(donor_mean + i);
      const float dm[4] = {dm0.x, dm0.y, dm0.z, dm0.w};
      const bool has_mask = (mask_coord >= i && mask_coord < i + 4);
      const int mask_lane = has_mask ? (mask_coord - i) : -1;

      for (int b = 0; b < batch; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i;
        const float4 x0 = *reinterpret_cast<const float4*>(xb);
        float xv[4] = {x0.x, x0.y, x0.z, x0.w};
        if (has_mask) xv[mask_lane] = 0.0f;
        float s = acc[b];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          s = fmaf(xv[k] - dm[k], wv[k], s);
        }
        acc[b] = s;
      }
    }
  } else {
    for (int i = lane; i < d_in; i += 32) {
      const float wv = w_row[i];
      const float dmv = donor_mean[i];
      for (int b = 0; b < batch; ++b) {
        const float xv = (i == mask_coord) ? 0.0f : x[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i];
        acc[b] = fmaf(xv - dmv, wv, acc[b]);
      }
    }
  }

  const float tm = target_mean[j];
  for (int b = 0; b < batch; ++b) {
    const float sum = warp_reduce_sum(acc[b]);
    if (lane == 0) {
      out[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_out) + j] = fmaf(scale, sum, tm);
    }
  }
}

__global__ void center_to_f16_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean, int batch,
                                     int d_in, int mask_coord, __half* __restrict__ out_f16) {
  const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col = static_cast<int>(idx % static_cast<std::size_t>(d_in));
  const float xv = (col == mask_coord) ? 0.0f : x[idx];
  out_f16[idx] = __float2half_rn(xv - donor_mean[col]);
}

__global__ void center_to_bf16_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean, int batch,
                                      int d_in, int mask_coord, std::uint16_t* __restrict__ out_bf16) {
  const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col = static_cast<int>(idx % static_cast<std::size_t>(d_in));
  const float xv = (col == mask_coord) ? 0.0f : x[idx];
  out_bf16[idx] = f32_to_bf16_bits_dev(xv - donor_mean[col]);
}

__global__ void center_to_f32_kernel(const float* __restrict__ x, const float* __restrict__ donor_mean, int batch,
                                     int d_in, int mask_coord, float* __restrict__ out_f32) {
  const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col = static_cast<int>(idx % static_cast<std::size_t>(d_in));
  const float xv = (col == mask_coord) ? 0.0f : x[idx];
  out_f32[idx] = xv - donor_mean[col];
}

__global__ void scale_add_target_mean_kernel(float* __restrict__ y, const float* __restrict__ target_mean, float scale,
                                             int batch, int d_out) {
  const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_out);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col = static_cast<int>(idx % static_cast<std::size_t>(d_out));
  y[idx] = fmaf(scale, y[idx], target_mean[col]);
}

}  // namespace

struct PlatosKeyGpu::Impl {
  PlatosKeyPrecision precision = PlatosKeyPrecision::Fp16;
  int d_in = 0;
  int d_out = 0;
  int mask_coord = -1;
  float scale = 1.0f;
  __half* d_rt_f16 = nullptr;            // [d_out, d_in] FP16 (default / Inkling / Qwen)
  std::uint16_t* d_rt_bf16 = nullptr;    // [d_out, d_in] BF16 (sm_80+ EG2, zero FP16)
  float* d_rt_f32 = nullptr;             // [d_out, d_in] FP32 (sm_70 V100 EG2, zero FP16)
  float* d_donor_mean = nullptr;         // [d_in] FP32
  float* d_target_mean = nullptr;        // [d_out] FP32
  std::size_t dev_bytes = 0;

  // Lazily allocated cuBLASLt handle + cached per-batch plans for large batches (batch > 8)
  struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr;
    cublasLtMatrixLayout_t lb = nullptr;
    cublasLtMatrixLayout_t lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool has_algo = false;
  };
  mutable std::mutex cublas_mu;
  mutable cublasLtHandle_t lt_handle = nullptr;
  mutable void* lt_ws = nullptr;
  mutable std::size_t lt_ws_bytes = 0;
  mutable __half* d_x_f16 = nullptr;
  mutable std::size_t x_f16_cap = 0;
  mutable std::uint16_t* d_x_bf16 = nullptr;
  mutable std::size_t x_bf16_cap = 0;
  mutable float* d_x_f32 = nullptr;
  mutable std::size_t x_f32_cap = 0;
  mutable std::map<int, LtPlan> lt_plans;

  ~Impl() {
    for (auto& [b, p] : lt_plans) {
      if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
      if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
      if (p.la) cublasLtMatrixLayoutDestroy(p.la);
      if (p.op) cublasLtMatmulDescDestroy(p.op);
    }
    lt_plans.clear();
    if (d_rt_f16) cudaFree(d_rt_f16);
    if (d_rt_bf16) cudaFree(d_rt_bf16);
    if (d_rt_f32) cudaFree(d_rt_f32);
    if (d_donor_mean) cudaFree(d_donor_mean);
    if (d_target_mean) cudaFree(d_target_mean);
    if (d_x_f16) cudaFree(d_x_f16);
    if (d_x_bf16) cudaFree(d_x_bf16);
    if (d_x_f32) cudaFree(d_x_f32);
    if (lt_ws) cudaFree(lt_ws);
    if (lt_handle) cublasLtDestroy(lt_handle);
  }
};

PlatosKeyGpu::PlatosKeyGpu(std::shared_ptr<const PlatosKey> key, std::unique_ptr<Impl> impl)
    : key_(std::move(key)), impl_(std::move(impl)) {}

PlatosKeyGpu::~PlatosKeyGpu() = default;

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::create(std::shared_ptr<const PlatosKey> key, void* cuda_stream,
                                                         PlatosKeyPrecision precision) {
  if (!key) throw std::invalid_argument("PlatosKeyGpu::create: null PlatosKey");
  if (precision == PlatosKeyPrecision::Bf16 && current_sm_major() < 8) {
    throw std::invalid_argument("PlatosKeyGpu::create: Bf16 precision requires SM >= 8.0 (use Fp32 on SM 7.0)");
  }
  auto impl = std::make_unique<Impl>();
  impl->precision = precision;
  impl->d_in = key->in_dim();
  impl->d_out = key->out_dim();
  impl->mask_coord = key->metadata().forbidden_coord_d2027_zero_masked ? 2027 : -1;
  impl->scale = key->scale();

  const std::size_t dm_bytes = key->donor_mean_f32().size() * sizeof(float);
  const std::size_t tm_bytes = key->target_mean_f32().size() * sizeof(float);
  CUDA_CHECK(cudaMalloc(&impl->d_donor_mean, dm_bytes));
  CUDA_CHECK(cudaMalloc(&impl->d_target_mean, tm_bytes));

  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  std::size_t rt_bytes = 0;
  if (precision == PlatosKeyPrecision::Fp32) {
    const int d_in = impl->d_in;
    const int d_out = impl->d_out;
    const auto& r_f32 = key->weights_f32();
    std::vector<float> rt_f32(static_cast<std::size_t>(d_out) * static_cast<std::size_t>(d_in));
    for (int i = 0; i < d_in; ++i) {
      const float* src_row = r_f32.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
      for (int j = 0; j < d_out; ++j) {
        rt_f32[static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in) + static_cast<std::size_t>(i)] = src_row[j];
      }
    }
    rt_bytes = rt_f32.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&impl->d_rt_f32, rt_bytes));
    CUDA_CHECK(cudaMemcpyAsync(impl->d_rt_f32, rt_f32.data(), rt_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else if (precision == PlatosKeyPrecision::Bf16) {
    const int d_in = impl->d_in;
    const int d_out = impl->d_out;
    const auto& r_f32 = key->weights_f32();
    std::vector<std::uint16_t> rt_bf16(static_cast<std::size_t>(d_out) * static_cast<std::size_t>(d_in));
    for (int i = 0; i < d_in; ++i) {
      const float* src_row = r_f32.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_out);
      for (int j = 0; j < d_out; ++j) {
        rt_bf16[static_cast<std::size_t>(j) * static_cast<std::size_t>(d_in) + static_cast<std::size_t>(i)] =
            act::host_float_to_bf16_bits(src_row[j]);
      }
    }
    rt_bytes = rt_bf16.size() * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMalloc(&impl->d_rt_bf16, rt_bytes));
    CUDA_CHECK(cudaMemcpyAsync(impl->d_rt_bf16, rt_bf16.data(), rt_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    rt_bytes = key->weights_f16_transposed().size() * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&impl->d_rt_f16, rt_bytes));
    CUDA_CHECK(cudaMemcpyAsync(impl->d_rt_f16, key->weights_f16_transposed().data(), rt_bytes, cudaMemcpyHostToDevice,
                               stream));
  }
  impl->dev_bytes = rt_bytes + dm_bytes + tm_bytes;

  CUDA_CHECK(cudaMemcpyAsync(impl->d_donor_mean, key->donor_mean_f32().data(), dm_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(impl->d_target_mean, key->target_mean_f32().data(), tm_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::shared_ptr<const PlatosKeyGpu>(new PlatosKeyGpu(std::move(key), std::move(impl)));
}

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::load(const std::filesystem::path& safetensors_path,
                                                       int expected_in_dim, void* cuda_stream,
                                                       PlatosKeyPrecision precision) {
  return create(PlatosKey::load(safetensors_path, expected_in_dim), cuda_stream, precision);
}

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::load(const std::filesystem::path& safetensors_path,
                                                       const std::filesystem::path& metadata_json_path,
                                                       int expected_in_dim, void* cuda_stream,
                                                       PlatosKeyPrecision precision) {
  return create(PlatosKey::load(safetensors_path, metadata_json_path, expected_in_dim), cuda_stream, precision);
}

PlatosKeyPrecision PlatosKeyGpu::precision() const noexcept { return impl_->precision; }

std::size_t PlatosKeyGpu::device_bytes() const noexcept { return impl_->dev_bytes; }

void PlatosKeyGpu::project(const float* d_in_x, int batch, float* d_out_y, bool l2_normalize,
                           void* cuda_stream) const {
  if (batch <= 0) return;
  if (!d_in_x || !d_out_y) throw std::invalid_argument("PlatosKeyGpu::project: null device pointer");
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const int d_in = impl_->d_in;
  const int d_out = impl_->d_out;
  const PlatosKeyPrecision prec = impl_->precision;

  auto launch_small_batch = [&](const float* x_ptr, int b_cnt, float* y_ptr) {
    const int blocks = (d_out + 7) / 8;
    if (prec == PlatosKeyPrecision::Fp32) {
      if (b_cnt == 1) {
        platos_project_small_batch_f32_kernel<1><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f32, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else if (b_cnt <= 4) {
        platos_project_small_batch_f32_kernel<4><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f32, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else {
        platos_project_small_batch_f32_kernel<8><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f32, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      }
    } else if (prec == PlatosKeyPrecision::Bf16) {
      if (b_cnt == 1) {
        platos_project_small_batch_bf16_kernel<1><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_bf16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else if (b_cnt <= 4) {
        platos_project_small_batch_bf16_kernel<4><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_bf16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else {
        platos_project_small_batch_bf16_kernel<8><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_bf16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      }
    } else {
      if (b_cnt == 1) {
        platos_project_small_batch_f16_kernel<1><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else if (b_cnt <= 4) {
        platos_project_small_batch_f16_kernel<4><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      } else {
        platos_project_small_batch_f16_kernel<8><<<blocks, 256, 0, stream>>>(
            x_ptr, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, b_cnt, d_in, d_out,
            impl_->mask_coord, y_ptr);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  };

  if (batch <= 8) {
    launch_small_batch(d_in_x, batch, d_out_y);
  } else {
    // cuBLASLt matmul for larger batches in exact configured precision:
    //   Fp16 -> CUDA_R_16F  x CUDA_R_16F  -> CUDA_R_32F (Inkling/Qwen unchanged)
    //   Bf16 -> CUDA_R_16BF x CUDA_R_16BF -> CUDA_R_32F (sm_80+ EG2, zero FP16)
    //   Fp32 -> CUDA_R_32F  x CUDA_R_32F  -> CUDA_R_32F (sm_70 V100 EG2, zero FP16)
    // Row-major C[B, d_out] = X_c[B, d_in] @ R_T[d_out, d_in]^T
    // Col-major C^T[d_out, B] = op(T)(R_T_col[d_in, d_out]) @ X_c_col[d_in, B]
    std::lock_guard<std::mutex> lk(impl_->cublas_mu);
    if (!impl_->lt_handle) {
      cublas_check(cublasLtCreate(&impl_->lt_handle), "create");
      impl_->lt_ws_bytes = 4u << 20;
      CUDA_CHECK(cudaMalloc(&impl_->lt_ws, impl_->lt_ws_bytes));
    }

    const cudaDataType_t mat_dtype = (prec == PlatosKeyPrecision::Fp32)   ? CUDA_R_32F
                                     : (prec == PlatosKeyPrecision::Bf16) ? CUDA_R_16BF
                                                                          : CUDA_R_16F;
    auto it = impl_->lt_plans.find(batch);
    if (it == impl_->lt_plans.end()) {
      Impl::LtPlan plan;
      cublasLtMatmulPreference_t pref = nullptr;
      auto cleanup = [&] {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (plan.lc) cublasLtMatrixLayoutDestroy(plan.lc);
        if (plan.lb) cublasLtMatrixLayoutDestroy(plan.lb);
        if (plan.la) cublasLtMatrixLayoutDestroy(plan.la);
        if (plan.op) cublasLtMatmulDescDestroy(plan.op);
      };
      try {
        cublas_check(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
        const cublasOperation_t ta = CUBLAS_OP_T;
        const cublasOperation_t tb = CUBLAS_OP_N;
        cublas_check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
        cublas_check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
        cublas_check(cublasLtMatrixLayoutCreate(&plan.la, mat_dtype, d_in, d_out, d_in), "la");
        cublas_check(cublasLtMatrixLayoutCreate(&plan.lb, mat_dtype, d_in, batch, d_in), "lb");
        cublas_check(cublasLtMatrixLayoutCreate(&plan.lc, CUDA_R_32F, d_out, batch, d_out), "lc");
        cublas_check(cublasLtMatmulPreferenceCreate(&pref), "pref");
        cublas_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &impl_->lt_ws_bytes, sizeof(impl_->lt_ws_bytes)),
                     "pref ws");
        cublasLtMatmulHeuristicResult_t res{};
        int found = 0;
        cublas_check(cublasLtMatmulAlgoGetHeuristic(impl_->lt_handle, plan.op, plan.la, plan.lb, plan.lc, plan.lc,
                                                    pref, 1, &res, &found),
                     "heuristic");
        cublasLtMatmulPreferenceDestroy(pref);
        pref = nullptr;
        if (found == 0) {
          if (prec == PlatosKeyPrecision::Fp16) {
            throw std::runtime_error("PlatosKeyGpu: no cuBLASLt heuristic found");
          }
          cleanup();
          plan = Impl::LtPlan{};
          plan.has_algo = false;
        } else {
          plan.algo = res.algo;
          plan.has_algo = true;
        }
        it = impl_->lt_plans.emplace(batch, plan).first;
      } catch (...) {
        cleanup();
        throw;
      }
    }
    const Impl::LtPlan& plan = it->second;
    if (!plan.has_algo) {
      // Exact precision-preserving chunked fallback when cuBLASLt reports no heuristic for odd shapes.
      for (int b0 = 0; b0 < batch; b0 += 8) {
        const int bm = std::min(8, batch - b0);
        launch_small_batch(d_in_x + static_cast<std::size_t>(b0) * static_cast<std::size_t>(d_in), bm,
                           d_out_y + static_cast<std::size_t>(b0) * static_cast<std::size_t>(d_out));
      }
    } else {
      const std::size_t need = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
      const int threads = 256;
      const int c_blocks = static_cast<int>((need + threads - 1) / threads);
      const void* a_ptr = nullptr;
      const void* b_ptr = nullptr;
      if (prec == PlatosKeyPrecision::Fp32) {
        if (impl_->x_f32_cap < need) {
          if (impl_->d_x_f32) cudaFree(impl_->d_x_f32);
          CUDA_CHECK(cudaMalloc(&impl_->d_x_f32, need * sizeof(float)));
          impl_->x_f32_cap = need;
        }
        center_to_f32_kernel<<<c_blocks, threads, 0, stream>>>(d_in_x, impl_->d_donor_mean, batch, d_in,
                                                               impl_->mask_coord, impl_->d_x_f32);
        CUDA_CHECK(cudaGetLastError());
        a_ptr = impl_->d_rt_f32;
        b_ptr = impl_->d_x_f32;
      } else if (prec == PlatosKeyPrecision::Bf16) {
        if (impl_->x_bf16_cap < need) {
          if (impl_->d_x_bf16) cudaFree(impl_->d_x_bf16);
          CUDA_CHECK(cudaMalloc(&impl_->d_x_bf16, need * sizeof(std::uint16_t)));
          impl_->x_bf16_cap = need;
        }
        center_to_bf16_kernel<<<c_blocks, threads, 0, stream>>>(d_in_x, impl_->d_donor_mean, batch, d_in,
                                                                impl_->mask_coord, impl_->d_x_bf16);
        CUDA_CHECK(cudaGetLastError());
        a_ptr = impl_->d_rt_bf16;
        b_ptr = impl_->d_x_bf16;
      } else {
        if (impl_->x_f16_cap < need) {
          if (impl_->d_x_f16) cudaFree(impl_->d_x_f16);
          CUDA_CHECK(cudaMalloc(&impl_->d_x_f16, need * sizeof(__half)));
          impl_->x_f16_cap = need;
        }
        center_to_f16_kernel<<<c_blocks, threads, 0, stream>>>(d_in_x, impl_->d_donor_mean, batch, d_in,
                                                               impl_->mask_coord, impl_->d_x_f16);
        CUDA_CHECK(cudaGetLastError());
        a_ptr = impl_->d_rt_f16;
        b_ptr = impl_->d_x_f16;
      }

      const float alpha = 1.0f;
      const float beta = 0.0f;
      cublas_check(cublasLtMatmul(impl_->lt_handle, plan.op, &alpha, a_ptr, plan.la, b_ptr, plan.lb, &beta, d_out_y,
                                  plan.lc, d_out_y, plan.lc, &plan.algo, impl_->lt_ws, impl_->lt_ws_bytes, stream),
                   "matmul");

      const std::size_t out_total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_out);
      const int o_blocks = static_cast<int>((out_total + threads - 1) / threads);
      scale_add_target_mean_kernel<<<o_blocks, threads, 0, stream>>>(d_out_y, impl_->d_target_mean, impl_->scale, batch,
                                                                     d_out);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  if (l2_normalize) {
    embed_l2_normalize_f32_cuda(d_out_y, batch, d_out, cuda_stream);
  }
}

void embed_pool_f32_cuda(const float* d_hidden, const std::int32_t* d_seq_offsets, int batch, int dim,
                         int pooling_mode, bool l2_normalize, float* d_out, void* cuda_stream) {
  if (batch <= 0 || dim <= 0) return;
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const dim3 grid((dim + threads - 1) / threads, batch);
  pool_f32_kernel<<<grid, threads, 0, stream>>>(d_hidden, d_seq_offsets, batch, dim, pooling_mode, d_out);
  CUDA_CHECK(cudaGetLastError());
  if (l2_normalize) {
    embed_l2_normalize_f32_cuda(d_out, batch, dim, cuda_stream);
  }
}

void embed_pool_act_cuda(const std::uint16_t* d_hidden_act, bool is_fp16, const std::int32_t* d_seq_offsets,
                         int batch, int dim, int pooling_mode, bool l2_normalize, float* d_out, void* cuda_stream) {
  if (batch <= 0 || dim <= 0) return;
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const dim3 grid((dim + threads - 1) / threads, batch);
  pool_act_kernel<<<grid, threads, 0, stream>>>(d_hidden_act, is_fp16, d_seq_offsets, batch, dim, pooling_mode, d_out);
  CUDA_CHECK(cudaGetLastError());
  if (l2_normalize) {
    embed_l2_normalize_f32_cuda(d_out, batch, dim, cuda_stream);
  }
}

void embed_l2_normalize_f32_cuda(float* d_data, int batch, int dim, void* cuda_stream) {
  if (batch <= 0 || dim <= 0) return;
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  l2_normalize_rows_kernel<<<batch, 256, 0, stream>>>(d_data, batch, dim);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine
