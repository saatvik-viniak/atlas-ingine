#include "ingine/platos_key.h"

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
                                          float scale, int batch, int d_in, int d_out, float* __restrict__ out) {
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

      for (int b = 0; b < batch; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i;
        const float4 x0 = *reinterpret_cast<const float4*>(xb);
        const float4 x1 = *reinterpret_cast<const float4*>(xb + 4);
        float s = acc[b];
        s = fmaf(x0.x - dm[0], wv[0], s);
        s = fmaf(x0.y - dm[1], wv[1], s);
        s = fmaf(x0.z - dm[2], wv[2], s);
        s = fmaf(x0.w - dm[3], wv[3], s);
        s = fmaf(x1.x - dm[4], wv[4], s);
        s = fmaf(x1.y - dm[5], wv[5], s);
        s = fmaf(x1.z - dm[6], wv[6], s);
        s = fmaf(x1.w - dm[7], wv[7], s);
        acc[b] = s;
      }
    }
  } else {
    for (int i = lane; i < d_in; i += 32) {
      const float wv = __half2float(w_row[i]);
      const float dmv = donor_mean[i];
      for (int b = 0; b < batch; ++b) {
        const float xv = x[static_cast<std::size_t>(b) * static_cast<std::size_t>(d_in) + i];
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
                                     int d_in, __half* __restrict__ out_f16) {
  const std::size_t total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int col = static_cast<int>(idx % static_cast<std::size_t>(d_in));
  out_f16[idx] = __float2half_rn(x[idx] - donor_mean[col]);
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
  int d_in = 0;
  int d_out = 0;
  float scale = 1.0f;
  __half* d_rt_f16 = nullptr;      // [d_out, d_in] FP16
  float* d_donor_mean = nullptr;   // [d_in] FP32
  float* d_target_mean = nullptr;  // [d_out] FP32
  std::size_t dev_bytes = 0;

  // Lazily allocated cuBLASLt handle + cached per-batch plans for large batches (batch > 8)
  struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr;
    cublasLtMatrixLayout_t lb = nullptr;
    cublasLtMatrixLayout_t lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };
  mutable std::mutex cublas_mu;
  mutable cublasLtHandle_t lt_handle = nullptr;
  mutable void* lt_ws = nullptr;
  mutable std::size_t lt_ws_bytes = 0;
  mutable __half* d_x_f16 = nullptr;
  mutable std::size_t x_f16_cap = 0;
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
    if (d_donor_mean) cudaFree(d_donor_mean);
    if (d_target_mean) cudaFree(d_target_mean);
    if (d_x_f16) cudaFree(d_x_f16);
    if (lt_ws) cudaFree(lt_ws);
    if (lt_handle) cublasLtDestroy(lt_handle);
  }
};

PlatosKeyGpu::PlatosKeyGpu(std::shared_ptr<const PlatosKey> key, std::unique_ptr<Impl> impl)
    : key_(std::move(key)), impl_(std::move(impl)) {}

PlatosKeyGpu::~PlatosKeyGpu() = default;

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::create(std::shared_ptr<const PlatosKey> key, void* cuda_stream) {
  if (!key) throw std::invalid_argument("PlatosKeyGpu::create: null PlatosKey");
  auto impl = std::make_unique<Impl>();
  impl->d_in = key->in_dim();
  impl->d_out = key->out_dim();
  impl->scale = key->scale();

  const std::size_t rt_bytes = key->weights_f16_transposed().size() * sizeof(__half);
  const std::size_t dm_bytes = key->donor_mean_f32().size() * sizeof(float);
  const std::size_t tm_bytes = key->target_mean_f32().size() * sizeof(float);

  CUDA_CHECK(cudaMalloc(&impl->d_rt_f16, rt_bytes));
  CUDA_CHECK(cudaMalloc(&impl->d_donor_mean, dm_bytes));
  CUDA_CHECK(cudaMalloc(&impl->d_target_mean, tm_bytes));
  impl->dev_bytes = rt_bytes + dm_bytes + tm_bytes;

  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  CUDA_CHECK(cudaMemcpyAsync(impl->d_rt_f16, key->weights_f16_transposed().data(), rt_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(impl->d_donor_mean, key->donor_mean_f32().data(), dm_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(impl->d_target_mean, key->target_mean_f32().data(), tm_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::shared_ptr<const PlatosKeyGpu>(new PlatosKeyGpu(std::move(key), std::move(impl)));
}

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::load(const std::filesystem::path& safetensors_path,
                                                       int expected_in_dim, void* cuda_stream) {
  return create(PlatosKey::load(safetensors_path, expected_in_dim), cuda_stream);
}

std::shared_ptr<const PlatosKeyGpu> PlatosKeyGpu::load(const std::filesystem::path& safetensors_path,
                                                       const std::filesystem::path& metadata_json_path,
                                                       int expected_in_dim, void* cuda_stream) {
  return create(PlatosKey::load(safetensors_path, metadata_json_path, expected_in_dim), cuda_stream);
}

std::size_t PlatosKeyGpu::device_bytes() const noexcept { return impl_->dev_bytes; }

void PlatosKeyGpu::project(const float* d_in_x, int batch, float* d_out_y, bool l2_normalize,
                           void* cuda_stream) const {
  if (batch <= 0) return;
  if (!d_in_x || !d_out_y) throw std::invalid_argument("PlatosKeyGpu::project: null device pointer");
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const int d_in = impl_->d_in;
  const int d_out = impl_->d_out;

  if (batch <= 8) {
    const int blocks = (d_out + 7) / 8;
    if (batch == 1) {
      platos_project_small_batch_f16_kernel<1><<<blocks, 256, 0, stream>>>(
          d_in_x, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, batch, d_in, d_out,
          d_out_y);
    } else if (batch <= 4) {
      platos_project_small_batch_f16_kernel<4><<<blocks, 256, 0, stream>>>(
          d_in_x, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, batch, d_in, d_out,
          d_out_y);
    } else {
      platos_project_small_batch_f16_kernel<8><<<blocks, 256, 0, stream>>>(
          d_in_x, impl_->d_donor_mean, impl_->d_rt_f16, impl_->d_target_mean, impl_->scale, batch, d_in, d_out,
          d_out_y);
    }
    CUDA_CHECK(cudaGetLastError());
  } else {
    // Tensor-Core cuBLASLt FP16xFP16 -> FP32 matmul for larger batches:
    // Row-major C[B, d_out] = X_c[B, d_in] @ R_T[d_out, d_in]^T
    // Col-major C^T[d_out, B] = op(T)(R_T_col[d_in, d_out]) @ X_c_col[d_in, B]
    std::lock_guard<std::mutex> lk(impl_->cublas_mu);
    if (!impl_->lt_handle) {
      cublas_check(cublasLtCreate(&impl_->lt_handle), "create");
      impl_->lt_ws_bytes = 4u << 20;
      CUDA_CHECK(cudaMalloc(&impl_->lt_ws, impl_->lt_ws_bytes));
    }
    const std::size_t need = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_in);
    if (impl_->x_f16_cap < need) {
      if (impl_->d_x_f16) cudaFree(impl_->d_x_f16);
      CUDA_CHECK(cudaMalloc(&impl_->d_x_f16, need * sizeof(__half)));
      impl_->x_f16_cap = need;
    }
    const int threads = 256;
    const int c_blocks = static_cast<int>((need + threads - 1) / threads);
    center_to_f16_kernel<<<c_blocks, threads, 0, stream>>>(d_in_x, impl_->d_donor_mean, batch, d_in, impl_->d_x_f16);
    CUDA_CHECK(cudaGetLastError());

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
        cublas_check(cublasLtMatrixLayoutCreate(&plan.la, CUDA_R_16F, d_in, d_out, d_in), "la");
        cublas_check(cublasLtMatrixLayoutCreate(&plan.lb, CUDA_R_16F, d_in, batch, d_in), "lb");
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
        if (found == 0) throw std::runtime_error("PlatosKeyGpu: no cuBLASLt heuristic found");
        plan.algo = res.algo;
        it = impl_->lt_plans.emplace(batch, plan).first;
      } catch (...) {
        cleanup();
        throw;
      }
    }
    const Impl::LtPlan& plan = it->second;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasLtMatmul(impl_->lt_handle, plan.op, &alpha, impl_->d_rt_f16, plan.la, impl_->d_x_f16, plan.lb,
                                &beta, d_out_y, plan.lc, d_out_y, plan.lc, &plan.algo, impl_->lt_ws, impl_->lt_ws_bytes,
                                stream),
                 "matmul");

    const std::size_t out_total = static_cast<std::size_t>(batch) * static_cast<std::size_t>(d_out);
    const int o_blocks = static_cast<int>((out_total + threads - 1) / threads);
    scale_add_target_mean_kernel<<<o_blocks, threads, 0, stream>>>(d_out_y, impl_->d_target_mean, impl_->scale, batch,
                                                                   d_out);
    CUDA_CHECK(cudaGetLastError());
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
