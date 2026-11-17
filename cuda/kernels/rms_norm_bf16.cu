#include <cuda_bf16.h>

// One block per row. Rows of whole 8-value vectors are read with 16-byte
// loads, and the first `kCached` vectors of each thread stay in registers
// between the sum and the scaling, so a decode row (one row, thousands of
// columns) costs one load round instead of two serial passes. Blocks hold up
// to 1024 threads (`rms_norm_threads` sizes them to the row).
namespace {

constexpr unsigned int kCached = 4u;

__device__ __forceinline__ float squares(const uint4& vector) {
  const unsigned int words[4] = {vector.x, vector.y, vector.z, vector.w};
  float sum = 0.0f;
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
    sum = fmaf(pair.x, pair.x, sum);
    sum = fmaf(pair.y, pair.y, sum);
  }
  return sum;
}

template <bool kWeighted>
__device__ __forceinline__ uint4 scaled(const uint4& vector, const uint4& weight, float inverse) {
  const unsigned int words[4] = {vector.x, vector.y, vector.z, vector.w};
  const unsigned int scales[4] = {weight.x, weight.y, weight.z, weight.w};
  unsigned int result[4];
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
    pair.x *= inverse;
    pair.y *= inverse;
    if constexpr (kWeighted) {
      const float2 scale =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&scales[i]));
      pair.x *= scale.x;
      pair.y *= scale.y;
    }
    const __nv_bfloat162 packed = __floats2bfloat162_rn(pair.x, pair.y);
    result[i] = *reinterpret_cast<const unsigned int*>(&packed);
  }
  return make_uint4(result[0], result[1], result[2], result[3]);
}

// The block's sum of `value`, in every thread.
__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warps[32];
  __shared__ float total;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x >> 5u] = value;
  __syncthreads();
  if (threadIdx.x < 32u) {
    value = threadIdx.x < (blockDim.x + 31u) / 32u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (threadIdx.x == 0u) total = value;
  }
  __syncthreads();
  return total;
}

template <bool kWeighted>
__device__ void rms_norm_vectors(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int columns, float epsilon) {
  const unsigned long long row = blockIdx.x;
  const unsigned int vectors = columns >> 3u;
  const uint4* in = reinterpret_cast<const uint4*>(input + row * columns);
  uint4* out = reinterpret_cast<uint4*>(output + row * columns);
  const uint4* scales = reinterpret_cast<const uint4*>(weight);
  const uint4 none = make_uint4(0u, 0u, 0u, 0u);
  uint4 cache[kCached];
  float sum = 0.0f;
#pragma unroll
  for (unsigned int k = 0; k < kCached; ++k) {
    const unsigned int v = threadIdx.x + k * blockDim.x;
    cache[k] = v < vectors ? in[v] : none;
    sum += squares(cache[k]);
  }
  for (unsigned int v = threadIdx.x + kCached * blockDim.x; v < vectors; v += blockDim.x) {
    sum += squares(in[v]);
  }
  const float inverse = rsqrtf(block_sum(sum) / columns + epsilon);
#pragma unroll
  for (unsigned int k = 0; k < kCached; ++k) {
    const unsigned int v = threadIdx.x + k * blockDim.x;
    if (v < vectors) out[v] = scaled<kWeighted>(cache[k], kWeighted ? scales[v] : none, inverse);
  }
  for (unsigned int v = threadIdx.x + kCached * blockDim.x; v < vectors; v += blockDim.x) {
    out[v] = scaled<kWeighted>(in[v], kWeighted ? scales[v] : none, inverse);
  }
}

template <bool kWeighted>
__device__ void rms_norm_scalar(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int columns, float epsilon) {
  const unsigned long long row = blockIdx.x;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]);
    sum = fmaf(value, value, sum);
  }
  const float inverse = rsqrtf(block_sum(sum) / columns + epsilon);
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]) * inverse;
    output[row * columns + column] =
        __float2bfloat16_rn(kWeighted ? value * __bfloat162float(weight[column]) : value);
  }
}

template <bool kWeighted>
__device__ void rms_norm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, float epsilon) {
  if (blockIdx.x >= rows) return;
  if ((columns & 7u) == 0u) {
    rms_norm_vectors<kWeighted>(input, weight, output, columns, epsilon);
  } else {
    rms_norm_scalar<kWeighted>(input, weight, output, columns, epsilon);
  }
}

}  // namespace

extern "C" __global__ void libmir_cuda_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    float epsilon) {
  rms_norm<true>(input, weight, output, rows, columns, epsilon);
}

extern "C" __global__ void libmir_cuda_rms_norm_unit_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, float epsilon) {
  rms_norm<false>(input, nullptr, output, rows, columns, epsilon);
}
