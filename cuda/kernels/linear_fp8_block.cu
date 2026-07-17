#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr unsigned int kBlock = 128;
constexpr unsigned int kWarps = 8;

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ float block_max(float value, float* maxima) {
  value = warp_max(value);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  if (lane == 0u) maxima[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32u ? maxima[lane] : 0.0f;
  return warp_max(value);
}
}  // namespace

extern "C" __global__ void libmir_cuda_quantize_block_fp8_weight(
    const __nv_bfloat16* source, unsigned char* weight, float* scales,
    unsigned int rows, unsigned int columns) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  const unsigned int block = blockIdx.y;
  const unsigned int column = block * kBlock + threadIdx.x;
  const unsigned int index = row * columns + column;
  const float value = __bfloat162float(source[index]);
  __shared__ float maxima[kWarps];
  const float maximum = block_max(fabsf(value), maxima);
  __shared__ float scale;
  if (threadIdx.x == 0u) {
    scale = maximum == 0.0f ? 1.0f : maximum / 448.0f;
    scales[row * (columns / kBlock) + block] = scale;
  }
  __syncthreads();
  weight[index] = __nv_fp8_e4m3(value / scale).__x;
}

extern "C" __global__ void libmir_cuda_block_fp8x4_bf16_linear(
    const __nv_bfloat16* input, const unsigned char* weight,
    const float* scales, __nv_bfloat16* output, unsigned int rows,
    unsigned int columns) {
  constexpr unsigned int kRowsPerWarp = 8;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int first_row =
      blockIdx.x * kWarps * kRowsPerWarp + warp * kRowsPerWarp;
  float sums[kRowsPerWarp] = {};
  for (unsigned int group = lane; group < columns / 4u; group += 32u) {
    const unsigned int column = group * 4u;
    const float4 values = make_float4(
        __bfloat162float(input[column]), __bfloat162float(input[column + 1u]),
        __bfloat162float(input[column + 2u]),
        __bfloat162float(input[column + 3u]));
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        __nv_fp8x4_e4m3 packed;
        packed.__x = reinterpret_cast<const unsigned int*>(
            weight + row * columns)[group];
        const float4 quantized = static_cast<float4>(packed);
        float dot = values.x * quantized.x;
        dot = fmaf(values.y, quantized.y, dot);
        dot = fmaf(values.z, quantized.z, dot);
        dot = fmaf(values.w, quantized.w, dot);
        const float scale = scales[row * (columns / kBlock) + group / 32u];
        sums[item] = fmaf(dot, scale, sums[item]);
      }
    }
  }
  #pragma unroll
  for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  if (lane == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) output[row] = __float2bfloat16_rn(sums[item]);
    }
  }
}
