#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr unsigned int kWarps = 8;
constexpr float kE4M3Max = 448.0f;
constexpr float kMinDynamicScale = 1.0f / (kE4M3Max * 512.0f);

template <typename Scale>
__device__ __forceinline__ float scale_value(const Scale* scale);

template <>
__device__ __forceinline__ float scale_value(const float* scale) {
  return scale[0];
}

template <>
__device__ __forceinline__ float scale_value(const __nv_bfloat16* scale) {
  return __bfloat162float(scale[0]);
}

__device__ __forceinline__ unsigned char encode(float value, float scale) {
  const float normalized = fminf(
      kE4M3Max, fmaxf(-kE4M3Max, value / scale));
  return static_cast<unsigned char>(__nv_fp8_e4m3(normalized).__x);
}

template <typename Scale>
__device__ __forceinline__ void static_quantize(
    const __nv_bfloat16* input, unsigned char* output, float* scales,
    const Scale* input_scale, unsigned int tokens, unsigned int columns) {
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  __shared__ float scale;
  if (threadIdx.x == 0u) {
    scale = scale_value(input_scale);
    scales[token] = scale;
  }
  __syncthreads();
  const __nv_bfloat16* row = input + token * columns;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    output[token * columns + column] =
        encode(__bfloat162float(row[column]), scale);
  }
}
}  // namespace

extern "C" __global__ void libmir_cuda_dynamic_e4m3_quantize_bf16(
    const __nv_bfloat16* input, unsigned char* output, float* scales,
    unsigned int tokens, unsigned int columns) {
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const __nv_bfloat16* row = input + token * columns;
  __shared__ float warp_maxima[kWarps];
  __shared__ float scale;
  float maximum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(row[column])));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffffu, maximum, offset));
  }
  if (lane == 0u) warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0u) {
    maximum = lane < kWarps ? warp_maxima[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum,
                      __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    if (lane == 0u) {
      scale = fmaxf(maximum / kE4M3Max, kMinDynamicScale);
      scales[token] = scale;
    }
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    output[token * columns + column] =
        encode(__bfloat162float(row[column]), scale);
  }
}

extern "C" __global__ void libmir_cuda_static_e4m3_quantize_bf16_f32_scale(
    const __nv_bfloat16* input, unsigned char* output, float* scales,
    const float* input_scale, unsigned int tokens, unsigned int columns) {
  static_quantize(input, output, scales, input_scale, tokens, columns);
}

extern "C" __global__ void libmir_cuda_static_e4m3_quantize_bf16_bf16_scale(
    const __nv_bfloat16* input, unsigned char* output, float* scales,
    const __nv_bfloat16* input_scale, unsigned int tokens,
    unsigned int columns) {
  static_quantize(input, output, scales, input_scale, tokens, columns);
}
