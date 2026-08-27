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

extern "C" __global__ void libmir_cuda_dynamic_e4m3_norm_gate_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, unsigned char* output, float* scales,
    unsigned int tokens, unsigned int columns, unsigned int value_heads,
    unsigned int gate_stride, unsigned int gate_offset, float epsilon,
    float weight_shift) {
  constexpr unsigned int head_width = 128u;
  const unsigned int heads_per_wave = blockDim.x / head_width;
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  const unsigned int group = threadIdx.x / head_width;
  const unsigned int column = threadIdx.x % head_width;
  const unsigned int lane = column & 31u;
  const unsigned int group_warp = column >> 5u;
  const unsigned int warp = threadIdx.x >> 5u;
  extern __shared__ __nv_bfloat16 gated[];
  __shared__ float warp_values[32];
  __shared__ float output_scale;
  float maximum = 0.0f;
  for (unsigned int wave = 0u; wave * heads_per_wave < value_heads; ++wave) {
    const unsigned int head = wave * heads_per_wave + group;
    const bool valid = head < value_heads;
    const unsigned int local = head * head_width + column;
    const unsigned int index = token * columns + local;
    const float input_value = valid ? __bfloat162float(input[index]) : 0.0f;
    float sum = input_value * input_value;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) warp_values[group * 4u + group_warp] = sum;
    __syncthreads();
    if (group_warp == 0u) {
      sum = lane < 4u ? warp_values[group * 4u + lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      if (lane == 0u) {
        warp_values[group * 4u] = rsqrtf(sum / head_width + epsilon);
      }
    }
    __syncthreads();
    if (valid) {
      const __nv_bfloat16 normalized = __float2bfloat16_rn(
          input_value * warp_values[group * 4u] *
          (__bfloat162float(weight[column]) + weight_shift));
      const unsigned int gate_index =
          token * gate_stride + gate_offset + local;
      const float gate_value = __bfloat162float(gate[gate_index]);
      const float activated = gate_value / (1.0f + __expf(-gate_value));
      const __nv_bfloat16 value = __float2bfloat16_rn(
          activated * __bfloat162float(normalized));
      gated[local] = value;
      maximum = fmaxf(maximum, fabsf(__bfloat162float(value)));
    }
    __syncthreads();
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffffu, maximum, offset));
  }
  if (lane == 0u) warp_values[warp] = maximum;
  __syncthreads();
  if (warp == 0u) {
    maximum = lane < blockDim.x / 32u ? warp_values[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(maximum,
                      __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    if (lane == 0u) {
      output_scale = fmaxf(maximum / kE4M3Max, kMinDynamicScale);
      scales[token] = output_scale;
    }
  }
  __syncthreads();
  for (unsigned int local = threadIdx.x; local < columns; local += blockDim.x) {
    output[token * columns + local] =
        encode(__bfloat162float(gated[local]), output_scale);
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
