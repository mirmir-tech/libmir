#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr unsigned int kWarps = 8;
constexpr unsigned int kRowsPerWarp = 8;
constexpr float kE4M3Max = 448.0f;
constexpr float kMinDynamicScale = 1.0f / (kE4M3Max * 512.0f);

template <typename Scale>
__device__ __forceinline__ float scale_value(Scale value);

template <>
__device__ __forceinline__ float scale_value(float value) {
  return value;
}

template <>
__device__ __forceinline__ float scale_value(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float dynamic_e4m3(float value, float scale) {
  const float normalized = fminf(kE4M3Max, fmaxf(-kE4M3Max, value / scale));
  return static_cast<float>(__nv_fp8_e4m3(normalized)) * scale;
}

template <unsigned int RowsPerWarp, typename Packed, typename Scale>
__device__ __forceinline__ void direct_fp8_bf16_linear(
    const __nv_bfloat16* input, const unsigned char* weight,
    const Scale* scales, const Scale* input_scale,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int tokens,
    unsigned int rows, unsigned int columns, unsigned int scale_rows,
    unsigned int scale_columns, unsigned int scale_row_size,
    unsigned int scale_group_size,
    unsigned int inverse_scale, unsigned int has_bias,
    unsigned int activation_mode, unsigned int cache_input) {
  const unsigned int token = blockIdx.y;
  if (token >= tokens) return;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int first_row =
      blockIdx.x * kWarps * RowsPerWarp + warp * RowsPerWarp;
  const __nv_bfloat16* token_input = input + token * columns;
  __shared__ float warp_maxima[kWarps];
  __shared__ float activation_scale;
  float thread_maximum = 0.0f;
  if (activation_mode == 1u) {
    for (unsigned int column = threadIdx.x; column < columns;
         column += blockDim.x) {
      thread_maximum = fmaxf(
          thread_maximum, fabsf(__bfloat162float(token_input[column])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      thread_maximum = fmaxf(
          thread_maximum,
          __shfl_down_sync(0xffffffffu, thread_maximum, offset));
    }
    if (lane == 0u) warp_maxima[warp] = thread_maximum;
    __syncthreads();
    if (warp == 0u) {
      thread_maximum = lane < kWarps ? warp_maxima[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        thread_maximum = fmaxf(
            thread_maximum,
            __shfl_down_sync(0xffffffffu, thread_maximum, offset));
      }
      if (lane == 0u) {
        activation_scale = fmaxf(thread_maximum / kE4M3Max,
                                 kMinDynamicScale);
      }
    }
    __syncthreads();
  } else if (activation_mode == 2u) {
    if (threadIdx.x == 0u) activation_scale = scale_value(input_scale[0]);
    __syncthreads();
  }
  extern __shared__ float cached_input[];
  if (cache_input != 0u) {
    for (unsigned int column = threadIdx.x; column < columns;
         column += blockDim.x) {
      float value = __bfloat162float(token_input[column]);
      if (activation_mode != 0u) {
        value = dynamic_e4m3(value, activation_scale);
      }
      cached_input[column] = value;
    }
    __syncthreads();
  }
  float sums[RowsPerWarp] = {};
  for (unsigned int group = lane; group < columns / 4u; group += 32u) {
    const unsigned int column = group * 4u;
    float4 values;
    if (cache_input != 0u) {
      values = make_float4(cached_input[column], cached_input[column + 1u],
                           cached_input[column + 2u], cached_input[column + 3u]);
    } else {
      values = make_float4(
          __bfloat162float(token_input[column]),
          __bfloat162float(token_input[column + 1u]),
          __bfloat162float(token_input[column + 2u]),
          __bfloat162float(token_input[column + 3u]));
    }
    if (activation_mode != 0u && cache_input == 0u) {
      values.x = dynamic_e4m3(values.x, activation_scale);
      values.y = dynamic_e4m3(values.y, activation_scale);
      values.z = dynamic_e4m3(values.z, activation_scale);
      values.w = dynamic_e4m3(values.w, activation_scale);
    }
    #pragma unroll
    for (unsigned int item = 0; item < RowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        Packed packed;
        packed.__x = reinterpret_cast<const unsigned int*>(
            weight + row * columns)[group];
        const float4 quantized = static_cast<float4>(packed);
        float dot = values.x * quantized.x;
        dot = fmaf(values.y, quantized.y, dot);
        dot = fmaf(values.z, quantized.z, dot);
        dot = fmaf(values.w, quantized.w, dot);
        if (scale_columns == 1u) {
          sums[item] += dot;
        } else {
          const unsigned int scale_row = scale_rows == 1u ? 0u : row / scale_row_size;
          const unsigned int scale_column = column / scale_group_size;
          const float scale =
              scale_value(scales[scale_row * scale_columns + scale_column]);
          sums[item] += inverse_scale == 0u ? dot * scale : dot / scale;
        }
      }
    }
  }
  #pragma unroll
  for (unsigned int item = 0; item < RowsPerWarp; ++item) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  if (lane == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < RowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        float value = sums[item];
        if (scale_columns == 1u) {
          const unsigned int scale_row = scale_rows == 1u ? 0u : row / scale_row_size;
          const float scale = scale_value(scales[scale_row]);
          value = inverse_scale == 0u ? value * scale : value / scale;
        }
        value += has_bias == 0u ? 0.0f : __bfloat162float(bias[row]);
        output[token * rows + row] = __float2bfloat16_rn(value);
      }
    }
  }
}
}

extern "C" __global__ void libmir_cuda_direct_fp8_bf16_linear_f32_scale(
    const __nv_bfloat16* input, const unsigned char* weight,
    const float* scales, const float* input_scale,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int tokens,
    unsigned int rows, unsigned int columns, unsigned int scale_rows,
    unsigned int scale_columns, unsigned int scale_row_size,
    unsigned int scale_group_size,
    unsigned int inverse_scale, unsigned int has_bias,
    unsigned int activation_mode, unsigned int e5m2) {
  if (e5m2 != 0u) {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e5m2>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size, inverse_scale, has_bias,
        activation_mode, 0u);
  } else {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e4m3>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size, inverse_scale, has_bias,
        activation_mode, 0u);
  }
}

extern "C" __global__ void libmir_cuda_direct_fp8_bf16_linear_f32_scale_cached(
    const __nv_bfloat16* input, const unsigned char* weight,
    const float* scales, const float* input_scale,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int tokens, unsigned int rows, unsigned int columns,
    unsigned int scale_rows, unsigned int scale_columns,
    unsigned int scale_row_size, unsigned int scale_group_size,
    unsigned int inverse_scale, unsigned int has_bias,
    unsigned int activation_mode, unsigned int e5m2,
    unsigned int cache_input) {
  if (e5m2 != 0u) {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e5m2>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size,
        inverse_scale, has_bias, activation_mode, cache_input);
  } else {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e4m3>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size,
        inverse_scale, has_bias, activation_mode, cache_input);
  }
}

extern "C" __global__ void libmir_cuda_direct_fp8_bf16_linear_bf16_scale(
    const __nv_bfloat16* input, const unsigned char* weight,
    const __nv_bfloat16* scales, const __nv_bfloat16* input_scale,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output, unsigned int tokens,
    unsigned int rows, unsigned int columns, unsigned int scale_rows,
    unsigned int scale_columns, unsigned int scale_row_size,
    unsigned int scale_group_size,
    unsigned int inverse_scale, unsigned int has_bias,
    unsigned int activation_mode, unsigned int e5m2) {
  if (e5m2 != 0u) {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e5m2>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size, inverse_scale, has_bias,
        activation_mode, 0u);
  } else {
    direct_fp8_bf16_linear<kRowsPerWarp, __nv_fp8x4_e4m3>(
        input, weight, scales, input_scale, bias, output, tokens, rows, columns,
        scale_rows, scale_columns, scale_row_size, scale_group_size, inverse_scale, has_bias,
        activation_mode, 0u);
  }
}
