#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ float gptq_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short gptq_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

__device__ __forceinline__ float gptq_value(
    const int* weight, const int* zero_points, const unsigned short* scales,
    const int* group_indices,
    unsigned int row, unsigned int feature, unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size, unsigned int zero_offset) {
  const unsigned int packed_output = output_features / 8u;
  const unsigned int weight_word = static_cast<unsigned int>(
      weight[(feature / 8u) * output_features + row]);
  const unsigned int group = static_cast<unsigned int>(group_indices[feature]);
  if (group >= input_features / group_size) {
    return 0.0f;
  }
  const unsigned int zero_word = static_cast<unsigned int>(
      zero_points[group * packed_output + row / 8u]);
  const unsigned int value = (weight_word >> ((feature & 7u) * 4u)) & 15u;
  const unsigned int encoded_zero = (zero_word >> ((row & 7u) * 4u)) & 15u;
  const unsigned int zero = (encoded_zero + zero_offset) & 15u;
  const float scale = __half2float(
      reinterpret_cast<const __half*>(scales)[group * output_features + row]);
  return (static_cast<float>(value) - static_cast<float>(zero)) * scale;
}

extern "C" __global__ void libmir_cuda_gptq_gemv_bf16(
    const unsigned short* input, const int* weight, const int* zero_points,
    const unsigned short* scales, const int* group_indices, unsigned short* output,
    unsigned int input_features, unsigned int output_features,
    unsigned int group_size, unsigned int zero_offset) {
  constexpr unsigned int rows_per_block = 8u;
  constexpr unsigned int values_per_thread = 8u;
  const unsigned int row = blockIdx.x * rows_per_block + threadIdx.y;
  if (row >= output_features) {
    return;
  }
  float sum = 0.0f;
  for (unsigned int base = threadIdx.x * values_per_thread; base < input_features;
       base += 32u * values_per_thread) {
#pragma unroll
    for (unsigned int offset = 0; offset < values_per_thread; ++offset) {
      const unsigned int feature = base + offset;
      if (feature < input_features) {
        sum += gptq_bf16_to_float(input[feature]) *
            gptq_value(weight, zero_points, scales, group_indices, row, feature,
                       input_features, output_features, group_size, zero_offset);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) {
    output[row] = gptq_float_to_bf16(sum);
  }
}

extern "C" __global__ void libmir_cuda_gptq_qmm_bf16(
    const unsigned short* input, const int* weight, const int* zero_points,
    const unsigned short* scales, const int* group_indices, unsigned short* output, unsigned int tokens,
    unsigned int input_features, unsigned int output_features,
    unsigned int group_size, unsigned int zero_offset) {
  constexpr unsigned int tile = 16u;
  constexpr unsigned int warps = 4u;
  constexpr unsigned int token_block = tile * warps;
  __shared__ __nv_bfloat16 input_tiles[warps][tile * tile];
  __shared__ __nv_bfloat16 weight_tile[tile * tile];
  __shared__ float output_tiles[warps][tile * tile];
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int thread = warp * 32u + lane;
  const unsigned int output_base = blockIdx.x * tile;
  const unsigned int token_base = blockIdx.y * token_block + warp * tile;
  wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (unsigned int feature_base = 0; feature_base < input_features;
       feature_base += tile) {
    for (unsigned int index = lane; index < tile * tile; index += 32u) {
      const unsigned int token = token_base + index / tile;
      const unsigned int feature = feature_base + index % tile;
      input_tiles[warp][index] = token < tokens
          ? reinterpret_cast<const __nv_bfloat16*>(
                input)[token * input_features + feature]
          : __float2bfloat16(0.0f);
    }
    for (unsigned int index = thread; index < tile * tile;
         index += 32u * warps) {
      const unsigned int row = output_base + index / tile;
      const unsigned int feature = feature_base + index % tile;
      float value = 0.0f;
      if (row < output_features) {
        value = gptq_value(weight, zero_points, scales, group_indices, row,
                           feature, input_features, output_features, group_size,
                           zero_offset);
      }
      weight_tile[index] = __float2bfloat16(value);
    }
    __syncthreads();
    wmma::fragment<
        wmma::matrix_a, tile, tile, tile, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<
        wmma::matrix_b, tile, tile, tile, __nv_bfloat16, wmma::col_major> b;
    wmma::load_matrix_sync(a, input_tiles[warp], tile);
    wmma::load_matrix_sync(b, weight_tile, tile);
    wmma::mma_sync(accumulator, a, b, accumulator);
    __syncthreads();
  }
  wmma::store_matrix_sync(output_tiles[warp], accumulator, tile, wmma::mem_row_major);
  for (unsigned int index = lane; index < tile * tile; index += 32u) {
    const unsigned int token = token_base + index / tile;
    const unsigned int row = output_base + index % tile;
    if (token < tokens && row < output_features) {
      reinterpret_cast<__nv_bfloat16*>(output)[token * output_features + row] =
          __float2bfloat16(output_tiles[warp][index]);
    }
  }
}
