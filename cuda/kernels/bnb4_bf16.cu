#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ float bnb4_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short bnb4_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

__device__ __forceinline__ float bnb4_value(
    const unsigned char* weight, const unsigned char* absmax,
    const unsigned char* quant_map_raw, const unsigned char* nested_absmax_raw,
    const unsigned char* nested_quant_map_raw, unsigned int element,
    unsigned int block_size, unsigned int nested_block_size, float nested_offset) {
  const unsigned char packed = weight[element >> 1u];
  const unsigned int code = (element & 1u) == 0u ? packed >> 4u : packed & 0x0fu;
  const float* quant_map = reinterpret_cast<const float*>(quant_map_raw);
  const unsigned int block = element / block_size;
  float scale;
  if (nested_block_size == 0u) {
    scale = reinterpret_cast<const float*>(absmax)[block];
  } else {
    const float* nested_absmax = reinterpret_cast<const float*>(nested_absmax_raw);
    const float* nested_quant_map = reinterpret_cast<const float*>(nested_quant_map_raw);
    scale = nested_quant_map[absmax[block]] * nested_absmax[block / nested_block_size]
        + nested_offset;
  }
  return quant_map[code] * scale;
}

extern "C" __global__ void libmir_cuda_bnb4_gemv_bf16(
    const unsigned short* input, const unsigned char* weight,
    const unsigned char* absmax, const unsigned char* quant_map,
    const unsigned char* nested_absmax, const unsigned char* nested_quant_map,
    unsigned short* output, unsigned int input_features, unsigned int output_features,
    unsigned int block_size, unsigned int nested_block_size, float nested_offset) {
  constexpr unsigned int rows_per_block = 8u;
  constexpr unsigned int values_per_thread = 8u;
  const unsigned int row = blockIdx.x * rows_per_block + threadIdx.y;
  if (row >= output_features) return;
  float sum = 0.0f;
  for (unsigned int base = threadIdx.x * values_per_thread; base < input_features;
       base += 32u * values_per_thread) {
#pragma unroll
    for (unsigned int offset = 0; offset < values_per_thread; ++offset) {
      const unsigned int feature = base + offset;
      const unsigned int element = row * input_features + feature;
      sum += bnb4_bf16_to_float(input[feature]) * bnb4_value(
          weight, absmax, quant_map, nested_absmax, nested_quant_map, element,
          block_size, nested_block_size, nested_offset);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) output[row] = bnb4_float_to_bf16(sum);
}

extern "C" __global__ void libmir_cuda_bnb4_qmm_bf16(
    const unsigned short* input, const unsigned char* weight,
    const unsigned char* absmax, const unsigned char* quant_map,
    const unsigned char* nested_absmax, const unsigned char* nested_quant_map,
    unsigned short* output, unsigned int tokens, unsigned int input_features,
    unsigned int output_features, unsigned int block_size,
    unsigned int nested_block_size, float nested_offset) {
  constexpr unsigned int tile = 16u;
  constexpr unsigned int warps = 4u;
  __shared__ __nv_bfloat16 input_tiles[warps][tile * tile];
  __shared__ __nv_bfloat16 weight_tile[tile * tile];
  __shared__ float output_tiles[warps][tile * tile];
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int thread = warp * 32u + lane;
  const unsigned int output_base = blockIdx.x * tile;
  const unsigned int token_base = blockIdx.y * tile * warps + warp * tile;
  wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (unsigned int feature_base = 0; feature_base < input_features; feature_base += tile) {
    for (unsigned int index = lane; index < tile * tile; index += 32u) {
      const unsigned int token = token_base + index / tile;
      const unsigned int feature = feature_base + index % tile;
      input_tiles[warp][index] = token < tokens
          ? reinterpret_cast<const __nv_bfloat16*>(input)[token * input_features + feature]
          : __float2bfloat16(0.0f);
    }
    for (unsigned int index = thread; index < tile * tile; index += 32u * warps) {
      const unsigned int row = output_base + index / tile;
      const unsigned int feature = feature_base + index % tile;
      float value = 0.0f;
      if (row < output_features) {
        value = bnb4_value(weight, absmax, quant_map, nested_absmax, nested_quant_map,
            row * input_features + feature, block_size, nested_block_size, nested_offset);
      }
      weight_tile[index] = __float2bfloat16(value);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, tile, tile, tile, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, tile, tile, tile, __nv_bfloat16, wmma::col_major> b;
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
