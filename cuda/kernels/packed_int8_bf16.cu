#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ float packed_int8_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short packed_int8_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

__device__ __forceinline__ float packed_integer_value(
    const int* weight, unsigned int row, unsigned int feature,
    const unsigned short* scales, unsigned int input_features,
    unsigned int bits, unsigned int group_size) {
  const unsigned int values_per_word = 32u / bits;
  const unsigned int words_per_row = input_features / values_per_word;
  const unsigned int word = static_cast<unsigned int>(
      weight[row * words_per_row + feature / values_per_word]);
  const unsigned int mask = (1u << bits) - 1u;
  const unsigned int value =
      (word >> ((feature % values_per_word) * bits)) & mask;
  const unsigned int groups_per_row = input_features / group_size;
  const float scale = packed_int8_bf16_to_float(
      scales[row * groups_per_row + feature / group_size]);
  return (static_cast<float>(value) - static_cast<float>(1u << (bits - 1u))) * scale;
}

extern "C" __global__ void libmir_cuda_packed_int8_embedding_bf16(
    const unsigned int* selected, const int* weight, const unsigned short* scales,
    unsigned short* output, unsigned int selected_start, unsigned int tokens,
    unsigned int vocab, unsigned int hidden, float output_scale,
    unsigned int bits, unsigned int group_size) {
  const unsigned int token = blockIdx.y;
  const unsigned int feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || feature >= hidden) {
    return;
  }
  const unsigned int row = selected[selected_start + token];
  if (row >= vocab) {
    return;
  }
  const float value =
      packed_integer_value(weight, row, feature, scales, hidden, bits, group_size);
  output[token * hidden + feature] = packed_int8_float_to_bf16(value * output_scale);
}

extern "C" __global__ void libmir_cuda_packed_int8_gemv_bf16(
    const unsigned short* input, const int* weight, const unsigned short* scales,
    unsigned short* output, unsigned int input_features, unsigned int output_features,
    unsigned int bits, unsigned int group_size) {
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
      sum += packed_int8_bf16_to_float(input[feature]) *
          packed_integer_value(
              weight, row, feature, scales, input_features, bits, group_size);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) {
    output[row] = packed_int8_float_to_bf16(sum);
  }
}

extern "C" __global__ void libmir_cuda_packed_int8_qmm_bf16(
    const unsigned short* input, const int* weight, const unsigned short* scales,
    unsigned short* output, unsigned int tokens, unsigned int input_features,
    unsigned int output_features, unsigned int bits, unsigned int group_size) {
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
        value = packed_integer_value(
            weight, row, feature, scales, input_features, bits, group_size);
      }
      weight_tile[index % tile + (index / tile) * tile] = __float2bfloat16(value);
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
