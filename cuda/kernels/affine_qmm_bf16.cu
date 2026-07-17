#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ float libmir_cuda_qmm_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

template <unsigned int bits>
__device__ __forceinline__ float libmir_cuda_qmm_weight(
    const unsigned int* weight, const unsigned short* scales,
    const unsigned short* biases, unsigned int matrix_row, unsigned int feature,
    unsigned int words_per_row, unsigned int groups_per_row, unsigned int group_size) {
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int mask = (1u << bits) - 1u;
  const unsigned int word = weight[matrix_row * words_per_row + feature / values_per_word];
  const unsigned int lane = feature % values_per_word;
  const unsigned int group = matrix_row * groups_per_row + feature / group_size;
  const float scale = libmir_cuda_qmm_bf16_to_float(scales[group]);
  const float bias = libmir_cuda_qmm_bf16_to_float(biases[group]);
  return scale * static_cast<float>((word >> (lane * bits)) & mask) + bias;
}

template <unsigned int bits>
__device__ __forceinline__ void libmir_cuda_affine_qmm_bf16_impl(
    const unsigned short* input, const unsigned int* weight,
    const unsigned short* scales, const unsigned short* biases,
    unsigned short* output, unsigned int tokens, unsigned int input_features,
    unsigned int output_features, unsigned int group_size, unsigned int matrix_index) {
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
  const unsigned int words_per_row = input_features / (32u / bits);
  const unsigned int groups_per_row = input_features / group_size;
  wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (unsigned int feature_base = 0; feature_base < input_features; feature_base += tile) {
    for (unsigned int index = lane; index < tile * tile; index += 32u) {
      const unsigned int token_offset = index / tile;
      const unsigned int feature_offset = index % tile;
      const unsigned int token = token_base + token_offset;
      const unsigned int feature = feature_base + feature_offset;
      input_tiles[warp][index] = token < tokens
          ? reinterpret_cast<const __nv_bfloat16*>(input)[token * input_features + feature]
          : __float2bfloat16(0.0f);
    }
    for (unsigned int index = thread; index < tile * tile; index += 32u * warps) {
      const unsigned int output_offset = index / tile;
      const unsigned int feature_offset = index % tile;
      const unsigned int row = output_base + output_offset;
      const unsigned int feature = feature_base + feature_offset;
      float value = 0.0f;
      if (row < output_features) {
        value = libmir_cuda_qmm_weight<bits>(
            weight, scales, biases, matrix_index * output_features + row, feature,
            words_per_row, groups_per_row, group_size);
      }
      weight_tile[feature_offset + output_offset * tile] = __float2bfloat16(value);
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
    const unsigned int token_offset = index / tile;
    const unsigned int output_offset = index % tile;
    const unsigned int token = token_base + token_offset;
    const unsigned int row = output_base + output_offset;
    if (token < tokens && row < output_features) {
      reinterpret_cast<__nv_bfloat16*>(output)[token * output_features + row] =
          __float2bfloat16(output_tiles[warp][index]);
    }
  }
}

#define LIBMIR_CUDA_AFFINE_QMM(NAME, BITS)                                            \
  extern "C" __global__ void NAME(                                                   \
      const unsigned short* input, const unsigned int* weight,                        \
      const unsigned short* scales, const unsigned short* biases,                     \
      unsigned short* output, unsigned int tokens, unsigned int input_features,       \
      unsigned int output_features, unsigned int group_size, unsigned int matrix_index) { \
    libmir_cuda_affine_qmm_bf16_impl<BITS>(                                           \
        input, weight, scales, biases, output, tokens, input_features,                \
        output_features, group_size, matrix_index);                                   \
  }

LIBMIR_CUDA_AFFINE_QMM(libmir_cuda_affine_qmm_bf16_int4, 4)
LIBMIR_CUDA_AFFINE_QMM(libmir_cuda_affine_qmm_bf16_int8, 8)
