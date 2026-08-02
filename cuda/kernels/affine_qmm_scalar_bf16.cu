__device__ __forceinline__ float libmir_cuda_scalar_qmm_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short libmir_cuda_scalar_qmm_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

template <unsigned int bits, unsigned int values_per_thread>
__device__ __forceinline__ void libmir_cuda_affine_qmm_scalar_bf16_impl(
    const unsigned short* input, const unsigned int* weight,
    const unsigned short* scales, const unsigned short* biases,
    unsigned short* output, unsigned int tokens, unsigned int input_features,
    unsigned int output_features, unsigned int group_size, unsigned int matrix_index) {
  constexpr unsigned int token_tile = 8u;
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int words_per_thread = values_per_thread / values_per_word;
  constexpr unsigned int mask = (1u << bits) - 1u;
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int token_base = blockIdx.y * token_tile;
  if (row >= output_features) return;
  const unsigned int words_per_row = input_features / values_per_word;
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int matrix_row = matrix_index * output_features + row;
  const unsigned int weight_base = matrix_row * words_per_row;
  const unsigned int group_base = matrix_row * groups_per_row;
  float sums[token_tile] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (unsigned int input_base = threadIdx.x * values_per_thread;
       input_base < input_features; input_base += 32u * values_per_thread) {
    const unsigned int group = group_base + input_base / group_size;
    const float scale = libmir_cuda_scalar_qmm_bf16_to_float(scales[group]);
    const float bias = libmir_cuda_scalar_qmm_bf16_to_float(biases[group]);
#pragma unroll
    for (unsigned int packed = 0; packed < words_per_thread; ++packed) {
      const unsigned int word = weight[weight_base + input_base / values_per_word + packed];
#pragma unroll
      for (unsigned int lane = 0; lane < values_per_word; ++lane) {
        const unsigned int feature = input_base + packed * values_per_word + lane;
        const float quantized = static_cast<float>((word >> (lane * bits)) & mask);
        const float dequantized = scale * quantized + bias;
#pragma unroll
        for (unsigned int token_offset = 0; token_offset < token_tile; ++token_offset) {
          const unsigned int token = token_base + token_offset;
          if (token < tokens) {
            const float value = libmir_cuda_scalar_qmm_bf16_to_float(input[token * input_features + feature]);
            sums[token_offset] += value * dequantized;
          }
        }
      }
    }
  }
#pragma unroll
  for (unsigned int token_offset = 0; token_offset < token_tile; ++token_offset) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token_offset] += __shfl_down_sync(0xffffffffu, sums[token_offset], offset);
    }
    const unsigned int token = token_base + token_offset;
    if (threadIdx.x == 0 && token < tokens) {
      output[token * output_features + row] =
          libmir_cuda_scalar_qmm_float_to_bf16(sums[token_offset]);
    }
  }
}

#define LIBMIR_CUDA_AFFINE_QMM_SCALAR(NAME, BITS, VALUES)                             \
  extern "C" __global__ void NAME(                                                   \
      const unsigned short* input, const unsigned int* weight,                        \
      const unsigned short* scales, const unsigned short* biases,                     \
      unsigned short* output, unsigned int tokens, unsigned int input_features,       \
      unsigned int output_features, unsigned int group_size, unsigned int matrix_index) { \
    libmir_cuda_affine_qmm_scalar_bf16_impl<BITS, VALUES>(                            \
        input, weight, scales, biases, output, tokens, input_features,                \
        output_features, group_size, matrix_index);                                   \
  }

LIBMIR_CUDA_AFFINE_QMM_SCALAR(libmir_cuda_affine_qmm_scalar_bf16_int4, 4, 16)
LIBMIR_CUDA_AFFINE_QMM_SCALAR(libmir_cuda_affine_qmm_scalar_bf16_int8, 8, 8)

extern "C" __global__ void libmir_cuda_affine_qmm_scalar_bf16_fallback(
    const unsigned short* input, const unsigned int* weight,
    const unsigned short* scales, const unsigned short* biases,
    unsigned short* output, unsigned int tokens, unsigned int input_features,
    unsigned int output_features, unsigned int group_size, unsigned int matrix_index) {
  constexpr unsigned int bits = LIBMIR_AFFINE_BITS;
  constexpr unsigned int token_tile = 8u;
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int token_base = blockIdx.y * token_tile;
  if (row >= output_features) return;
  const unsigned int words_per_row =
      libmir_cuda_affine_words<bits>(input_features);
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int matrix_row = matrix_index * output_features + row;
  const unsigned int* row_weight = weight + matrix_row * words_per_row;
  const unsigned int group_base = matrix_row * groups_per_row;
  float sums[token_tile] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (unsigned int feature = threadIdx.x; feature < input_features; feature += 32u) {
    const unsigned int group = group_base + feature / group_size;
    const float scale = libmir_cuda_scalar_qmm_bf16_to_float(scales[group]);
    const float bias = libmir_cuda_scalar_qmm_bf16_to_float(biases[group]);
    const float quantized =
        static_cast<float>(libmir_cuda_affine_unpack<bits>(row_weight, feature));
    const float dequantized = scale * quantized + bias;
#pragma unroll
    for (unsigned int token_offset = 0; token_offset < token_tile; ++token_offset) {
      const unsigned int token = token_base + token_offset;
      if (token < tokens) {
        const float value = libmir_cuda_scalar_qmm_bf16_to_float(
            input[token * input_features + feature]);
        sums[token_offset] += value * dequantized;
      }
    }
  }
#pragma unroll
  for (unsigned int token_offset = 0; token_offset < token_tile; ++token_offset) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token_offset] +=
          __shfl_down_sync(0xffffffffu, sums[token_offset], offset);
    }
    const unsigned int token = token_base + token_offset;
    if (threadIdx.x == 0 && token < tokens) {
      output[token * output_features + row] =
          libmir_cuda_scalar_qmm_float_to_bf16(sums[token_offset]);
    }
  }
}
