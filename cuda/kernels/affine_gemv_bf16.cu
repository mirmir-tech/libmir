__device__ __forceinline__ float libmir_cuda_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short libmir_cuda_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

template <unsigned int bits, unsigned int values_per_thread>
__device__ __forceinline__ void libmir_cuda_affine_gemv_bf16_impl(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* output,
    unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size,
    unsigned int matrix_index) {
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int words_per_thread = values_per_thread / values_per_word;
  constexpr unsigned int rows_per_block = 8u;
  const unsigned int row = blockIdx.x * rows_per_block + threadIdx.y;
  if (row >= output_features) {
    return;
  }

  const unsigned int words_per_row = input_features / values_per_word;
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int matrix_row = matrix_index * output_features + row;
  const unsigned int weight_base = matrix_row * words_per_row;
  const unsigned int group_base = matrix_row * groups_per_row;
  constexpr unsigned int mask = (1u << bits) - 1u;
  float sum = 0.0f;

  for (unsigned int input_base = threadIdx.x * values_per_thread;
       input_base < input_features;
       input_base += 32u * values_per_thread) {
    const unsigned int group = input_base / group_size;
    const float scale = libmir_cuda_bf16_to_float(scales[group_base + group]);
    const float bias = libmir_cuda_bf16_to_float(biases[group_base + group]);
#pragma unroll
    for (unsigned int packed = 0; packed < words_per_thread; ++packed) {
      const unsigned int word = weight[
          weight_base + input_base / values_per_word + packed];
#pragma unroll
      for (unsigned int lane = 0; lane < values_per_word; ++lane) {
        const unsigned int input_index =
            input_base + packed * values_per_word + lane;
        const float quantized =
            static_cast<float>((word >> (lane * bits)) & mask);
        const float dequantized = scale * quantized + bias;
        sum += libmir_cuda_bf16_to_float(input[input_index]) * dequantized;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) {
    output[row] = libmir_cuda_float_to_bf16(sum);
  }
}

extern "C" __global__ void libmir_cuda_affine_gemv_bf16_int4(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* output,
    unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size,
    unsigned int matrix_index) {
  libmir_cuda_affine_gemv_bf16_impl<4, 16>(
      input,
      weight,
      scales,
      biases,
      output,
      input_features,
      output_features,
      group_size,
      matrix_index);
}

extern "C" __global__ void libmir_cuda_affine_gemv_bf16_int8(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* output,
    unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size,
    unsigned int matrix_index) {
  libmir_cuda_affine_gemv_bf16_impl<8, 8>(
      input,
      weight,
      scales,
      biases,
      output,
      input_features,
      output_features,
      group_size,
      matrix_index);
}

extern "C" __global__ void libmir_cuda_affine_gemv_bf16_fallback(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* output,
    unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size,
    unsigned int matrix_index) {
  const unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= output_features) {
    return;
  }
  constexpr unsigned int bits = LIBMIR_AFFINE_BITS;
  const unsigned int words_per_row =
      libmir_cuda_affine_words<bits>(input_features);
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int matrix_row = matrix_index * output_features + row;
  const unsigned int* row_weight = weight + matrix_row * words_per_row;
  const unsigned int group_base = matrix_row * groups_per_row;
  float sum = 0.0f;
  for (unsigned int feature = threadIdx.x; feature < input_features; feature += 32u) {
    const unsigned int group = group_base + feature / group_size;
    const float scale = libmir_cuda_bf16_to_float(scales[group]);
    const float bias = libmir_cuda_bf16_to_float(biases[group]);
    const float quantized =
        static_cast<float>(libmir_cuda_affine_unpack<bits>(row_weight, feature));
    sum += libmir_cuda_bf16_to_float(input[feature]) *
        (scale * quantized + bias);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) {
    output[row] = libmir_cuda_float_to_bf16(sum);
  }
}
