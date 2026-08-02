__device__ __forceinline__ float libmir_cuda_pair_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short libmir_cuda_pair_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

template <unsigned int bits, unsigned int values_per_thread>
__device__ __forceinline__ void libmir_cuda_selected_affine_pair_bf16_impl(
    const unsigned short* input,
    const unsigned int* selected,
    const unsigned int* gate_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    const unsigned int* up_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    unsigned short* gate_output,
    unsigned short* up_output,
    unsigned int input_features,
    unsigned int output_features,
    unsigned int group_size,
    unsigned int expert_count) {
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int words_per_thread = values_per_thread / values_per_word;
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int slot = blockIdx.y;
  const unsigned int token = blockIdx.z;
  if (row >= output_features) {
    return;
  }
  input += token * input_features;
  const unsigned int selected_index = token * gridDim.y + slot;
  const unsigned int expert = selected[selected_index];
  if (expert >= expert_count) {
    if (threadIdx.x == 0) {
      gate_output[selected_index * output_features + row] = 0;
      up_output[selected_index * output_features + row] = 0;
    }
    return;
  }

  const unsigned int words_per_row = input_features / values_per_word;
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int expert_row = expert * output_features + row;
  const unsigned int weight_base = expert_row * words_per_row;
  const unsigned int group_base = expert_row * groups_per_row;
  constexpr unsigned int mask = (1u << bits) - 1u;
  float gate_sum = 0.0f;
  float up_sum = 0.0f;

  for (unsigned int input_base = threadIdx.x * values_per_thread;
       input_base < input_features;
       input_base += 32u * values_per_thread) {
    const unsigned int group = group_base + input_base / group_size;
    const float gate_scale = libmir_cuda_pair_bf16_to_float(gate_scales[group]);
    const float gate_bias = libmir_cuda_pair_bf16_to_float(gate_biases[group]);
    const float up_scale = libmir_cuda_pair_bf16_to_float(up_scales[group]);
    const float up_bias = libmir_cuda_pair_bf16_to_float(up_biases[group]);
#pragma unroll
    for (unsigned int packed = 0; packed < words_per_thread; ++packed) {
      const unsigned int word_index = weight_base + input_base / values_per_word + packed;
      const unsigned int gate_word = gate_weight[word_index];
      const unsigned int up_word = up_weight[word_index];
#pragma unroll
      for (unsigned int lane = 0; lane < values_per_word; ++lane) {
        const unsigned int input_index = input_base + packed * values_per_word + lane;
        const float value = libmir_cuda_pair_bf16_to_float(input[input_index]);
        const float gate_quantized = static_cast<float>((gate_word >> (lane * bits)) & mask);
        const float up_quantized = static_cast<float>((up_word >> (lane * bits)) & mask);
        gate_sum += value * (gate_scale * gate_quantized + gate_bias);
        up_sum += value * (up_scale * up_quantized + up_bias);
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
    up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
  }
  if (threadIdx.x == 0) {
    const unsigned int output_index = selected_index * output_features + row;
    gate_output[output_index] = libmir_cuda_pair_float_to_bf16(gate_sum);
    up_output[output_index] = libmir_cuda_pair_float_to_bf16(up_sum);
  }
}

#define LIBMIR_CUDA_SELECTED_PAIR(NAME, BITS, VALUES)                                  \
  extern "C" __global__ void NAME(                                                    \
      const unsigned short* input, const unsigned int* selected,                       \
      const unsigned int* gate_weight, const unsigned short* gate_scales,              \
      const unsigned short* gate_biases, const unsigned int* up_weight,                 \
      const unsigned short* up_scales, const unsigned short* up_biases,                 \
      unsigned short* gate_output, unsigned short* up_output,                           \
      unsigned int input_features, unsigned int output_features,                        \
      unsigned int group_size, unsigned int expert_count) {                             \
    libmir_cuda_selected_affine_pair_bf16_impl<BITS, VALUES>(                           \
        input, selected, gate_weight, gate_scales, gate_biases, up_weight, up_scales,   \
        up_biases, gate_output, up_output, input_features, output_features, group_size, \
        expert_count);                                                                  \
  }

LIBMIR_CUDA_SELECTED_PAIR(libmir_cuda_selected_affine_pair_bf16_int4, 4, 16)
LIBMIR_CUDA_SELECTED_PAIR(libmir_cuda_selected_affine_pair_bf16_int8, 8, 8)

extern "C" __global__ void libmir_cuda_selected_affine_pair_bf16_fallback(
    const unsigned short* input, const unsigned int* selected,
    const unsigned int* gate_weight, const unsigned short* gate_scales,
    const unsigned short* gate_biases, const unsigned int* up_weight,
    const unsigned short* up_scales, const unsigned short* up_biases,
    unsigned short* gate_output, unsigned short* up_output,
    unsigned int input_features, unsigned int output_features,
    unsigned int group_size, unsigned int expert_count) {
  constexpr unsigned int bits = LIBMIR_AFFINE_BITS;
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int slot = blockIdx.y;
  const unsigned int token = blockIdx.z;
  if (row >= output_features) return;
  input += token * input_features;
  const unsigned int selected_index = token * gridDim.y + slot;
  const unsigned int expert = selected[selected_index];
  if (expert >= expert_count) {
    if (threadIdx.x == 0) {
      gate_output[selected_index * output_features + row] = 0;
      up_output[selected_index * output_features + row] = 0;
    }
    return;
  }
  const unsigned int words_per_row =
      libmir_cuda_affine_words<bits>(input_features);
  const unsigned int groups_per_row = input_features / group_size;
  const unsigned int expert_row = expert * output_features + row;
  const unsigned int* gate_row = gate_weight + expert_row * words_per_row;
  const unsigned int* up_row = up_weight + expert_row * words_per_row;
  const unsigned int group_base = expert_row * groups_per_row;
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (unsigned int feature = threadIdx.x; feature < input_features; feature += 32u) {
    const unsigned int group = group_base + feature / group_size;
    const float value = libmir_cuda_pair_bf16_to_float(input[feature]);
    const float gate_quantized =
        static_cast<float>(libmir_cuda_affine_unpack<bits>(gate_row, feature));
    const float up_quantized =
        static_cast<float>(libmir_cuda_affine_unpack<bits>(up_row, feature));
    gate_sum += value * (
        libmir_cuda_pair_bf16_to_float(gate_scales[group]) * gate_quantized +
        libmir_cuda_pair_bf16_to_float(gate_biases[group]));
    up_sum += value * (
        libmir_cuda_pair_bf16_to_float(up_scales[group]) * up_quantized +
        libmir_cuda_pair_bf16_to_float(up_biases[group]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
    up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
  }
  if (threadIdx.x == 0) {
    const unsigned int output_index = selected_index * output_features + row;
    gate_output[output_index] = libmir_cuda_pair_float_to_bf16(gate_sum);
    up_output[output_index] = libmir_cuda_pair_float_to_bf16(up_sum);
  }
}
