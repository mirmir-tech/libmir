__device__ __forceinline__ float libmir_cuda_reduce_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short libmir_cuda_reduce_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

template <unsigned int bits, unsigned int values_per_thread>
__device__ __forceinline__ void libmir_cuda_selected_affine_reduce_bf16_impl(
    const unsigned short* input, const unsigned int* selected,
    const unsigned short* routing_weights, const unsigned int* weight,
    const unsigned short* scales, const unsigned short* biases, unsigned short* output,
    unsigned int input_features, unsigned int output_features, unsigned int group_size,
    unsigned int expert_count, unsigned int selected_count) {
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int words_per_thread = values_per_thread / values_per_word;
  constexpr unsigned int mask = (1u << bits) - 1u;
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int token = blockIdx.y;
  if (row >= output_features) return;
  input += token * selected_count * input_features;
  selected += token * selected_count;
  routing_weights += token * selected_count;
  output += token * output_features;
  const unsigned int words_per_row = input_features / values_per_word;
  const unsigned int groups_per_row = input_features / group_size;
  float reduced = 0.0f;

  for (unsigned int slot = 0; slot < selected_count; ++slot) {
    const unsigned int expert = selected[slot];
    float sum = 0.0f;
    if (expert < expert_count) {
      const unsigned int expert_row = expert * output_features + row;
      const unsigned int weight_base = expert_row * words_per_row;
      const unsigned int group_base = expert_row * groups_per_row;
      const unsigned int input_slot = slot * input_features;
      for (unsigned int input_base = threadIdx.x * values_per_thread;
           input_base < input_features; input_base += 32u * values_per_thread) {
        const unsigned int group = group_base + input_base / group_size;
        const float scale = libmir_cuda_reduce_bf16_to_float(scales[group]);
        const float bias = libmir_cuda_reduce_bf16_to_float(biases[group]);
#pragma unroll
        for (unsigned int packed = 0; packed < words_per_thread; ++packed) {
          const unsigned int word = weight[weight_base + input_base / values_per_word + packed];
#pragma unroll
          for (unsigned int lane = 0; lane < values_per_word; ++lane) {
            const unsigned int input_index = input_slot + input_base + packed * values_per_word + lane;
            const float value = libmir_cuda_reduce_bf16_to_float(input[input_index]);
            const float quantized = static_cast<float>((word >> (lane * bits)) & mask);
            sum += value * (scale * quantized + bias);
          }
        }
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0 && expert < expert_count) {
      const float projected = libmir_cuda_reduce_bf16_to_float(libmir_cuda_reduce_float_to_bf16(sum));
      reduced += projected * libmir_cuda_reduce_bf16_to_float(routing_weights[slot]);
    }
  }
  if (threadIdx.x == 0) output[row] = libmir_cuda_reduce_float_to_bf16(reduced);
}

#define LIBMIR_CUDA_SELECTED_REDUCE(NAME, BITS, VALUES)                               \
  extern "C" __global__ void NAME(                                                   \
      const unsigned short* input, const unsigned int* selected,                      \
      const unsigned short* routing_weights, const unsigned int* weight,              \
      const unsigned short* scales, const unsigned short* biases,                     \
      unsigned short* output, unsigned int input_features, unsigned int output_features, \
      unsigned int group_size, unsigned int expert_count, unsigned int selected_count) { \
    libmir_cuda_selected_affine_reduce_bf16_impl<BITS, VALUES>(                       \
        input, selected, routing_weights, weight, scales, biases, output,             \
        input_features, output_features, group_size, expert_count, selected_count);   \
  }

LIBMIR_CUDA_SELECTED_REDUCE(libmir_cuda_selected_affine_reduce_bf16_int4, 4, 16)
LIBMIR_CUDA_SELECTED_REDUCE(libmir_cuda_selected_affine_reduce_bf16_int8, 8, 8)
