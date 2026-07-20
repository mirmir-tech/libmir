#include <cuda_bf16.h>

template <unsigned int bits>
__device__ __forceinline__ void libmir_cuda_affine_embedding_impl(
    const unsigned int* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* biases, const unsigned int* selected,
    __nv_bfloat16* output, unsigned int selected_start,
    unsigned int tokens, unsigned int vocab, unsigned int hidden,
    unsigned int group_size, float output_scale) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int token = selected[selected_start + index / hidden];
  if (token >= vocab) return;
  const unsigned int column = index % hidden;
  constexpr unsigned int values_per_word = 32u / bits;
  constexpr unsigned int mask = (1u << bits) - 1u;
  const unsigned int words_per_row = hidden / values_per_word;
  const unsigned int groups_per_row = hidden / group_size;
  const unsigned int word = weight[token * words_per_row + column / values_per_word];
  const unsigned int quantized = (word >> ((column % values_per_word) * bits)) & mask;
  const unsigned int group = token * groups_per_row + column / group_size;
  const float value = __bfloat162float(scales[group]) * quantized
      + __bfloat162float(biases[group]);
  output[index] = __float2bfloat16_rn(value * output_scale);
}

extern "C" __global__ void libmir_cuda_affine_embedding_bf16_int4(
    const unsigned int* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* biases, const unsigned int* selected,
    __nv_bfloat16* output, unsigned int selected_start,
    unsigned int tokens, unsigned int vocab, unsigned int hidden,
    unsigned int group_size, float output_scale) {
  libmir_cuda_affine_embedding_impl<4>(
      weight, scales, biases, selected, output, selected_start,
      tokens, vocab, hidden, group_size, output_scale);
}

extern "C" __global__ void libmir_cuda_affine_embedding_bf16_int8(
    const unsigned int* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* biases, const unsigned int* selected,
    __nv_bfloat16* output, unsigned int selected_start,
    unsigned int tokens, unsigned int vocab, unsigned int hidden,
    unsigned int group_size, float output_scale) {
  libmir_cuda_affine_embedding_impl<8>(
      weight, scales, biases, selected, output, selected_start,
      tokens, vocab, hidden, group_size, output_scale);
}
