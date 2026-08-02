#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
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

template <typename Encoded>
__device__ __forceinline__ float decode(unsigned char raw) {
  Encoded value;
  value.__x = raw;
  return static_cast<float>(value);
}

template <typename Scale>
__device__ __forceinline__ void direct_fp8_embedding(
    const unsigned char* weight, const Scale* scales,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int selected_start, unsigned int tokens, unsigned int vocab,
    unsigned int hidden, unsigned int scale_rows, unsigned int scale_columns,
    unsigned int scale_row_size, unsigned int scale_group_size,
    unsigned int inverse_scale, float output_scale, unsigned int e5m2) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = tokens * hidden;
  if (index >= total) return;
  const unsigned int output_row = index / hidden;
  const unsigned int column = index % hidden;
  const unsigned int row = selected[selected_start + output_row];
  if (row >= vocab) return;
  const unsigned char raw = weight[row * hidden + column];
  const float value = e5m2 != 0u ? decode<__nv_fp8_e5m2>(raw)
                                 : decode<__nv_fp8_e4m3>(raw);
  const unsigned int scale_row = scale_rows == 1u ? 0u : row / scale_row_size;
  const unsigned int scale_column = column / scale_group_size;
  const float scale = scale_value(scales[scale_row * scale_columns + scale_column]);
  const float scaled = inverse_scale == 0u ? value * scale : value / scale;
  output[index] = __float2bfloat16_rn(scaled * output_scale);
}
}  // namespace

extern "C" __global__ void libmir_cuda_direct_fp8_embedding_f32_scale(
    const unsigned char* weight, const float* scales,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int selected_start, unsigned int tokens, unsigned int vocab,
    unsigned int hidden, unsigned int scale_rows, unsigned int scale_columns,
    unsigned int scale_row_size, unsigned int scale_group_size,
    unsigned int inverse_scale, float output_scale, unsigned int e5m2) {
  direct_fp8_embedding(weight, scales, selected, output, selected_start, tokens,
                       vocab, hidden, scale_rows, scale_columns, scale_row_size,
                       scale_group_size, inverse_scale, output_scale, e5m2);
}

extern "C" __global__ void libmir_cuda_direct_fp8_embedding_bf16_scale(
    const unsigned char* weight, const __nv_bfloat16* scales,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int selected_start, unsigned int tokens, unsigned int vocab,
    unsigned int hidden, unsigned int scale_rows, unsigned int scale_columns,
    unsigned int scale_row_size, unsigned int scale_group_size,
    unsigned int inverse_scale, float output_scale, unsigned int e5m2) {
  direct_fp8_embedding(weight, scales, selected, output, selected_start, tokens,
                       vocab, hidden, scale_rows, scale_columns, scale_row_size,
                       scale_group_size, inverse_scale, output_scale, e5m2);
}
