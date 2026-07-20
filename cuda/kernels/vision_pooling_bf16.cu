#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_pool_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int grid_height, unsigned int grid_width,
    unsigned int hidden, unsigned int kernel) {
  const unsigned int output_height = grid_height / kernel;
  const unsigned int output_width = grid_width / kernel;
  const unsigned int elements = output_height * output_width * hidden;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int column = index % hidden;
  const unsigned int token = index / hidden;
  const unsigned int output_x = token % output_width;
  const unsigned int output_y = token / output_width;
  float sum = 0.0f;
  for (unsigned int inner_y = 0; inner_y < kernel; ++inner_y) {
    for (unsigned int inner_x = 0; inner_x < kernel; ++inner_x) {
      const unsigned int y = output_y * kernel + inner_y;
      const unsigned int x = output_x * kernel + inner_x;
      sum += __bfloat162float(input[(y * grid_width + x) * hidden + column]);
    }
  }
  const float mean = sum / static_cast<float>(kernel * kernel);
  output[index] = __float2bfloat16_rn(mean * sqrtf(static_cast<float>(hidden)));
}

extern "C" __global__ void libmir_cuda_vision_standardize_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    const __nv_bfloat16* scale, __nv_bfloat16* output,
    unsigned int tokens, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int column = index % hidden;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(
      (value - __bfloat162float(bias[column])) * __bfloat162float(scale[column]));
}

extern "C" __global__ void libmir_cuda_vision_standardize_f32(
    const __nv_bfloat16* input, const float* bias,
    const float* scale, __nv_bfloat16* output,
    unsigned int tokens, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int column = index % hidden;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(
      (value - bias[column]) * scale[column]);
}
