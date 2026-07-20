#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_sigmoid_multiply_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns) return;
  const float gate_value = __bfloat162float(gate[index / columns]);
  const float multiplier = 1.0f / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16_rn(
      multiplier * __bfloat162float(input[index]));
}

extern "C" __global__ void libmir_cuda_sigmoid_multiply_elementwise_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float multiplier = 1.0f / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16_rn(
      multiplier * __bfloat162float(input[index]));
}
