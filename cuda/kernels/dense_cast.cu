#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void libmir_cuda_dense_f16_to_bf16(
    const __half* input, __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(__half2float(input[index]));
  }
}

extern "C" __global__ void libmir_cuda_dense_f32_to_bf16(
    const float* input, __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
}
