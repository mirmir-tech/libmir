#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_l2_normalize_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int elements, float epsilon) {
  float sum = 0.0f;
  for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x) {
    const float value = __bfloat162float(input[index]);
    sum += value * value;
  }
  __shared__ float reductions[8];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x / 32u;
  if (lane == 0u) reductions[warp] = sum;
  __syncthreads();
  if (warp == 0u) {
    sum = lane < 8u ? reductions[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0u) reductions[0] = rsqrtf(fmaxf(sum, epsilon));
  }
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < elements; index += blockDim.x)
    output[index] = __float2bfloat16_rn(__bfloat162float(input[index]) * reductions[0]);
}
