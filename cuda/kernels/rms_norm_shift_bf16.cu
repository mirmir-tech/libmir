#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_rms_norm_shift_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    float epsilon, float weight_shift) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warps[8];
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x / 32u] = sum;
  __syncthreads();
  if (threadIdx.x < 32u) {
    sum = threadIdx.x < 8u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0u) warps[0] = rsqrtf(sum / columns + epsilon);
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float scale = __bfloat162float(weight[column]) + weight_shift;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * warps[0] * scale);
  }
}

extern "C" __global__ void libmir_cuda_residual_rms_norm_shift_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* update,
    const __nv_bfloat16* weight, __nv_bfloat16* residual,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    float epsilon, float weight_shift) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const __nv_bfloat16 rounded = __float2bfloat16_rn(
        __bfloat162float(input[index]) + __bfloat162float(update[index]));
    residual[index] = rounded;
    const float value = __bfloat162float(rounded);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warps[8];
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x / 32u] = sum;
  __syncthreads();
  if (threadIdx.x < 32u) {
    sum = threadIdx.x < 8u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0u) warps[0] = rsqrtf(sum / columns + epsilon);
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float scale = __bfloat162float(weight[column]) + weight_shift;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(residual[index]) * warps[0] * scale);
  }
}
