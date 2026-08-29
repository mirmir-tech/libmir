#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_cuda_rms_norm_shift_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

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

extern "C" __global__ void libmir_cuda_residual_rms_norm_shift_nvfp4_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* update,
    const __nv_bfloat16* weight, const float* global_scale,
    __nv_bfloat16* residual, __nv_bfloat16* output,
    unsigned char* packed, unsigned char* scales, unsigned int rows,
    unsigned int columns, float epsilon, float weight_shift) {
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

  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int blocks = columns / 16u;
  for (unsigned int block = warp; block < blocks; block += 8u) {
    const unsigned int column = block * 16u + lane;
    float value = 0.0f;
    if (lane < 16u) {
      const unsigned int index = row * columns + column;
      const float norm_scale = __bfloat162float(weight[column]) + weight_shift;
      const __nv_bfloat16 rounded = __float2bfloat16_rn(
          __bfloat162float(residual[index]) * warps[0] * norm_scale);
      output[index] = rounded;
      value = __bfloat162float(rounded);
    }
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    float divisor = 0.0f;
    if (lane == 0u) {
      const float global = global_scale[0];
      __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
      divisor = static_cast<float>(scale) * global;
      scales[libmir_cuda_rms_norm_shift_scale_offset(row, block, columns)] = scale.__x;
    }
    divisor = __shfl_sync(0xffffffffu, divisor, 0);
    const unsigned int pair_lane = (lane & 7u) * 2u;
    const float first = __shfl_sync(0xffffffffu, value, pair_lane);
    const float second = __shfl_sync(0xffffffffu, value, pair_lane + 1u);
    if (lane < 8u) {
      const float2 pair = make_float2(first / divisor, second / divisor);
      __nv_fp4x2_e2m1 converted(pair);
      packed[row * columns / 2u + block * 8u + lane] = converted.__x;
    }
  }
}
