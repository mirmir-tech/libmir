#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_rms_nvfp4_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

extern "C" __global__ void libmir_cuda_rms_inverse_bf16(
    const __nv_bfloat16* input, float* inverse, unsigned int rows,
    unsigned int columns, float epsilon) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
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
    if (threadIdx.x == 0u) inverse[row] = rsqrtf(sum / columns + epsilon);
  }
}

extern "C" __global__ void libmir_cuda_rms_norm_nvfp4(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const float* inverse, const float* global_scale, unsigned char* packed,
    unsigned char* scales, unsigned int rows, unsigned int columns) {
  const unsigned int group = blockIdx.x;
  const unsigned int lane = threadIdx.x;
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int row = group / blocks_per_row;
  const unsigned int block = group % blocks_per_row;
  if (row >= rows) return;
  const unsigned int feature = block * 16u + lane;
  float value = 0.0f;
  if (lane < 16u) {
    const unsigned int offset = row * columns + feature;
    value = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[offset]) * inverse[row] *
        __bfloat162float(weight[feature])));
  }
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  __shared__ float divisor;
  if (lane == 0u) {
    const float global = global_scale[0];
    __nv_fp8_e4m3 encoded(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(encoded) * global;
    scales[libmir_rms_nvfp4_scale_offset(row, block, columns)] = encoded.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const unsigned int feature_first = block * 16u + lane * 2u;
    const float2 pair = make_float2(
        __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(input[first]) * inverse[row] *
            __bfloat162float(weight[feature_first]))) /
            divisor,
        __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(input[first + 1u]) * inverse[row] *
            __bfloat162float(weight[feature_first + 1u]))) /
            divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[row * columns / 2u + block * 8u + lane] = converted.__x;
  }
}
