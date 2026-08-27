#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_cuda_nvfp4_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

extern "C" __global__ void libmir_cuda_nvfp4_prepare_weight(
    const unsigned char* source_weight, const unsigned char* source_scales,
    unsigned char* weight, unsigned char* scales, unsigned int rows,
    unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int packed_elements = rows * columns / 2u;
  if (index < packed_elements) weight[index] = source_weight[index];
  const unsigned int scale_elements = rows * columns / 16u;
  if (index < scale_elements) {
    const unsigned int blocks_per_row = columns / 16u;
    const unsigned int row = index / blocks_per_row;
    const unsigned int block = index % blocks_per_row;
    scales[libmir_cuda_nvfp4_scale_offset(row, block, columns)] =
        source_scales[index];
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_prepare_selected_weight(
    const unsigned char* source_weight, const unsigned char* source_scales,
    const float* source_input_scales, const float* source_weight_scales,
    const unsigned int* selected, unsigned char* weight, unsigned char* scales,
    float* input_scale, float* weight_scale, unsigned int experts,
    unsigned int rank, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int expert = selected[rank];
  if (expert >= experts) return;
  const unsigned int packed_elements = rows * columns / 2u;
  if (index < packed_elements) {
    weight[index] = source_weight[expert * packed_elements + index];
  }
  const unsigned int scale_elements = rows * columns / 16u;
  if (index < scale_elements) {
    const unsigned int blocks_per_row = columns / 16u;
    const unsigned int row = index / blocks_per_row;
    const unsigned int block = index % blocks_per_row;
    scales[libmir_cuda_nvfp4_scale_offset(row, block, columns)] =
        source_scales[expert * scale_elements + index];
  }
  if (index == 0u) {
    input_scale[0] = source_input_scales[expert];
    weight_scale[0] = source_weight_scales[expert];
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_scale_selected_bf16(
    const __nv_bfloat16* input, const float* input_scale,
    const float* weight_scale, __nv_bfloat16* output,
    unsigned int output_offset, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float scale = input_scale[0] * weight_scale[0];
    output[output_offset + index] =
        __float2bfloat16_rn(__bfloat162float(input[index]) * scale);
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_selected_bf16(
    const unsigned short* input, unsigned int input_offset,
    const float* global_scale, unsigned char* packed, unsigned char* scales,
    unsigned int columns) {
  const unsigned int block = blockIdx.x;
  const unsigned int lane = threadIdx.x;
  const unsigned int feature = block * 16u + lane;
  float value = 0.0f;
  if (lane < 16u) {
    value = __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(input)[input_offset + feature]);
  }
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  __shared__ float divisor;
  if (lane == 0u) {
    const float global = global_scale[0];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    scales[libmir_cuda_nvfp4_scale_offset(0u, block, columns)] = scale.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = input_offset + block * 16u + lane * 2u;
    const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(input);
    const float2 pair = make_float2(
        __bfloat162float(values[first]) / divisor,
        __bfloat162float(values[first + 1u]) / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[block * 8u + lane] = converted.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_bf16(
    const unsigned short* input, const float* global_scale,
    unsigned char* packed, unsigned char* scales, unsigned int rows,
    unsigned int columns) {
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int work = blockIdx.x * (blockDim.x / 32u) + warp;
  const unsigned int linear_block = work * 2u + lane / 16u;
  const bool valid = linear_block < rows * blocks_per_row;
  const unsigned int row = linear_block / blocks_per_row;
  const unsigned int block = linear_block % blocks_per_row;
  const unsigned int half_lane = lane % 16u;
  float value = 0.0f;
  if (valid) {
    value = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(input)[
        row * columns + block * 16u + half_lane]);
  }
  float amax = fabsf(value);
  for (int offset = 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset, 16));
  }
  float divisor = 0.0f;
  if (valid && half_lane == 0u) {
    const float global = global_scale[0];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    scales[libmir_cuda_nvfp4_scale_offset(row, block, columns)] = scale.__x;
  }
  divisor = __shfl_sync(0xffffffffu, divisor, 0, 16);
  const unsigned int pair_lane = (half_lane & 7u) * 2u;
  const float first_value = __shfl_sync(0xffffffffu, value, pair_lane, 16);
  const float second_value = __shfl_sync(0xffffffffu, value, pair_lane + 1u, 16);
  if (valid && half_lane < 8u) {
    const float2 pair = make_float2(first_value / divisor, second_value / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[row * columns / 2u + block * 8u + half_lane] = converted.__x;
  }
}
