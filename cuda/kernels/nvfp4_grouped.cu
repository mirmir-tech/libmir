#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_grouped_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

__device__ __forceinline__ float libmir_grouped_activation(
    float value, unsigned int activation) {
  if (activation == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
         (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

extern "C" __global__ void libmir_cuda_nvfp4_prepare_bank_scales(
    const unsigned char* source, unsigned char* output,
    unsigned int experts, unsigned int rows, unsigned int columns,
    unsigned int output_stride) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int per_expert = rows * columns / 16u;
  if (index >= experts * per_expert) return;
  const unsigned int expert = index / per_expert;
  const unsigned int local = index % per_expert;
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int row = local / blocks_per_row;
  const unsigned int block = local % blocks_per_row;
  output[expert * output_stride +
         libmir_grouped_scale_offset(row, block, columns)] = source[index];
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_indexed_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const float* global_scales, unsigned char* packed,
    unsigned char* scales, unsigned int groups,
    unsigned int selected_count, unsigned int input_rows,
    unsigned int columns, unsigned int scale_stride,
    unsigned int ranked) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int group = blockIdx.x / blocks_per_row;
  const unsigned int block = blockIdx.x % blocks_per_row;
  const unsigned int lane = threadIdx.x;
  if (group >= groups) return;
  const unsigned int row = ranked == 0u ? group / selected_count : group;
  if (row >= input_rows) return;
  const unsigned int expert = selected[group];
  const unsigned int feature = block * 16u + lane;
  float value = lane < 16u ? __bfloat162float(input[row * columns + feature]) : 0.0f;
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  __shared__ float divisor;
  if (lane == 0u) {
    const float global = global_scales[expert];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    scales[group * scale_stride + libmir_grouped_scale_offset(0u, block, columns)] = scale.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const float2 pair = make_float2(
        __bfloat162float(input[first]) / divisor,
        __bfloat162float(input[first + 1u]) / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[group * columns / 2u + block * 8u + lane] = converted.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_indexed_pair_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const float* left_globals, const float* right_globals,
    unsigned char* left_packed, unsigned char* right_packed,
    unsigned char* left_scales, unsigned char* right_scales,
    unsigned int groups, unsigned int selected_count, unsigned int input_rows,
    unsigned int columns, unsigned int scale_stride) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int group = blockIdx.x / blocks_per_row;
  const unsigned int block = blockIdx.x % blocks_per_row;
  const unsigned int lane = threadIdx.x;
  if (group >= groups) return;
  const unsigned int row = group / selected_count;
  if (row >= input_rows) return;
  const unsigned int expert = selected[group];
  const unsigned int feature = block * 16u + lane;
  const float value = lane < 16u
                          ? __bfloat162float(input[row * columns + feature])
                          : 0.0f;
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  __shared__ float divisors[2];
  if (lane == 0u) {
    const float globals[2] = {left_globals[expert], right_globals[expert]};
    __nv_fp8_e4m3 scales[2] = {
        __nv_fp8_e4m3(amax == 0.0f ? 1.0f : amax / (6.0f * globals[0])),
        __nv_fp8_e4m3(amax == 0.0f ? 1.0f : amax / (6.0f * globals[1]))};
    divisors[0] = static_cast<float>(scales[0]) * globals[0];
    divisors[1] = static_cast<float>(scales[1]) * globals[1];
    const unsigned int offset =
        group * scale_stride + libmir_grouped_scale_offset(0u, block, columns);
    left_scales[offset] = scales[0].__x;
    right_scales[offset] = scales[1].__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const float2 pair = make_float2(__bfloat162float(input[first]),
                                    __bfloat162float(input[first + 1u]));
    __nv_fp4x2_e2m1 left(
        make_float2(pair.x / divisors[0], pair.y / divisors[0]));
    __nv_fp4x2_e2m1 right(
        make_float2(pair.x / divisors[1], pair.y / divisors[1]));
    const unsigned int output = group * columns / 2u + block * 8u + lane;
    left_packed[output] = left.__x;
    right_packed[output] = right.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_gated_quantize_indexed_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const unsigned int* selected, const float* global_scales,
    unsigned char* packed, unsigned char* scales, unsigned int groups,
    unsigned int columns, unsigned int scale_stride,
    unsigned int activation) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int group = blockIdx.x / blocks_per_row;
  const unsigned int block = blockIdx.x % blocks_per_row;
  const unsigned int lane = threadIdx.x;
  if (group >= groups) return;
  const unsigned int feature = block * 16u + lane;
  __shared__ float values[16];
  __shared__ float divisor;
  float value = 0.0f;
  if (lane < 16u) {
    const unsigned int index = group * columns + feature;
    const float activated = libmir_grouped_activation(
        __bfloat162float(gate[index]), activation);
    value = __bfloat162float(__float2bfloat16_rn(
        activated * __bfloat162float(up[index])));
    values[lane] = value;
  }
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  if (lane == 0u) {
    const float global = global_scales[selected[group]];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    scales[group * scale_stride +
           libmir_grouped_scale_offset(0u, block, columns)] = scale.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const float2 pair = make_float2(
        values[lane * 2u] / divisor, values[lane * 2u + 1u] / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[group * columns / 2u + block * 8u + lane] = converted.__x;
  }
}
