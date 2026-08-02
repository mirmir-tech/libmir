#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_gated_nvfp4_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

__device__ __forceinline__ float libmir_gated_nvfp4_activate(
    float value, unsigned int activation) {
  if (activation == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

extern "C" __global__ void libmir_cuda_gated_nvfp4(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const float* global_scale, unsigned char* packed, unsigned char* scales,
    unsigned int rows, unsigned int columns, unsigned int activation) {
  const unsigned int group = blockIdx.x;
  const unsigned int lane = threadIdx.x;
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int row = group / blocks_per_row;
  const unsigned int block = group % blocks_per_row;
  if (row >= rows) return;
  const unsigned int feature = block * 16u + lane;
  float value = 0.0f;
  if (lane < 16u) {
    const unsigned int index = row * columns + feature;
    const float activated = __bfloat162float(__float2bfloat16_rn(
        libmir_gated_nvfp4_activate(
            __bfloat162float(gate[index]), activation)));
    value = __bfloat162float(__float2bfloat16_rn(
        activated * __bfloat162float(up[index])));
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
    scales[libmir_gated_nvfp4_scale_offset(row, block, columns)] = encoded.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const float first_activated = __bfloat162float(__float2bfloat16_rn(
        libmir_gated_nvfp4_activate(
            __bfloat162float(gate[first]), activation)));
    const float second_activated = __bfloat162float(__float2bfloat16_rn(
        libmir_gated_nvfp4_activate(
            __bfloat162float(gate[first + 1u]), activation)));
    const float first_value = __bfloat162float(__float2bfloat16_rn(
        first_activated * __bfloat162float(up[first])));
    const float second_value = __bfloat162float(__float2bfloat16_rn(
        second_activated * __bfloat162float(up[first + 1u])));
    __nv_fp4x2_e2m1 converted(
        make_float2(first_value / divisor, second_value / divisor));
    packed[row * columns / 2u + block * 8u + lane] = converted.__x;
  }
}
