#include <cuda_bf16.h>

namespace {
__device__ __constant__ float kMxFp4Values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
}

extern "C" __global__ void libmir_cuda_mxfp4_bf16_linear(
    const __nv_bfloat16* input, const unsigned char* weight,
    const unsigned char* scales, const __nv_bfloat16* bias,
    __nv_bfloat16* output, unsigned int tokens, unsigned int rows,
    unsigned int columns, unsigned int has_bias) {
  const unsigned int token = blockIdx.y;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int row = blockIdx.x * 8u + warp;
  if (token >= tokens || row >= rows) return;
  const unsigned int groups = columns / 32u;
  float total = 0.0f;
  for (unsigned int group = 0; group < groups; ++group) {
    const unsigned int column = group * 32u + lane;
    const unsigned char packed = weight[(row * groups + group) * 16u + lane / 2u];
    const unsigned int code = lane % 2u == 0u ? packed & 15u : packed >> 4u;
    const float scale = ldexpf(1.0f, int(scales[row * groups + group]) - 127);
    total = fmaf(__bfloat162float(input[token * columns + column]),
                 kMxFp4Values[code] * scale, total);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down_sync(0xffffffffu, total, offset);
  }
  if (lane == 0u) {
    const float value = total +
        (has_bias == 0u ? 0.0f : __bfloat162float(bias[row]));
    output[token * rows + row] = __float2bfloat16_rn(value);
  }
}

extern "C" __global__ void libmir_cuda_mxfp4_bf16_gathered_linear(
    const __nv_bfloat16* input, const unsigned char* weight,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int assignments, unsigned int matrices, unsigned int rows,
    unsigned int columns, unsigned int selections_per_input,
    unsigned int has_bias) {
  const unsigned int assignment = blockIdx.y;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int row = blockIdx.x * (blockDim.x >> 5u) + warp;
  if (assignment >= assignments || row >= rows) return;
  const unsigned int matrix = selected[assignment];
  if (matrix >= matrices) {
    if (lane == 0u) output[assignment * rows + row] = __float2bfloat16(0.0f);
    return;
  }
  const unsigned int groups = columns / 32u;
  const unsigned int matrix_row = matrix * rows + row;
  const unsigned int input_row = assignment / selections_per_input;
  float total = 0.0f;
  for (unsigned int group = 0; group < groups; ++group) {
    const unsigned int column = group * 32u + lane;
    const unsigned char packed =
        weight[(matrix_row * groups + group) * 16u + lane / 2u];
    const unsigned int code = lane % 2u == 0u ? packed & 15u : packed >> 4u;
    const float scale = ldexpf(1.0f, int(scales[matrix_row * groups + group]) - 127);
    total = fmaf(__bfloat162float(input[input_row * columns + column]),
                 kMxFp4Values[code] * scale, total);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down_sync(0xffffffffu, total, offset);
  }
  if (lane == 0u) {
    const float value = total + (has_bias == 0u
        ? 0.0f : __bfloat162float(bias[matrix_row]));
    output[assignment * rows + row] = __float2bfloat16_rn(value);
  }
}

extern "C" __global__ void libmir_cuda_mxfp4_embedding_bf16(
    const unsigned char* weight, const unsigned char* scales,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int selected_start, unsigned int tokens, unsigned int vocab,
    unsigned int hidden, float output_scale) {
  const unsigned int feature = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int token = blockIdx.y;
  if (feature >= hidden || token >= tokens) return;
  const unsigned int row = selected[selected_start + token];
  if (row >= vocab) return;
  const unsigned int groups = hidden / 32u;
  const unsigned int group = feature / 32u;
  const unsigned int offset = feature % 32u;
  const unsigned char packed = weight[(row * groups + group) * 16u + offset / 2u];
  const unsigned int code = offset % 2u == 0u ? packed & 15u : packed >> 4u;
  const float scale = ldexpf(1.0f, int(scales[row * groups + group]) - 127);
  output[token * hidden + feature] =
      __float2bfloat16_rn(kMxFp4Values[code] * scale * output_scale);
}
