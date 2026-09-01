#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_clamped_routed_marlin_gate_up_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int assignments, unsigned int intermediate,
    unsigned int padded_intermediate, float limit) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= assignments * padded_intermediate) return;
  const unsigned int assignment = index / padded_intermediate;
  const unsigned int unit = index % padded_intermediate;
  if (unit >= intermediate) {
    output[index] = __float2bfloat16_rn(0.0f);
    return;
  }
  const unsigned int expert = selected[assignment];
  const unsigned int input_base = assignment * padded_intermediate * 2u;
  const unsigned int bias_base = expert * intermediate * 2u;
  float gate = __bfloat162float(input[input_base + unit]) +
               __bfloat162float(bias[bias_base + unit * 2u]);
  float up = __bfloat162float(input[input_base + padded_intermediate + unit]) +
             __bfloat162float(bias[bias_base + unit * 2u + 1u]);
  gate = fminf(gate, limit);
  up = fminf(limit, fmaxf(-limit, up));
  output[index] = __float2bfloat16_rn(
      gate / (1.0f + expf(-1.702f * gate)) * (up + 1.0f));
}

extern "C" __global__ void libmir_cuda_clamped_routed_marlin_down_reduce_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    const unsigned int* selected, const __nv_bfloat16* routing,
    __nv_bfloat16* output, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int padded_hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int token = index / hidden;
  const unsigned int column = index % hidden;
  float sum = 0.0f;
  for (unsigned int route = 0u; route < top_k; ++route) {
    const unsigned int assignment = token * top_k + route;
    const unsigned int expert = selected[assignment];
    const __nv_bfloat16 biased = __float2bfloat16_rn(
        __bfloat162float(input[assignment * padded_hidden + column]) +
        __bfloat162float(bias[expert * hidden + column]));
    sum += __bfloat162float(biased) * __bfloat162float(routing[assignment]);
  }
  output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void libmir_cuda_clamped_routed_marlin_pad_rows_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, unsigned int padded_columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * padded_columns) return;
  const unsigned int row = index / padded_columns;
  const unsigned int column = index % padded_columns;
  output[index] = column < columns
      ? input[row * columns + column]
      : __float2bfloat16_rn(0.0f);
}
