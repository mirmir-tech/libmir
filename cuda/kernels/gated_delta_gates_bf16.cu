#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_gated_delta_alpha_beta_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* alpha_weight,
    const __nv_bfloat16* beta_weight, __nv_bfloat16* alpha,
    __nv_bfloat16* beta, unsigned int tokens, unsigned int columns,
    unsigned int heads) {
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int head = blockIdx.x * 8u + warp;
  const unsigned int token = blockIdx.y;
  if (head >= heads || token >= tokens) return;
  const __nv_bfloat16* token_input = input + token * columns;
  const __nv_bfloat16* alpha_row = alpha_weight + head * columns;
  const __nv_bfloat16* beta_row = beta_weight + head * columns;
  float alpha_sum = 0.0f;
  float beta_sum = 0.0f;
  for (unsigned int column = lane; column < columns; column += 32u) {
    const float value = __bfloat162float(token_input[column]);
    alpha_sum = fmaf(value, __bfloat162float(alpha_row[column]), alpha_sum);
    beta_sum = fmaf(value, __bfloat162float(beta_row[column]), beta_sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    alpha_sum += __shfl_down_sync(0xffffffffu, alpha_sum, offset);
    beta_sum += __shfl_down_sync(0xffffffffu, beta_sum, offset);
  }
  if (lane == 0u) {
    const unsigned int output = token * heads + head;
    alpha[output] = __float2bfloat16_rn(alpha_sum);
    beta[output] = __float2bfloat16_rn(beta_sum);
  }
}
