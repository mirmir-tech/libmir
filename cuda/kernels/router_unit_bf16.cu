#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_router_topk_unit_bf16(
    const __nv_bfloat16* scores, unsigned int* selected,
    __nv_bfloat16* weights, unsigned int experts, unsigned int top_k,
    unsigned int tokens) {
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  scores += token * experts;
  selected += token * top_k;
  weights += token * top_k;
  const unsigned int lane = threadIdx.x;
  float local_values[8];
  unsigned int local_indices[8];
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) {
    const unsigned int expert = lane + item * 32;
    local_values[item] = expert < experts
        ? __bfloat162float(scores[expert]) : -3.402823466e+38F;
    local_indices[item] = expert;
  }
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    float best_score = local_values[0];
    unsigned int best = local_indices[0];
#pragma unroll
    for (unsigned int item = 1; item < 8; ++item) {
      const float candidate = local_values[item];
      const unsigned int index = local_indices[item];
      if (candidate > best_score || (candidate == best_score && index < best)) {
        best_score = candidate;
        best = index;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float candidate = __shfl_down_sync(0xffffffffu, best_score, offset);
      const unsigned int index = __shfl_down_sync(0xffffffffu, best, offset);
      if (candidate > best_score || (candidate == best_score && index < best)) {
        best_score = candidate;
        best = index;
      }
    }
    best = __shfl_sync(0xffffffffu, best, 0);
    best_score = __shfl_sync(0xffffffffu, best_score, 0);
    if (lane == 0) {
      selected[rank] = best;
      weights[rank] = __float2bfloat16_rn(best_score);
    }
#pragma unroll
    for (unsigned int item = 0; item < 8; ++item) {
      if (local_indices[item] == best) local_values[item] = -3.402823466e+38F;
    }
  }
  if (lane != 0) return;
  const float maximum = __bfloat162float(weights[0]);
  float denominator = 0.0f;
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    denominator += expf(__bfloat162float(weights[rank]) - maximum);
  }
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    const float probability =
        expf(__bfloat162float(weights[rank]) - maximum) / denominator;
    weights[rank] = __float2bfloat16_rn(probability);
  }
}
