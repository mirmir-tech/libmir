#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_router_topk_unit_bf16(
    const __nv_bfloat16* scores, unsigned int* selected,
    __nv_bfloat16* weights, unsigned int experts, unsigned int top_k,
    unsigned int tokens, float scale) {
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
#ifdef LIBMIR_ROUTER_PROBABILITY
  // Weights are softmax probabilities over every expert, not renormalized
  // over the selected ones (`DeepSeek-V2`).
  float all_maximum = local_values[0];
#pragma unroll
  for (unsigned int item = 1; item < 8; ++item) all_maximum = fmaxf(all_maximum, local_values[item]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    all_maximum = fmaxf(all_maximum, __shfl_xor_sync(0xffffffffu, all_maximum, offset));
  }
  float all_sum = 0.0f;
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) {
    if (local_indices[item] < experts) all_sum += expf(local_values[item] - all_maximum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    all_sum += __shfl_xor_sync(0xffffffffu, all_sum, offset);
  }
#endif
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
#ifdef LIBMIR_ROUTER_PROBABILITY
      weights[rank] = __float2bfloat16_rn(expf(best_score - all_maximum) / all_sum * scale);
#else
      weights[rank] = __float2bfloat16_rn(best_score);
#endif
    }
#pragma unroll
    for (unsigned int item = 0; item < 8; ++item) {
      if (local_indices[item] == best) local_values[item] = -3.402823466e+38F;
    }
  }
#ifdef LIBMIR_ROUTER_PROBABILITY
  return;
#endif
  if (lane != 0) return;
  const float maximum = __bfloat162float(weights[0]);
  float denominator = 0.0f;
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    denominator += expf(__bfloat162float(weights[rank]) - maximum);
  }
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    const float probability =
        expf(__bfloat162float(weights[rank]) - maximum) / denominator;
    weights[rank] = __float2bfloat16_rn(probability * scale);
  }
}

// Sigmoid routing with a selection-only correction bias (`DeepSeek-V3`
// `noaux_tc`, `Moonlight`): experts are chosen by `sigmoid(logit) + bias` and
// weighted by their `sigmoid(logit)` renormalized over the chosen ones.
extern "C" __global__ void libmir_cuda_router_topk_sigmoid_bf16(
    const __nv_bfloat16* scores, const float* bias, unsigned int* selected,
    __nv_bfloat16* weights, unsigned int experts, unsigned int top_k,
    unsigned int tokens, float scale) {
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  scores += token * experts;
  selected += token * top_k;
  weights += token * top_k;
  const unsigned int lane = threadIdx.x;
  float choices[8];
  unsigned int indices[8];
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) {
    const unsigned int expert = lane + item * 32;
    choices[item] = expert < experts
        ? 1.0f / (1.0f + expf(-__bfloat162float(scores[expert]))) + bias[expert]
        : -3.402823466e+38F;
    indices[item] = expert;
  }
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    float best_choice = choices[0];
    unsigned int best = indices[0];
#pragma unroll
    for (unsigned int item = 1; item < 8; ++item) {
      if (choices[item] > best_choice ||
          (choices[item] == best_choice && indices[item] < best)) {
        best_choice = choices[item];
        best = indices[item];
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float candidate = __shfl_down_sync(0xffffffffu, best_choice, offset);
      const unsigned int index = __shfl_down_sync(0xffffffffu, best, offset);
      if (candidate > best_choice || (candidate == best_choice && index < best)) {
        best_choice = candidate;
        best = index;
      }
    }
    best = __shfl_sync(0xffffffffu, best, 0);
    if (lane == 0) selected[rank] = best;
#pragma unroll
    for (unsigned int item = 0; item < 8; ++item) {
      if (indices[item] == best) choices[item] = -3.402823466e+38F;
    }
  }
  if (lane != 0) return;
  float sum = 0.0f;
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    sum += 1.0f / (1.0f + expf(-__bfloat162float(scores[selected[rank]])));
  }
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    const float gate = 1.0f / (1.0f + expf(-__bfloat162float(scores[selected[rank]])));
    weights[rank] = __float2bfloat16_rn(gate / sum * scale);
  }
}
