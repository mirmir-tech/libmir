#include <cuda_bf16.h>

#define ROUTER_NEGATIVE -3.402823466e+38F

#ifdef LIBMIR_ROUTER_GROUPS
// Group-limited selection: the experts form LIBMIR_ROUTER_GROUPS contiguous
// groups, and only those of the LIBMIR_ROUTER_GROUP_LIMIT best-scoring
// groups stay selectable. A group scores its best value, or with `top_two`
// the sum of its two best. One warp serves the token; lane `l` holds experts
// `l + 32 * item`.
__device__ __forceinline__ void limit_groups(
    float (&values)[8], unsigned int experts, bool top_two) {
  __shared__ float staged[256];
  const unsigned int lane = threadIdx.x;
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) {
    const unsigned int expert = lane + item * 32;
    if (expert < experts) staged[expert] = values[item];
  }
  __syncwarp();
  const unsigned int size = experts / LIBMIR_ROUTER_GROUPS;
  float score = ROUTER_NEGATIVE;
  if (lane < LIBMIR_ROUTER_GROUPS) {
    float first = ROUTER_NEGATIVE;
    float second = ROUTER_NEGATIVE;
    for (unsigned int expert = lane * size; expert < (lane + 1u) * size; ++expert) {
      const float value = staged[expert];
      if (value > first) {
        second = first;
        first = value;
      } else if (value > second) {
        second = value;
      }
    }
    score = top_two ? first + second : first;
  }
  unsigned int kept = 0u;
  for (unsigned int rank = 0; rank < LIBMIR_ROUTER_GROUP_LIMIT; ++rank) {
    float best = score;
    unsigned int group = lane;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float candidate = __shfl_xor_sync(0xffffffffu, best, offset);
      const unsigned int index = __shfl_xor_sync(0xffffffffu, group, offset);
      if (candidate > best || (candidate == best && index < group)) {
        best = candidate;
        group = index;
      }
    }
    kept |= 1u << group;
    // Below every remaining score, including the idle lanes'.
    if (lane == group) score = __int_as_float(0xff800000);
  }
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) {
    const unsigned int expert = lane + item * 32;
    if (expert < experts && ((kept >> (expert / size)) & 1u) == 0u) {
      values[item] = ROUTER_NEGATIVE;
    }
  }
}
#endif

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
        ? __bfloat162float(scores[expert]) : ROUTER_NEGATIVE;
    local_indices[item] = expert;
  }
#ifdef LIBMIR_ROUTER_SPARSEMIXER_JITTER
  // Phi-3.5-MoE's sparsemixer measures each round's band on the logits
  // before earlier rounds removed their experts.
  float logits[8];
#pragma unroll
  for (unsigned int item = 0; item < 8; ++item) logits[item] = local_values[item];
#endif
#ifdef LIBMIR_ROUTER_PROBABILITY
  // Weights are softmax probabilities over every expert, not renormalized
  // over the selected ones (`norm_topk_prob` false).
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
#ifdef LIBMIR_ROUTER_GROUPS
  // The softmax above spans every expert; groups only limit the choice, and
  // a group's best logit ranks it as its best probability would.
  limit_groups(local_values, experts, false);
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
#ifdef LIBMIR_ROUTER_SPARSEMIXER_JITTER
    // The chosen expert's softmax over the remaining experts whose logit is
    // within `2 * jitter` of it, relative to `max(|logit|, best)`; a NaN
    // distance (both zero) stays in the band, as in transformers.
    float band = 0.0f;
#pragma unroll
    for (unsigned int item = 0; item < 8; ++item) {
      const float factor = fmaxf(fabsf(logits[item]), best_score);
      const bool far = (best_score - logits[item]) / factor > 2.0f * LIBMIR_ROUTER_SPARSEMIXER_JITTER;
      if (local_indices[item] < experts && local_values[item] > ROUTER_NEGATIVE && !far) {
        band += expf(local_values[item] - best_score);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      band += __shfl_xor_sync(0xffffffffu, band, offset);
    }
#endif
    if (lane == 0) {
      selected[rank] = best;
#if defined(LIBMIR_ROUTER_PROBABILITY)
      weights[rank] = __float2bfloat16_rn(expf(best_score - all_maximum) / all_sum * scale);
#elif defined(LIBMIR_ROUTER_INPUT_SCALED)
      // Llama 4: the selected logit's sigmoid, which scales the expert's
      // input rather than its output.
      weights[rank] = __float2bfloat16_rn(scale / (1.0f + expf(-best_score)));
#elif defined(LIBMIR_ROUTER_SPARSEMIXER_JITTER)
      weights[rank] = __float2bfloat16_rn(scale / band);
#else
      weights[rank] = __float2bfloat16_rn(best_score);
#endif
    }
#pragma unroll
    for (unsigned int item = 0; item < 8; ++item) {
      if (local_indices[item] == best) local_values[item] = ROUTER_NEGATIVE;
    }
  }
#if defined(LIBMIR_ROUTER_PROBABILITY) || defined(LIBMIR_ROUTER_SPARSEMIXER_JITTER) \
    || defined(LIBMIR_ROUTER_INPUT_SCALED)
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

// Sigmoid routing with a selection-only correction bias (`noaux_tc`):
// experts are chosen by `sigmoid(logit) + bias` and weighted by their
// `sigmoid(logit)` renormalized over the chosen ones.
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
        : ROUTER_NEGATIVE;
    indices[item] = expert;
  }
#ifdef LIBMIR_ROUTER_GROUPS
  limit_groups(choices, experts, true);
#endif
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
      if (indices[item] == best) choices[item] = ROUTER_NEGATIVE;
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
