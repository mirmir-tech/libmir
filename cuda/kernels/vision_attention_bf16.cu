#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_spatial_rope_bf16(
    const __nv_bfloat16* input, const unsigned int* positions,
    __nv_bfloat16* output, unsigned int tokens, unsigned int heads,
    unsigned int head_dim, float theta) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = tokens * heads * head_dim;
  if (index >= elements) return;
  const unsigned int dimension = index % head_dim;
  const unsigned int token = index / (heads * head_dim);
  const unsigned int coordinate_dim = head_dim / 2;
  const unsigned int quarter = coordinate_dim / 2;
  const unsigned int coordinate = dimension / coordinate_dim;
  const unsigned int local = dimension % coordinate_dim;
  const unsigned int pair_local = local < quarter ? local + quarter : local - quarter;
  const unsigned int pair_dimension = coordinate * coordinate_dim + pair_local;
  const unsigned int base = index - dimension;
  const float value = __bfloat162float(input[index]);
  float rotated = __bfloat162float(input[base + pair_dimension]);
  if (local < quarter) rotated = -rotated;
  const unsigned int frequency_index = local % quarter;
  const float exponent = -2.0f * frequency_index / coordinate_dim;
  const float inverse_frequency = powf(theta, exponent);
  const float position = static_cast<float>(positions[token * 2 + coordinate]);
  const float angle = position * inverse_frequency;
  output[index] = __float2bfloat16_rn(value * cosf(angle) + rotated * sinf(angle));
}

extern "C" __global__ void libmir_cuda_vision_attention_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* output,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, float scale) {
  const unsigned int query_token = blockIdx.x;
  const unsigned int query_head = blockIdx.y;
  const unsigned int dimension = threadIdx.x;
  const unsigned int group = query_heads / kv_heads;
  const unsigned int kv_head = query_head / group;
  const unsigned int query_base = (query_token * query_heads + query_head) * head_dim;
  float accumulator = 0.0f;
  float maximum = -3.402823466e+38F;
  float denominator = 0.0f;
  __shared__ float reductions[8];
  __shared__ float factors[3];
  for (unsigned int key_token = 0; key_token < tokens; ++key_token) {
    const unsigned int key_base = (key_token * kv_heads + kv_head) * head_dim;
    float dot = dimension < head_dim
        ? __bfloat162float(query[query_base + dimension]) *
          __bfloat162float(key[key_base + dimension])
        : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const unsigned int lane = dimension & 31u;
    const unsigned int warp = dimension / 32u;
    if (lane == 0u) reductions[warp] = dot;
    __syncthreads();
    if (warp == 0u) {
      dot = lane < 8u ? reductions[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      }
      if (lane == 0u) {
        const float score = dot * scale;
        const float next_maximum = fmaxf(maximum, score);
        factors[0] = isfinite(maximum) ? expf(maximum - next_maximum) : 0.0f;
        factors[1] = expf(score - next_maximum);
        denominator = denominator * factors[0] + factors[1];
        maximum = next_maximum;
        factors[2] = denominator;
      }
    }
    __syncthreads();
    if (dimension < head_dim) {
      accumulator = accumulator * factors[0] +
          factors[1] * __bfloat162float(value[key_base + dimension]);
    }
    denominator = factors[2];
    __syncthreads();
  }
  if (dimension < head_dim) {
    output[query_base + dimension] = __float2bfloat16_rn(accumulator / denominator);
  }
}
