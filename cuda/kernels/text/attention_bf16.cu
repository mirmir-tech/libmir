#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_text_attention_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* output,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, float scale, unsigned int causal) {
  const unsigned int query_token = blockIdx.x;
  const unsigned int query_head = blockIdx.y;
  const unsigned int dimension = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int query_base = (query_token * query_heads + query_head) * head_dim;
  const unsigned int key_end = causal != 0u ? query_token + 1u : tokens;
  float accumulator = 0.0f;
  float maximum = -3.402823466e+38F;
  float denominator = 0.0f;
  __shared__ float reductions[8];
  __shared__ float factors[3];
  for (unsigned int key_token = 0; key_token < key_end; ++key_token) {
    const unsigned int key_base = (key_token * kv_heads + kv_head) * head_dim;
    float dot = dimension < head_dim
        ? __bfloat162float(query[query_base + dimension]) *
          __bfloat162float(key[key_base + dimension]) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    const unsigned int lane = dimension & 31u;
    const unsigned int warp = dimension / 32u;
    if (lane == 0u) reductions[warp] = dot;
    __syncthreads();
    if (warp == 0u) {
      dot = lane < 8u ? reductions[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      if (lane == 0u) {
        const float score = dot * scale;
        const float next = fmaxf(maximum, score);
        factors[0] = isfinite(maximum) ? expf(maximum - next) : 0.0f;
        factors[1] = expf(score - next);
        denominator = denominator * factors[0] + factors[1];
        maximum = next;
        factors[2] = denominator;
      }
    }
    __syncthreads();
    if (dimension < head_dim)
      accumulator = accumulator * factors[0] +
          factors[1] * __bfloat162float(value[key_base + dimension]);
    denominator = factors[2];
    __syncthreads();
  }
  if (dimension < head_dim)
    output[query_base + dimension] = __float2bfloat16_rn(accumulator / denominator);
}
