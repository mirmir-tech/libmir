#include <cuda_fp16.h>

__device__ float rotated(const half* vector, unsigned int base, unsigned int dimension,
    unsigned int head_dim, unsigned int position, float theta, float factor) {
  const unsigned int half_dim = head_dim / 2u;
  const unsigned int local = dimension % half_dim;
  const unsigned int pair = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
  const float exponent = 2.0f * static_cast<float>(local) / static_cast<float>(head_dim);
  const float denominator = powf(theta * factor, exponent) * powf(factor, 2.0f / head_dim);
  const float angle = static_cast<float>(position) / denominator;
  const float value = __half2float(vector[base + dimension]);
  float paired = __half2float(vector[base + pair]);
  if (dimension < half_dim) paired = -paired;
  return value * cosf(angle) + paired * sinf(angle);
}

extern "C" __global__ void libmir_cuda_encoder_attention_f16(
    const half* qkv, half* output, unsigned int tokens, unsigned int heads,
    unsigned int head_dim, float scale, float theta, float ntk_factor) {
  const unsigned int query_token = blockIdx.x;
  const unsigned int head = blockIdx.y;
  const unsigned int dimension = threadIdx.x;
  const unsigned int hidden = heads * head_dim;
  const unsigned int q_base = query_token * hidden * 3u + head * head_dim;
  float accumulator = 0.0f;
  float maximum = -3.402823466e+38F;
  float denominator = 0.0f;
  __shared__ float reductions[8];
  __shared__ float factors[3];
  for (unsigned int key_token = 0; key_token < tokens; ++key_token) {
    const unsigned int row = key_token * hidden * 3u;
    const unsigned int k_base = row + hidden + head * head_dim;
    float dot = dimension < head_dim
        ? rotated(qkv, q_base, dimension, head_dim, query_token, theta, ntk_factor) *
          rotated(qkv, k_base, dimension, head_dim, key_token, theta, ntk_factor) : 0.0f;
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
    if (dimension < head_dim) {
      const unsigned int v_base = row + hidden * 2u + head * head_dim;
      accumulator = accumulator * factors[0] + factors[1] * __half2float(qkv[v_base + dimension]);
    }
    denominator = factors[2];
    __syncthreads();
  }
  if (dimension < head_dim)
    output[(query_token * heads + head) * head_dim + dimension] = __float2half(accumulator / denominator);
}
