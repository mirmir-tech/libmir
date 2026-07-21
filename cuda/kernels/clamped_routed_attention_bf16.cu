#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ float clamped_routed_load_page(const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

__device__ void clamped_routed_sink_step(float score, float& maximum, float& denominator,
                                 float& alpha, float& beta) {
  const float next = fmaxf(maximum, score);
  alpha = expf(maximum - next);
  beta = expf(score - next);
  denominator = denominator * alpha + beta;
  maximum = next;
}

extern "C" __global__ void libmir_cuda_clamped_routed_paged_attention_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_table,
    const __nv_bfloat16* sinks, __nv_bfloat16* output,
    unsigned int token_count, unsigned int block_count, unsigned int block_size,
    unsigned int query_heads, unsigned int kv_heads, unsigned int head_dim,
    unsigned int window, float scale) {
  const unsigned int query_head = blockIdx.x;
  if (query_head >= query_heads || token_count == 0u) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int first = window > 0u && token_count > window ? token_count - window : 0u;
  float accumulator = 0.0f;
  __shared__ float warp_sums[8];
  __shared__ float alpha, beta, denominator, maximum;
  if (lane == 0u) {
    maximum = __bfloat162float(sinks[query_head]);
    denominator = 1.0f;
  }
  __syncthreads();
  for (unsigned int token = first; token < token_count; ++token) {
    const unsigned int logical = token / block_size;
    if (logical >= block_count) return;
    const unsigned int page_token = block_table[logical] * block_size + token % block_size;
    float dot = 0.0f;
    for (unsigned int dim = lane; dim < head_dim; dim += blockDim.x) {
      dot = fmaf(__bfloat162float(query[query_head * head_dim + dim]),
                 clamped_routed_load_page(key_pages, (page_token * kv_heads + kv_head) * head_dim + dim), dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      float sum = lane < 8u ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0u) clamped_routed_sink_step(sum * scale, maximum, denominator, alpha, beta);
    }
    __syncthreads();
    if (lane < head_dim) {
      const unsigned int value_index = (page_token * kv_heads + kv_head) * head_dim + lane;
      accumulator = fmaf(accumulator, alpha, clamped_routed_load_page(value_pages, value_index) * beta);
    }
    __syncthreads();
  }
  if (lane < head_dim) output[query_head * head_dim + lane] =
      __float2bfloat16_rn(accumulator / denominator);
}

extern "C" __global__ void libmir_cuda_clamped_routed_paged_prefill_attention_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_table,
    const __nv_bfloat16* sinks, __nv_bfloat16* output,
    unsigned int query_tokens, unsigned int start_position, unsigned int block_count,
    unsigned int block_size, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int window, float scale) {
  const unsigned int query_token = blockIdx.x / query_heads;
  const unsigned int query_head = blockIdx.x % query_heads;
  if (query_token >= query_tokens) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int context = start_position + query_token + 1u;
  const unsigned int first = window > 0u && context > window ? context - window : 0u;
  float accumulator = 0.0f;
  __shared__ float warp_sums[8];
  __shared__ float alpha, beta, denominator, maximum;
  if (lane == 0u) {
    maximum = __bfloat162float(sinks[query_head]);
    denominator = 1.0f;
  }
  __syncthreads();
  for (unsigned int token = first; token < context; ++token) {
    const unsigned int logical = token / block_size;
    if (logical >= block_count) return;
    const unsigned int page_token = block_table[logical] * block_size + token % block_size;
    float dot = 0.0f;
    for (unsigned int dim = lane; dim < head_dim; dim += blockDim.x) {
      const unsigned int q_index = (query_token * query_heads + query_head) * head_dim + dim;
      const unsigned int k_index = (page_token * kv_heads + kv_head) * head_dim + dim;
      dot = fmaf(__bfloat162float(query[q_index]), clamped_routed_load_page(key_pages, k_index), dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      float sum = lane < 8u ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0u) clamped_routed_sink_step(sum * scale, maximum, denominator, alpha, beta);
    }
    __syncthreads();
    if (lane < head_dim) {
      const unsigned int value_index = (page_token * kv_heads + kv_head) * head_dim + lane;
      accumulator = fmaf(accumulator, alpha, clamped_routed_load_page(value_pages, value_index) * beta);
    }
    __syncthreads();
  }
  if (lane < head_dim) {
    const unsigned int out = (query_token * query_heads + query_head) * head_dim + lane;
    output[out] = __float2bfloat16_rn(accumulator / denominator);
  }
}
