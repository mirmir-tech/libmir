#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define LIBMIR_NEGATIVE_FLOAT_MAX -3.402823466e+38F

__device__ float load_split_page(const unsigned char* pages,
                                 unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_split_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_table,
    float* partial_values, float* partial_maxima, float* partial_denominators,
    unsigned int token_count, unsigned int block_count,
    unsigned int block_size, unsigned int query_heads,
    unsigned int kv_heads, unsigned int head_dim,
    unsigned int value_head_dim, unsigned int window, float scale,
    unsigned int partition_tokens, unsigned int active_partitions,
    unsigned int max_partitions, unsigned int minimum_tokens) {
  const unsigned int visible_tokens = window > 0
      ? min(token_count, window) : token_count;
  if (visible_tokens < minimum_tokens) return;
  const unsigned int query_head = blockIdx.x / active_partitions;
  const unsigned int partition = blockIdx.x % active_partitions;
  if (query_head >= query_heads || token_count == 0) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int first = window > 0 && token_count > window
      ? token_count - window : 0;
  const unsigned int begin = first + partition * partition_tokens;
  const unsigned int end = min(begin + partition_tokens, token_count);
  const unsigned int partial = query_head * max_partitions + partition;
  float accumulators[2] = {0.0f, 0.0f};
  __shared__ float warp_sums[8];
  __shared__ float alpha;
  __shared__ float beta;
  __shared__ float denominator;
  __shared__ float maximum;
  if (lane == 0) {
    denominator = 0.0f;
    maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
  }
  __syncthreads();

  for (unsigned int token = begin; token < end; ++token) {
    const unsigned int logical_block = token / block_size;
    if (logical_block >= block_count) return;
    const unsigned int physical_block = block_table[logical_block];
    const unsigned int page_token =
        physical_block * block_size + token % block_size;
    float dot = 0.0f;
    for (unsigned int dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
      const unsigned int q_index = query_head * head_dim + dimension;
      const unsigned int k_index =
          (page_token * kv_heads + kv_head) * head_dim + dimension;
      dot = fmaf(__bfloat162float(query[q_index]),
                 load_split_page(key_pages, k_index), dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      float sum = lane < blockDim.x / 32u ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      if (lane == 0u) {
        const float score = sum * scale;
        const float next_maximum = fmaxf(maximum, score);
        alpha = expf(maximum - next_maximum);
        beta = expf(score - next_maximum);
        denominator = denominator * alpha + beta;
        maximum = next_maximum;
      }
    }
    __syncthreads();
    unsigned int local = 0;
    for (unsigned int dimension = lane; dimension < value_head_dim;
         dimension += blockDim.x, ++local) {
      const unsigned int v_index =
          (page_token * kv_heads + kv_head) * value_head_dim + dimension;
      accumulators[local] = fmaf(accumulators[local], alpha,
          load_split_page(value_pages, v_index) * beta);
    }
    __syncthreads();
  }
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    partial_values[partial * value_head_dim + dimension] =
        accumulators[dimension / blockDim.x];
  }
  if (lane == 0) {
    partial_maxima[partial] = maximum;
    partial_denominators[partial] = denominator;
  }
}

extern "C" __global__ void libmir_cuda_paged_attention_merge_bf16(
    const float* partial_values, const float* partial_maxima,
    const float* partial_denominators, __nv_bfloat16* output,
    unsigned int query_heads, unsigned int value_head_dim,
    unsigned int active_partitions, unsigned int max_partitions,
    unsigned int visible_tokens, unsigned int minimum_tokens) {
  if (visible_tokens < minimum_tokens) return;
  const unsigned int query_head = blockIdx.x;
  if (query_head >= query_heads) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int base = query_head * max_partitions;
  __shared__ float maximum;
  __shared__ float denominator;
  if (lane == 0) {
    float merged_maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      merged_maximum = fmaxf(merged_maximum, partial_maxima[base + partition]);
    }
    float merged_denominator = 0.0f;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      merged_denominator += partial_denominators[base + partition] *
          expf(partial_maxima[base + partition] - merged_maximum);
    }
    maximum = merged_maximum;
    denominator = merged_denominator;
  }
  __syncthreads();
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    float numerator = 0.0f;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      const float weight = expf(partial_maxima[base + partition] - maximum);
      numerator += partial_values[(base + partition) * value_head_dim + dimension] * weight;
    }
    output[query_head * value_head_dim + dimension] =
        __float2bfloat16_rn(numerator / denominator);
  }
}
