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

__device__ float clamped_routed_load_kv(
    const unsigned char* pages, const __nv_bfloat16* current,
    unsigned int page_index, unsigned int current_index, bool from_current) {
  return from_current ? __bfloat162float(current[current_index])
                      : clamped_routed_load_page(pages, page_index);
}

__device__ void clamped_routed_sink_step(float score, float& maximum, float& denominator,
                                 float& alpha, float& beta) {
  const float next = fmaxf(maximum, score);
  alpha = expf(maximum - next);
  beta = expf(score - next);
  denominator = denominator * alpha + beta;
  maximum = next;
}

extern "C" __global__ void libmir_cuda_clamped_routed_sink_scale_bf16(
    __nv_bfloat16* output, const float* softmax_lse,
    const __nv_bfloat16* sinks, unsigned int query_tokens,
    unsigned int query_heads, unsigned int head_dim) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = query_tokens * query_heads * head_dim;
  if (index >= elements) return;
  const unsigned int row = index / head_dim;
  const unsigned int head = row % query_heads;
  const unsigned int token = row / query_heads;
  const float lse = softmax_lse[head * query_tokens + token];
  const float sink = __bfloat162float(sinks[head]);
  const float factor = 1.0f / (1.0f + expf(sink - lse));
  output[index] =
      __float2bfloat16_rn(__bfloat162float(output[index]) * factor);
}

extern "C" __global__ void libmir_cuda_clamped_routed_sink_merge_bf16(
    const float* partial_values, const float* partial_maxima,
    const float* partial_denominators, const __nv_bfloat16* sinks,
    __nv_bfloat16* output, unsigned int query_heads, unsigned int head_dim,
    unsigned int active_partitions, unsigned int max_partitions) {
  const unsigned int query_head = blockIdx.x;
  if (query_head >= query_heads) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int base = query_head * max_partitions;
  __shared__ float maximum;
  __shared__ float denominator;
  if (lane == 0u) {
    float merged_maximum = __bfloat162float(sinks[query_head]);
    for (unsigned int partition = 0u; partition < active_partitions;
         ++partition) {
      merged_maximum =
          fmaxf(merged_maximum, partial_maxima[base + partition]);
    }
    float merged_denominator =
        expf(__bfloat162float(sinks[query_head]) - merged_maximum);
    for (unsigned int partition = 0u; partition < active_partitions;
         ++partition) {
      merged_denominator += partial_denominators[base + partition] *
          expf(partial_maxima[base + partition] - merged_maximum);
    }
    maximum = merged_maximum;
    denominator = merged_denominator;
  }
  __syncthreads();
  for (unsigned int dimension = lane; dimension < head_dim;
       dimension += blockDim.x) {
    float numerator = 0.0f;
    for (unsigned int partition = 0u; partition < active_partitions;
         ++partition) {
      const float weight =
          expf(partial_maxima[base + partition] - maximum);
      numerator +=
          partial_values[(base + partition) * head_dim + dimension] *
          weight;
    }
    output[query_head * head_dim + dimension] =
        __float2bfloat16_rn(numerator / denominator);
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_sink_batch_merge_bf16(
    const float* partial_values, const float* partial_maxima,
    const float* partial_denominators, const unsigned int* token_counts,
    const __nv_bfloat16* sinks, __nv_bfloat16* output,
    unsigned int batch_size, unsigned int query_heads, unsigned int head_dim,
    unsigned int window, unsigned int partition_tokens,
    unsigned int max_partitions, unsigned int minimum_tokens) {
  const unsigned int sequence = blockIdx.y;
  const unsigned int query_head = blockIdx.x;
  if (sequence >= batch_size || query_head >= query_heads) return;
  const unsigned int token_count = token_counts[sequence];
  const unsigned int visible =
      window > 0u ? min(token_count, window) : token_count;
  if (visible < minimum_tokens) return;
  const unsigned int active =
      (visible + partition_tokens - 1u) / partition_tokens;
  const unsigned int base =
      (sequence * query_heads + query_head) * max_partitions;
  const unsigned int lane = threadIdx.x;
  __shared__ float maximum;
  __shared__ float denominator;
  if (lane == 0u) {
    float merged_maximum = __bfloat162float(sinks[query_head]);
    for (unsigned int partition = 0u; partition < active; ++partition) {
      merged_maximum =
          fmaxf(merged_maximum, partial_maxima[base + partition]);
    }
    float merged_denominator =
        expf(__bfloat162float(sinks[query_head]) - merged_maximum);
    for (unsigned int partition = 0u; partition < active; ++partition) {
      merged_denominator += partial_denominators[base + partition] *
          expf(partial_maxima[base + partition] - merged_maximum);
    }
    maximum = merged_maximum;
    denominator = merged_denominator;
  }
  __syncthreads();
  for (unsigned int dimension = lane; dimension < head_dim;
       dimension += blockDim.x) {
    float numerator = 0.0f;
    for (unsigned int partition = 0u; partition < active; ++partition) {
      const float weight =
          expf(partial_maxima[base + partition] - maximum);
      numerator +=
          partial_values[(base + partition) * head_dim + dimension] * weight;
    }
    const unsigned int out =
        (sequence * query_heads + query_head) * head_dim + dimension;
    output[out] = __float2bfloat16_rn(numerator / denominator);
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_paged_attention_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* current_keys,
    const __nv_bfloat16* current_values, const unsigned char* key_pages,
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
  if (head_dim == 64u && blockDim.x == 32u) {
    const unsigned int first_dim = lane * 2u;
    const unsigned int second_dim = first_dim + 1u;
    const unsigned int query_base = query_head * head_dim;
    const float first_query =
        __bfloat162float(query[query_base + first_dim]);
    const float second_query =
        __bfloat162float(query[query_base + second_dim]);
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;
    __shared__ float warp_alpha, warp_beta, warp_denominator, warp_maximum;
    if (lane == 0u) {
      warp_maximum = __bfloat162float(sinks[query_head]);
      warp_denominator = 1.0f;
    }
    __syncwarp();
    for (unsigned int token = first; token < token_count; ++token) {
      const unsigned int logical = token / block_size;
      if (logical >= block_count) return;
      const unsigned int page_token =
          block_table[logical] * block_size + token % block_size;
      const unsigned int page_base =
          (page_token * kv_heads + kv_head) * head_dim;
      const unsigned int current_base = kv_head * head_dim;
      const bool from_current = window > 0u && token + 1u == token_count;
      float dot = first_query *
                  clamped_routed_load_kv(
                      key_pages, current_keys, page_base + first_dim,
                      current_base + first_dim, from_current);
      dot = fmaf(
          second_query,
          clamped_routed_load_kv(
              key_pages, current_keys, page_base + second_dim,
              current_base + second_dim, from_current), dot);
      for (int offset = 16; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      if (lane == 0u)
        clamped_routed_sink_step(
            dot * scale, warp_maximum, warp_denominator, warp_alpha,
            warp_beta);
      __syncwarp();
      first_accumulator = fmaf(
          first_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + first_dim,
              current_base + first_dim, from_current) * warp_beta);
      second_accumulator = fmaf(
          second_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + second_dim,
              current_base + second_dim, from_current) * warp_beta);
      __syncwarp();
    }
    output[query_base + first_dim] =
        __float2bfloat16_rn(first_accumulator / warp_denominator);
    output[query_base + second_dim] =
        __float2bfloat16_rn(second_accumulator / warp_denominator);
    return;
  }
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
    const bool from_current = window > 0u && token + 1u == token_count;
    float dot = 0.0f;
    for (unsigned int dim = lane; dim < head_dim; dim += blockDim.x) {
      const unsigned int page_index =
          (page_token * kv_heads + kv_head) * head_dim + dim;
      const unsigned int current_index = kv_head * head_dim + dim;
      dot = fmaf(
          __bfloat162float(query[query_head * head_dim + dim]),
          clamped_routed_load_kv(
              key_pages, current_keys, page_index, current_index, from_current),
          dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      const unsigned int warps = blockDim.x / 32u;
      float sum = lane < warps ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0u) clamped_routed_sink_step(sum * scale, maximum, denominator, alpha, beta);
    }
    __syncthreads();
    if (lane < head_dim) {
      const unsigned int value_index = (page_token * kv_heads + kv_head) * head_dim + lane;
      const unsigned int current_index = kv_head * head_dim + lane;
      accumulator = fmaf(
          accumulator, alpha,
          clamped_routed_load_kv(
              value_pages, current_values, value_index, current_index,
              from_current) * beta);
    }
    __syncthreads();
  }
  if (lane < head_dim) output[query_head * head_dim + lane] =
      __float2bfloat16_rn(accumulator / denominator);
}

extern "C" __global__ void libmir_cuda_clamped_routed_paged_prefill_attention_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* current_keys,
    const __nv_bfloat16* current_values, const unsigned char* key_pages,
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
  if (head_dim == 64u && blockDim.x == 32u) {
    const unsigned int first_dim = lane * 2u;
    const unsigned int second_dim = first_dim + 1u;
    const unsigned int query_base =
        (query_token * query_heads + query_head) * head_dim;
    const float first_query =
        __bfloat162float(query[query_base + first_dim]);
    const float second_query =
        __bfloat162float(query[query_base + second_dim]);
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;
    __shared__ float warp_alpha, warp_beta, warp_denominator, warp_maximum;
    if (lane == 0u) {
      warp_maximum = __bfloat162float(sinks[query_head]);
      warp_denominator = 1.0f;
    }
    __syncwarp();
    for (unsigned int token = first; token < context; ++token) {
      const unsigned int logical = token / block_size;
      if (logical >= block_count) return;
      const unsigned int page_token =
          block_table[logical] * block_size + token % block_size;
      const unsigned int page_base =
          (page_token * kv_heads + kv_head) * head_dim;
      const bool from_current = window > 0u && token >= start_position;
      const unsigned int current_token =
          from_current ? token - start_position : 0u;
      const unsigned int current_base =
          (current_token * kv_heads + kv_head) * head_dim;
      float dot = first_query *
                  clamped_routed_load_kv(
                      key_pages, current_keys, page_base + first_dim,
                      current_base + first_dim, from_current);
      dot = fmaf(
          second_query,
          clamped_routed_load_kv(
              key_pages, current_keys, page_base + second_dim,
              current_base + second_dim, from_current), dot);
      for (int offset = 16; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      if (lane == 0u)
        clamped_routed_sink_step(
            dot * scale, warp_maximum, warp_denominator, warp_alpha,
            warp_beta);
      __syncwarp();
      first_accumulator = fmaf(
          first_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + first_dim,
              current_base + first_dim, from_current) * warp_beta);
      second_accumulator = fmaf(
          second_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + second_dim,
              current_base + second_dim, from_current) * warp_beta);
      __syncwarp();
    }
    output[query_base + first_dim] =
        __float2bfloat16_rn(first_accumulator / warp_denominator);
    output[query_base + second_dim] =
        __float2bfloat16_rn(second_accumulator / warp_denominator);
    return;
  }
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
    const bool from_current = window > 0u && token >= start_position;
    const unsigned int current_token =
        from_current ? token - start_position : 0u;
    float dot = 0.0f;
    for (unsigned int dim = lane; dim < head_dim; dim += blockDim.x) {
      const unsigned int q_index = (query_token * query_heads + query_head) * head_dim + dim;
      const unsigned int k_index = (page_token * kv_heads + kv_head) * head_dim + dim;
      const unsigned int current_index =
          (current_token * kv_heads + kv_head) * head_dim + dim;
      dot = fmaf(
          __bfloat162float(query[q_index]),
          clamped_routed_load_kv(
              key_pages, current_keys, k_index, current_index, from_current),
          dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      const unsigned int warps = blockDim.x / 32u;
      float sum = lane < warps ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0u) clamped_routed_sink_step(sum * scale, maximum, denominator, alpha, beta);
    }
    __syncthreads();
    if (lane < head_dim) {
      const unsigned int value_index = (page_token * kv_heads + kv_head) * head_dim + lane;
      const unsigned int current_index =
          (current_token * kv_heads + kv_head) * head_dim + lane;
      accumulator = fmaf(
          accumulator, alpha,
          clamped_routed_load_kv(
              value_pages, current_values, value_index, current_index,
              from_current) * beta);
    }
    __syncthreads();
  }
  if (lane < head_dim) {
    const unsigned int out = (query_token * query_heads + query_head) * head_dim + lane;
    output[out] = __float2bfloat16_rn(accumulator / denominator);
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_paged_batch_prefill_attention_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const __nv_bfloat16* current_keys, const __nv_bfloat16* current_values,
    const unsigned char* value_pages, const unsigned int* block_tables,
    const unsigned int* request_indices, const unsigned int* positions,
    const unsigned int* query_starts, const unsigned int* block_counts,
    const __nv_bfloat16* sinks,
    __nv_bfloat16* output, unsigned int query_tokens, unsigned int max_blocks,
    unsigned int block_size, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int window, float scale) {
  const unsigned int query_token = blockIdx.x / query_heads;
  const unsigned int query_head = blockIdx.x % query_heads;
  if (query_token >= query_tokens) return;
  const unsigned int request = request_indices[query_token];
  const unsigned int block_count = block_counts[request];
  const unsigned int table_start = request * max_blocks;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int context = positions[query_token] + 1u;
  const unsigned int first = window > 0u && context > window ? context - window : 0u;
  const unsigned int query_start = query_starts[request];
  const unsigned int start_position = positions[query_start];
  if (head_dim == 64u && blockDim.x == 32u) {
    const unsigned int first_dim = lane * 2u;
    const unsigned int second_dim = first_dim + 1u;
    const unsigned int query_base =
        (query_token * query_heads + query_head) * head_dim;
    const float first_query = __bfloat162float(query[query_base + first_dim]);
    const float second_query = __bfloat162float(query[query_base + second_dim]);
    float first_accumulator = 0.0f;
    float second_accumulator = 0.0f;
    __shared__ float warp_alpha, warp_beta, warp_denominator, warp_maximum;
    if (lane == 0u) {
      warp_maximum = __bfloat162float(sinks[query_head]);
      warp_denominator = 1.0f;
    }
    __syncwarp();
    for (unsigned int token = first; token < context; ++token) {
      const unsigned int logical = token / block_size;
      if (logical >= block_count) return;
      const unsigned int page_token =
          block_tables[table_start + logical] * block_size + token % block_size;
      const unsigned int page_base =
          (page_token * kv_heads + kv_head) * head_dim;
      const bool from_current = window > 0u && token >= start_position;
      const unsigned int current_token =
          from_current ? query_start + token - start_position : 0u;
      const unsigned int current_base =
          (current_token * kv_heads + kv_head) * head_dim;
      float dot = first_query *
                  clamped_routed_load_kv(
                      key_pages, current_keys, page_base + first_dim,
                      current_base + first_dim, from_current);
      dot = fmaf(
          second_query,
          clamped_routed_load_kv(
              key_pages, current_keys, page_base + second_dim,
              current_base + second_dim, from_current), dot);
      for (int offset = 16; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      if (lane == 0u)
        clamped_routed_sink_step(
            dot * scale, warp_maximum, warp_denominator, warp_alpha, warp_beta);
      __syncwarp();
      first_accumulator = fmaf(
          first_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + first_dim,
              current_base + first_dim, from_current) * warp_beta);
      second_accumulator = fmaf(
          second_accumulator, warp_alpha,
          clamped_routed_load_kv(
              value_pages, current_values, page_base + second_dim,
              current_base + second_dim, from_current) * warp_beta);
      __syncwarp();
    }
    output[query_base + first_dim] =
        __float2bfloat16_rn(first_accumulator / warp_denominator);
    output[query_base + second_dim] =
        __float2bfloat16_rn(second_accumulator / warp_denominator);
    return;
  }
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
    const unsigned int page_token =
        block_tables[table_start + logical] * block_size + token % block_size;
    const bool from_current = window > 0u && token >= start_position;
    const unsigned int current_token =
        from_current ? query_start + token - start_position : 0u;
    float dot = 0.0f;
    for (unsigned int dim = lane; dim < head_dim; dim += blockDim.x) {
      const unsigned int q_index =
          (query_token * query_heads + query_head) * head_dim + dim;
      const unsigned int k_index =
          (page_token * kv_heads + kv_head) * head_dim + dim;
      const unsigned int current_index =
          (current_token * kv_heads + kv_head) * head_dim + dim;
      dot = fmaf(
          __bfloat162float(query[q_index]),
          clamped_routed_load_kv(
              key_pages, current_keys, k_index, current_index, from_current),
          dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      const unsigned int warps = blockDim.x / 32u;
      float sum = lane < warps ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0u)
        clamped_routed_sink_step(
            sum * scale, maximum, denominator, alpha, beta);
    }
    __syncthreads();
    if (lane < head_dim) {
      const unsigned int value_index =
          (page_token * kv_heads + kv_head) * head_dim + lane;
      const unsigned int current_index =
          (current_token * kv_heads + kv_head) * head_dim + lane;
      accumulator = fmaf(
          accumulator, alpha,
          clamped_routed_load_kv(
              value_pages, current_values, value_index, current_index,
              from_current) * beta);
    }
    __syncthreads();
  }
  if (lane < head_dim) {
    const unsigned int out =
        (query_token * query_heads + query_head) * head_dim + lane;
    output[out] = __float2bfloat16_rn(accumulator / denominator);
  }
}
