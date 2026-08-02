#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ float load_page(const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

extern "C" __global__ void libmir_cuda_paged_prefill_attention_batch_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_tables,
    const unsigned int* block_counts, const unsigned int* request_indices,
    const unsigned int* positions, __nv_bfloat16* output,
    unsigned int query_tokens, unsigned int batch_size,
    unsigned int max_blocks, unsigned int block_size,
    unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int value_head_dim,
    unsigned int window, float scale) {
  const unsigned int packed_query = blockIdx.x / query_heads;
  const unsigned int query_head = blockIdx.x % query_heads;
  if (packed_query >= query_tokens) return;
  const unsigned int sequence = request_indices[packed_query];
  if (sequence >= batch_size) return;
  const unsigned int* block_table = block_tables + sequence * max_blocks;
  const unsigned int block_count = block_counts[sequence];
  const unsigned int query_position = positions[packed_query];
  const unsigned int context = query_position + 1u;
  const unsigned int first =
      window > 0 && context > window ? context - window : 0u;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  constexpr unsigned int token_tile = 8u;
  float accumulators[2] = {0.0f, 0.0f};
  const unsigned int warp = lane / 32u;
  const unsigned int warp_lane = lane & 31u;
  __shared__ float scores[token_tile];
  __shared__ float weights[token_tile];
  __shared__ float alpha;
  __shared__ float denominator;
  __shared__ float maximum;
  if (lane == 0) {
    denominator = 0.0f;
    maximum = -3.402823466e+38F;
  }
  __syncthreads();

  for (unsigned int token_base = first; token_base < context;
       token_base += token_tile) {
    const unsigned int token = token_base + warp;
    float dot = 0.0f;
    if (token < context) {
      const unsigned int logical_block = token / block_size;
      if (logical_block >= block_count) return;
      const unsigned int page_token =
          block_table[logical_block] * block_size + token % block_size;
      for (unsigned int dimension = warp_lane; dimension < head_dim;
           dimension += 32u) {
        const unsigned int q_index =
            (packed_query * query_heads + query_head) * head_dim + dimension;
        const unsigned int k_index =
            (page_token * kv_heads + kv_head) * head_dim + dimension;
        dot = fmaf(
            __bfloat162float(query[q_index]), load_page(key_pages, k_index), dot);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if (warp_lane == 0u) {
      scores[warp] = token < context ? dot * scale : -3.402823466e+38F;
    }
    __syncthreads();
    if (lane == 0u) {
      float batch_maximum = scores[0];
#pragma unroll
      for (unsigned int index = 1; index < token_tile; ++index) {
        batch_maximum = fmaxf(batch_maximum, scores[index]);
      }
      const float next_maximum = fmaxf(maximum, batch_maximum);
      alpha = expf(maximum - next_maximum);
      denominator *= alpha;
#pragma unroll
      for (unsigned int index = 0; index < token_tile; ++index) {
        const float weight = expf(scores[index] - next_maximum);
        weights[index] = weight;
        denominator += weight;
      }
      maximum = next_maximum;
    }
    __syncthreads();
    unsigned int local = 0;
    for (unsigned int dimension = lane; dimension < value_head_dim;
         dimension += blockDim.x, ++local) {
      float weighted_value = 0.0f;
#pragma unroll
      for (unsigned int index = 0; index < token_tile; ++index) {
        const unsigned int value_token = token_base + index;
        if (value_token < context) {
          const unsigned int logical_block = value_token / block_size;
          const unsigned int page_token =
              block_table[logical_block] * block_size +
              value_token % block_size;
          const unsigned int v_index =
              (page_token * kv_heads + kv_head) * value_head_dim + dimension;
          weighted_value = fmaf(
              load_page(value_pages, v_index), weights[index], weighted_value);
        }
      }
      accumulators[local] = fmaf(accumulators[local], alpha, weighted_value);
    }
  }
  unsigned int local = 0;
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x, ++local) {
    const unsigned int out =
        (packed_query * query_heads + query_head) * value_head_dim + dimension;
    output[out] = __float2bfloat16_rn(accumulators[local] / denominator);
  }
}
