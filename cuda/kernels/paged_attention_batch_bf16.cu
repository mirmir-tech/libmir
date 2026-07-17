#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ float load_batch_page(const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_batch_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_tables,
    const unsigned int* token_counts, const unsigned int* block_counts,
    __nv_bfloat16* output, unsigned int batch_size,
    unsigned int max_blocks, unsigned int block_size,
    unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int value_head_dim,
    unsigned int window, float scale) {
  const unsigned int sequence = blockIdx.y;
  const unsigned int query_head = blockIdx.x;
  if (sequence >= batch_size || query_head >= query_heads) return;
  const unsigned int token_count = token_counts[sequence];
  const unsigned int block_count = block_counts[sequence];
  if (token_count == 0 || block_count == 0 || block_count > max_blocks) return;

  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int first = window > 0 && token_count > window
      ? token_count - window : 0;
  const unsigned int query_offset = sequence * query_heads * head_dim;
  const unsigned int table_offset = sequence * max_blocks;
  const unsigned int output_offset = sequence * query_heads * value_head_dim;
  float accumulators[2] = {0.0f, 0.0f};
  __shared__ float warp_sums[8];
  __shared__ float alpha;
  __shared__ float beta;
  __shared__ float denominator;
  __shared__ float maximum;
  if (lane == 0) {
    denominator = 0.0f;
    maximum = -3.402823466e+38F;
  }
  __syncthreads();

  for (unsigned int token = first; token < token_count; ++token) {
    const unsigned int logical_block = token / block_size;
    if (logical_block >= block_count) return;
    const unsigned int physical_block = block_tables[table_offset + logical_block];
    const unsigned int page_token = physical_block * block_size + token % block_size;
    float dot = 0.0f;
    for (unsigned int dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
      const unsigned int q_index = query_offset + query_head * head_dim + dimension;
      const unsigned int k_index =
          (page_token * kv_heads + kv_head) * head_dim + dimension;
      dot = fmaf(__bfloat162float(query[q_index]),
          load_batch_page(key_pages, k_index), dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      float sum = lane < 8u ? warp_sums[lane] : 0.0f;
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
          load_batch_page(value_pages, v_index) * beta);
    }
    __syncthreads();
  }
  unsigned int local = 0;
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x, ++local) {
    const unsigned int out = output_offset + query_head * value_head_dim + dimension;
    output[out] = __float2bfloat16_rn(accumulators[local] / denominator);
  }
}
