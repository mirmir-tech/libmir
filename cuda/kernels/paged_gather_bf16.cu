#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_gather_paged_kv_bf16(
    const unsigned char* key_pages, const unsigned char* value_pages,
    const unsigned int* block_table, unsigned char* keys, unsigned char* values,
    unsigned int context_tokens, unsigned int block_count,
    unsigned int block_size, unsigned int key_width,
    unsigned int value_width) {
  const unsigned int token = blockIdx.x;
  if (token >= context_tokens) return;
  const unsigned int logical_block = token / block_size;
  if (logical_block >= block_count) return;
  const unsigned int physical_token =
      block_table[logical_block] * block_size + token % block_size;
  const __nv_bfloat16* source_keys =
      reinterpret_cast<const __nv_bfloat16*>(key_pages);
  const __nv_bfloat16* source_values =
      reinterpret_cast<const __nv_bfloat16*>(value_pages);
  __nv_bfloat16* target_keys = reinterpret_cast<__nv_bfloat16*>(keys);
  __nv_bfloat16* target_values = reinterpret_cast<__nv_bfloat16*>(values);
  for (unsigned int element = threadIdx.x; element < key_width;
       element += blockDim.x) {
    target_keys[token * key_width + element] =
        source_keys[physical_token * key_width + element];
  }
  for (unsigned int element = threadIdx.x; element < value_width;
       element += blockDim.x) {
    target_values[token * value_width + element] =
        source_values[physical_token * value_width + element];
  }
}

extern "C" __global__ void libmir_cuda_gather_paged_kv_batch_bf16(
    const unsigned char* key_pages, const unsigned char* value_pages,
    const unsigned int* block_tables, const unsigned int* context_starts,
    unsigned char* keys, unsigned char* values, unsigned int batch_size,
    unsigned int max_blocks, unsigned int block_size,
    unsigned int key_width, unsigned int value_width) {
  const unsigned int sequence = blockIdx.y;
  const unsigned int token = blockIdx.x;
  if (sequence >= batch_size) return;
  const unsigned int target_start = context_starts[sequence];
  const unsigned int context_tokens =
      context_starts[sequence + 1] - target_start;
  if (token >= context_tokens) return;
  const unsigned int logical_block = token / block_size;
  if (logical_block >= max_blocks) return;
  const unsigned int physical_token =
      block_tables[sequence * max_blocks + logical_block] * block_size +
      token % block_size;
  const __nv_bfloat16* source_keys =
      reinterpret_cast<const __nv_bfloat16*>(key_pages);
  const __nv_bfloat16* source_values =
      reinterpret_cast<const __nv_bfloat16*>(value_pages);
  __nv_bfloat16* target_keys = reinterpret_cast<__nv_bfloat16*>(keys);
  __nv_bfloat16* target_values = reinterpret_cast<__nv_bfloat16*>(values);
  for (unsigned int element = threadIdx.x; element < key_width;
       element += blockDim.x) {
    target_keys[(target_start + token) * key_width + element] =
        source_keys[physical_token * key_width + element];
  }
  for (unsigned int element = threadIdx.x; element < value_width;
       element += blockDim.x) {
    target_values[(target_start + token) * value_width + element] =
        source_values[physical_token * value_width + element];
  }
}
