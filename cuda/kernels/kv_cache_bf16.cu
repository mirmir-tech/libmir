#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ void store_page(unsigned char* pages, unsigned int index,
                           __nv_bfloat16 value) {
#ifdef LIBMIR_KV_FP8
  pages[index] = __nv_fp8_e4m3(__bfloat162float(value)).__x;
#else
  reinterpret_cast<__nv_bfloat16*>(pages)[index] = value;
#endif
}

extern "C" __global__ void libmir_cuda_store_paged_kv_bf16(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    unsigned char* key_pages, unsigned char* value_pages,
    unsigned int local_start, unsigned int token_count,
    unsigned int physical_block, unsigned int page_start,
    unsigned int block_size, unsigned int kv_heads,
    unsigned int key_head_dim, unsigned int value_head_dim) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int key_width = kv_heads * key_head_dim;
  const unsigned int value_width = kv_heads * value_head_dim;
  const unsigned int total = token_count * max(key_width, value_width);
  if (index >= total) return;

  const unsigned int token = index / max(key_width, value_width);
  const unsigned int feature = index % max(key_width, value_width);
  const unsigned int source_token = local_start + token;
  const unsigned int page_token = physical_block * block_size + page_start + token;
  if (feature < key_width) {
    store_page(key_pages, page_token * key_width + feature,
               keys[source_token * key_width + feature]);
  }
  if (feature < value_width) {
    store_page(value_pages, page_token * value_width + feature,
               values[source_token * value_width + feature]);
  }
}

extern "C" __global__ void libmir_cuda_store_paged_kv_batch_bf16(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    unsigned char* key_pages, unsigned char* value_pages,
    const unsigned int* block_tables, const unsigned int* token_counts,
    unsigned int batch_size, unsigned int max_blocks,
    unsigned int block_size, unsigned int kv_heads,
    unsigned int key_head_dim, unsigned int value_head_dim) {
  const unsigned int width = max(kv_heads * key_head_dim,
                                 kv_heads * value_head_dim);
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch_size * width) return;
  const unsigned int sequence = index / width;
  const unsigned int feature = index % width;
  const unsigned int token_count = token_counts[sequence];
  if (token_count == 0) return;
  const unsigned int position = token_count - 1u;
  const unsigned int logical_block = position / block_size;
  if (logical_block >= max_blocks) return;
  const unsigned int physical_block =
      block_tables[sequence * max_blocks + logical_block];
  const unsigned int page_token =
      physical_block * block_size + position % block_size;
  const unsigned int key_width = kv_heads * key_head_dim;
  const unsigned int value_width = kv_heads * value_head_dim;
  if (feature < key_width) {
    store_page(key_pages, page_token * key_width + feature,
               keys[sequence * key_width + feature]);
  }
  if (feature < value_width) {
    store_page(value_pages, page_token * value_width + feature,
               values[sequence * value_width + feature]);
  }
}
