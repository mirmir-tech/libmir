#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_stage_windowed_prefill_kv_bf16(
    const __nv_bfloat16* current_keys,
    const __nv_bfloat16* current_values,
    const unsigned char* ring_key_bytes,
    const unsigned char* ring_value_bytes,
    unsigned char* staged_key_bytes,
    unsigned char* staged_value_bytes,
    const unsigned int* ring_tables,
    const unsigned int* query_starts,
    const unsigned int* source_starts,
    const unsigned int* history_tokens,
    const unsigned int* context_tokens,
    unsigned int active_rows,
    unsigned int max_context_tokens,
    unsigned int ring_max_blocks,
    unsigned int staged_blocks_per_row,
    unsigned int block_size,
    unsigned int kv_heads,
    unsigned int head_dim) {
  const unsigned long long width = static_cast<unsigned long long>(kv_heads) * head_dim;
  const unsigned long long row_span =
      static_cast<unsigned long long>(max_context_tokens) * width;
  const unsigned long long total = static_cast<unsigned long long>(active_rows) * row_span;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const unsigned int row = static_cast<unsigned int>(index / row_span);
  const unsigned long long within_row = index % row_span;
  const unsigned int token = static_cast<unsigned int>(within_row / width);
  const unsigned int element = static_cast<unsigned int>(within_row % width);
  if (token >= context_tokens[row]) {
    return;
  }

  const auto* ring_keys = reinterpret_cast<const __nv_bfloat16*>(ring_key_bytes);
  const auto* ring_values = reinterpret_cast<const __nv_bfloat16*>(ring_value_bytes);
  auto* staged_keys = reinterpret_cast<__nv_bfloat16*>(staged_key_bytes);
  auto* staged_values = reinterpret_cast<__nv_bfloat16*>(staged_value_bytes);
  unsigned long long source_token;
  if (token < history_tokens[row]) {
    const unsigned int absolute = source_starts[row] + token;
    const unsigned int logical_block = absolute / block_size;
    const unsigned int physical_block = ring_tables[
        static_cast<unsigned long long>(row) * ring_max_blocks + logical_block];
    source_token = static_cast<unsigned long long>(physical_block) * block_size
        + absolute % block_size;
  } else {
    source_token = static_cast<unsigned long long>(query_starts[row])
        + token - history_tokens[row];
  }
  const unsigned long long staged_block =
      static_cast<unsigned long long>(row) * staged_blocks_per_row
      + token / block_size;
  const unsigned long long staged_token = staged_block * block_size + token % block_size;
  const unsigned long long source = source_token * width + element;
  const unsigned long long target = staged_token * width + element;
  if (token < history_tokens[row]) {
    staged_keys[target] = ring_keys[source];
    staged_values[target] = ring_values[source];
  } else {
    staged_keys[target] = current_keys[source];
    staged_values[target] = current_values[source];
  }
}
