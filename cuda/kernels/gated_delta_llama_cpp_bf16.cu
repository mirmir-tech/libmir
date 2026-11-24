// llama.cpp's GGUF Gated DeltaNet layout (`LinearAttentionLayout::LlamaCpp`)
// brought back to the Hugging Face one the kernels read: `ssm_a` holds
// `-exp(A_log)`, and value heads are tiled across key heads
// (`[K0, K1, .., K0, K1, ..]`) when there are more value than key heads.
#include <cuda_bf16.h>

// `A_log = ln(-ssm_a)`, in place.
extern "C" __global__ void libmir_cuda_gated_delta_decay_log_bf16(
    __nv_bfloat16* values, unsigned int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  values[index] = __float2bfloat16_rn(logf(-__bfloat162float(values[index])));
}

// Row `r` of `output` is row `rows[r]` of `input`, `row_bytes` each: the
// grouped order of stored value-head rows, whatever their storage.
extern "C" __global__ void libmir_cuda_gated_delta_gather_rows(
    const unsigned char* input, unsigned char* output, const unsigned int* rows,
    unsigned int count, unsigned int row_bytes) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(count) * row_bytes) return;
  const unsigned long long row = index / row_bytes;
  const unsigned long long byte = index % row_bytes;
  output[index] = input[static_cast<unsigned long long>(rows[row]) * row_bytes + byte];
}

// `[tokens, key_heads * per_key, head_dim]` grouped values in the tiled
// order the stored output projection's columns expect: grouped head
// `k * per_key + j` goes to `j * key_heads + k`.
extern "C" __global__ void libmir_cuda_gated_delta_tile_values_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int tokens,
    unsigned int key_heads, unsigned int per_key, unsigned int head_dim) {
  const unsigned int heads = key_heads * per_key;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(tokens) * heads * head_dim) return;
  const unsigned long long token = index / (static_cast<unsigned long long>(heads) * head_dim);
  const unsigned int within = index % (static_cast<unsigned long long>(heads) * head_dim);
  const unsigned int grouped = within / head_dim;
  const unsigned int tiled = (grouped % per_key) * key_heads + grouped / per_key;
  output[(token * heads + tiled) * head_dim + within % head_dim] = input[index];
}
