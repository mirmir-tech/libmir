#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_gated_attention_split_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* query,
    __nv_bfloat16* gate, unsigned int elements, unsigned int heads,
    unsigned int head_dim) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int width = heads * head_dim;
  const unsigned int row = index / width;
  const unsigned int within_row = index % width;
  const unsigned int head = within_row / head_dim;
  const unsigned int column = within_row % head_dim;
  const unsigned int source = row * width * 2 + head * head_dim * 2 + column;
  query[index] = input[source];
  gate[index] = input[source + head_dim];
}
