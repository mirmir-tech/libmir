#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_embedding_bf16(
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    const unsigned int* selected, unsigned int selected_start,
    unsigned int tokens, unsigned int vocab, unsigned int hidden,
    float scale) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = tokens * hidden;
  if (index < total) {
    const unsigned int row = index / hidden;
    const unsigned int column = index % hidden;
    const unsigned int token = selected[selected_start + row];
    if (token < vocab) {
      output[index] = __float2bfloat16_rn(
          __bfloat162float(weight[token * hidden + column]) * scale);
    }
  }
}
