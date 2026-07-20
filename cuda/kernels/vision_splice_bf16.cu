#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_embedding_splice_bf16(
    const __nv_bfloat16* image, __nv_bfloat16* hidden,
    unsigned int image_tokens, unsigned int hidden_width,
    unsigned int destination_start) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = image_tokens * hidden_width;
  if (index < elements) {
    hidden[destination_start * hidden_width + index] = image[index];
  }
}
