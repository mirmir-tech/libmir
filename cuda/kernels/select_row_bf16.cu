#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_select_row_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int row, unsigned int rows, unsigned int columns) {
  const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && column < columns) {
    output[column] = input[row * columns + column];
  }
}
