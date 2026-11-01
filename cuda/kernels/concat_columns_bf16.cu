#include <cuda_bf16.h>

// output[row] = [left[row] | right[row]], both halves `columns` wide.
extern "C" __global__ void libmir_cuda_concat_columns_bf16(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int width = 2 * columns;
  if (index < rows * width) {
    const unsigned int row = index / width;
    const unsigned int column = index % width;
    output[index] = column < columns ? left[row * columns + column]
                                     : right[row * columns + column - columns];
  }
}
