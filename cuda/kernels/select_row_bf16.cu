#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_select_row_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int row, unsigned int rows, unsigned int columns) {
  const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && column < columns) {
    output[column] = input[row * columns + column];
  }
}

extern "C" __global__ void libmir_cuda_copy_rows_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int input_start, unsigned int output_start,
    unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = rows * columns;
  if (index < elements) {
    const unsigned int row = index / columns;
    const unsigned int column = index % columns;
    output[(output_start + row) * columns + column] =
        input[(input_start + row) * columns + column];
  }
}

extern "C" __global__ void libmir_cuda_gather_rows_bf16(
    const __nv_bfloat16* input, const unsigned int* indices,
    __nv_bfloat16* output, unsigned int input_rows,
    unsigned int output_rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = output_rows * columns;
  if (index < elements) {
    const unsigned int output_row = index / columns;
    const unsigned int column = index % columns;
    const unsigned int input_row = indices[output_row];
    if (input_row < input_rows) {
      output[index] = input[input_row * columns + column];
    }
  }
}
