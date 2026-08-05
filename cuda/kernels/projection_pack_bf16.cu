#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_projection_pack_split2_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* first,
    __nv_bfloat16* second, unsigned int first_columns,
    unsigned int second_columns, unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int columns = first_columns + second_columns;
  const unsigned int elements = tokens * columns;
  if (index >= elements) return;
  const unsigned int token = index / columns;
  const unsigned int column = index - token * columns;
  if (column < first_columns) {
    first[token * first_columns + column] = input[index];
  } else {
    second[token * second_columns + column - first_columns] = input[index];
  }
}

extern "C" __global__ void libmir_cuda_projection_pack_split3_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* first,
    __nv_bfloat16* second, __nv_bfloat16* third,
    unsigned int first_columns, unsigned int second_columns,
    unsigned int third_columns, unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int columns = first_columns + second_columns + third_columns;
  const unsigned int elements = tokens * columns;
  if (index >= elements) return;
  const unsigned int token = index / columns;
  const unsigned int column = index - token * columns;
  if (column < first_columns) {
    first[token * first_columns + column] = input[index];
  } else if (column < first_columns + second_columns) {
    second[token * second_columns + column - first_columns] = input[index];
  } else {
    third[token * third_columns + column - first_columns - second_columns] = input[index];
  }
}
