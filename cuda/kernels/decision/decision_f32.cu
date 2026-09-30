// Laya decision model kernels. Activations and weights are f32 so results
// match the fp32 CPU and Metal backends; only the token table stays f16.
#include <cuda_fp16.h>

extern "C" __global__ void libmir_decision_embed(
    const unsigned int* ids, const half* table, float* output, unsigned int rows,
    unsigned int width) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  const half* source = table + static_cast<size_t>(ids[row]) * width;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    output[static_cast<size_t>(row) * width + column] = __half2float(source[column]);
}

extern "C" __global__ void libmir_decision_add_bias(
    float* values, const float* bias, unsigned int elements, unsigned int width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) values[index] += bias[index % width];
}

extern "C" __global__ void libmir_decision_gelu(float* values, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float x = values[index];
    values[index] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
  }
}

extern "C" __global__ void libmir_decision_relu(float* values, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) values[index] = fmaxf(values[index], 0.0f);
}

// hidden[b, t, :] += kinds_table[kinds[b], :]
extern "C" __global__ void libmir_decision_add_kind(
    float* hidden, const unsigned int* kinds, const float* table, unsigned int elements,
    unsigned int length, unsigned int width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int column = index % width;
  const unsigned int row = index / (length * width);
  hidden[index] += table[kinds[row] * width + column];
}

extern "C" __global__ void libmir_decision_gather(
    const float* input, const unsigned int* rows, float* output, unsigned int count,
    unsigned int width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * width) return;
  const unsigned int row = index / width;
  output[index] = input[static_cast<size_t>(rows[row]) * width + index % width];
}
