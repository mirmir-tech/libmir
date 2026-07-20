#include <cuda_fp16.h>

__device__ void reduce(float& value, float* reductions) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x / 32u;
  if (lane == 0u) reductions[warp] = value;
  __syncthreads();
  if (warp == 0u) {
    value = lane < 8u ? reductions[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0u) reductions[0] = value;
  }
  __syncthreads();
  value = reductions[0];
}

extern "C" __global__ void libmir_cuda_encoder_embedding_norm_f16(
    const unsigned int* ids, const half* words, const half* types,
    const half* weight, const half* bias, half* output,
    unsigned int rows, unsigned int columns, float epsilon) {
  const unsigned int row = blockIdx.x;
  float sum = 0.0f, square = 0.0f;
  __shared__ float reductions[8];
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __half2float(words[ids[row] * columns + column]) + __half2float(types[column]);
    sum += value; square += value * value;
  }
  reduce(sum, reductions);
  __shared__ float mean;
  if (threadIdx.x == 0u) mean = sum / columns;
  __syncthreads();
  reduce(square, reductions);
  __shared__ float inverse;
  if (threadIdx.x == 0u) inverse = rsqrtf(square / columns - mean * mean + epsilon);
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __half2float(words[ids[row] * columns + column]) + __half2float(types[column]);
    output[row * columns + column] = __float2half((value - mean) * inverse *
        __half2float(weight[column]) + __half2float(bias[column]));
  }
}

extern "C" __global__ void libmir_cuda_encoder_bias_f16(
    const half* input, const half* bias, half* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * columns)
    output[index] = __float2half(__half2float(input[index]) + __half2float(bias[index % columns]));
}

extern "C" __global__ void libmir_cuda_encoder_residual_norm_f16(
    const half* left, const half* right, const half* weight, const half* bias,
    half* output, unsigned int rows, unsigned int columns, float epsilon) {
  const unsigned int row = blockIdx.x;
  float sum = 0.0f, square = 0.0f;
  __shared__ float reductions[8];
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __half2float(left[row * columns + column]) + __half2float(right[row * columns + column]);
    sum += value; square += value * value;
  }
  reduce(sum, reductions);
  __shared__ float mean;
  if (threadIdx.x == 0u) mean = sum / columns;
  __syncthreads();
  reduce(square, reductions);
  __shared__ float inverse;
  if (threadIdx.x == 0u) inverse = rsqrtf(square / columns - mean * mean + epsilon);
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float value = __half2float(left[index]) + __half2float(right[index]);
    output[index] = __float2half((value - mean) * inverse * __half2float(weight[column]) + __half2float(bias[column]));
  }
}

extern "C" __global__ void libmir_cuda_encoder_gated_gelu_f16(
    const half* input, half* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns) return;
  const unsigned int row = index / columns, column = index % columns;
  const float up = __half2float(input[row * columns * 2u + column]);
  const float gate = __half2float(input[row * columns * 2u + columns + column]);
  const float gelu = 0.5f * gate * (1.0f + erff(gate * 0.7071067812f));
  output[index] = __float2half(up * gelu);
}

extern "C" __global__ void libmir_cuda_encoder_tanh_bias_f16(
    const half* input, const half* bias, half* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = __float2half(tanhf(__half2float(input[index]) + __half2float(bias[index])));
}
