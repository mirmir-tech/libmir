#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_convert_f32_bf16(
    const float* input, __nv_bfloat16* output, unsigned int elements,
    float scale, float bias) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(fmaf(input[index], scale, bias));
  }
}

extern "C" __global__ void libmir_cuda_vision_binary_bf16(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    __nv_bfloat16* output, unsigned int elements, unsigned int operation) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float a = __bfloat162float(left[index]);
  const float b = __bfloat162float(right[index]);
  output[index] = __float2bfloat16_rn(operation == 0 ? a + b : a * b);
}

extern "C" __global__ void libmir_cuda_vision_bias_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = rows * columns;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) + __bfloat162float(bias[index % columns]));
  }
}

extern "C" __global__ void libmir_cuda_vision_gelu_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int elements, unsigned int approximate) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float x = __bfloat162float(input[index]);
  const float value = approximate
      ? 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
                                (x + 0.044715f * x * x * x)))
      : 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
  output[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void libmir_cuda_vision_layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, float epsilon) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  float squares = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]);
    sum += value;
    squares = fmaf(value, value, squares);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
    squares += __shfl_down_sync(0xffffffffu, squares, offset);
  }
  __shared__ float warp_sum[8];
  __shared__ float warp_squares[8];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x / 32u;
  if (lane == 0u) {
    warp_sum[warp] = sum;
    warp_squares[warp] = squares;
  }
  __syncthreads();
  if (warp == 0u) {
    sum = lane < blockDim.x / 32u ? warp_sum[lane] : 0.0f;
    squares = lane < blockDim.x / 32u ? warp_squares[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
      squares += __shfl_down_sync(0xffffffffu, squares, offset);
    }
    if (lane == 0u) {
      const float mean = sum / columns;
      warp_sum[0] = mean;
      warp_squares[0] = rsqrtf(fmaxf(0.0f, squares / columns - mean * mean) + epsilon);
    }
  }
  __syncthreads();
  const float mean = warp_sum[0];
  const float inverse = warp_squares[0];
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float normalized = (__bfloat162float(input[index]) - mean) * inverse;
    output[index] = __float2bfloat16_rn(
        normalized * __bfloat162float(weight[column]) +
        __bfloat162float(bias[column]));
  }
}

extern "C" __global__ void libmir_cuda_vision_position_add_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* table,
    const unsigned int* positions, __nv_bfloat16* output,
    unsigned int tokens, unsigned int positions_per_axis, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = tokens * hidden;
  if (index >= elements) return;
  const unsigned int token = index / hidden;
  const unsigned int column = index % hidden;
  const unsigned int x = positions[token * 2];
  const unsigned int y = positions[token * 2 + 1];
  if (x >= positions_per_axis || y >= positions_per_axis) return;
  const unsigned int y_offset = positions_per_axis * hidden;
  const float position = __bfloat162float(table[x * hidden + column]) +
                         __bfloat162float(table[y_offset + y * hidden + column]);
  output[index] = __float2bfloat16_rn(__bfloat162float(input[index]) + position);
}
