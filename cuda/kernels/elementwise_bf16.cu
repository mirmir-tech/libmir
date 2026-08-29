#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_add_bf16(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(left[index]) + __bfloat162float(right[index]));
  }
}

extern "C" __global__ void libmir_cuda_multiply_scalar_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* scalar,
    __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * __bfloat162float(scalar[0]));
  }
}

extern "C" __global__ void libmir_cuda_gated_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, unsigned int elements, unsigned int activation) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = __bfloat162float(gate[index]);
  float activated;
  if (activation == 1u) {
    activated = value / (1.0f + expf(-value));
  } else {
    const float cube = value * value * value;
    activated = 0.5f * value *
        (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
  }
  const __nv_bfloat16 rounded = __float2bfloat16_rn(activated);
  output[index] = __float2bfloat16_rn(
      __bfloat162float(rounded) * __bfloat162float(up[index]));
}

extern "C" __global__ void libmir_cuda_packed_gated_bf16(
    const __nv_bfloat16* gate_input, const __nv_bfloat16* up_input,
    __nv_bfloat16* output, unsigned int columns, unsigned int elements,
    unsigned int layout, unsigned int activation) {
  const unsigned int row = blockIdx.y;
  const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int index = row * columns + column;
  if (column >= columns || index >= elements) return;
  const unsigned int gate_index = layout == 1u
      ? row * columns + column
      : layout == 2u
          ? row * columns * 2u + column * 2u
          : row * columns * 2u + column;
  const unsigned int up_index = layout == 1u
      ? row * columns + column
      : layout == 2u ? gate_index + 1u : gate_index + columns;
  const float value = __bfloat162float(gate_input[gate_index]);
  float activated;
  if (activation == 1u) {
    activated = value / (1.0f + expf(-value));
  } else {
    const float cube = value * value * value;
    activated = 0.5f * value *
        (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
  }
  const __nv_bfloat16 rounded = __float2bfloat16_rn(activated);
  output[index] = __float2bfloat16_rn(
      __bfloat162float(rounded) * __bfloat162float(up_input[up_index]));
}

extern "C" __global__ void libmir_cuda_weighted_reduce_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weights,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = tokens * columns;
  if (index >= total) return;
  const unsigned int token = index / columns;
  const unsigned int column = index % columns;
  float sum = 0.0f;
  for (unsigned int row = 0u; row < rows; ++row) {
    sum += __bfloat162float(input[(token * rows + row) * columns + column]) *
           __bfloat162float(weights[token * rows + row]);
  }
  output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void libmir_cuda_weighted_reduce_bucketed_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weights,
    const unsigned int* positions, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * columns) return;
  const unsigned int token = index / columns;
  const unsigned int column = index % columns;
  float sum = 0.0f;
  for (unsigned int row = 0u; row < rows; ++row) {
    const unsigned int assignment = token * rows + row;
    const unsigned int compact = positions[assignment];
    sum += __bfloat162float(input[compact * columns + column]) *
           __bfloat162float(weights[assignment]);
  }
  output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void libmir_cuda_weighted_reduce_bucketed_residual_shared_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weights,
    const unsigned int* positions, const __nv_bfloat16* residual,
    const __nv_bfloat16* shared, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * columns) return;
  const unsigned int token = index / columns;
  const unsigned int column = index % columns;
  float sum = 0.0f;
  for (unsigned int row = 0u; row < rows; ++row) {
    const unsigned int assignment = token * rows + row;
    const unsigned int compact = positions[assignment];
    sum += __bfloat162float(input[compact * columns + column]) *
           __bfloat162float(weights[assignment]);
  }
  const __nv_bfloat16 routed = __float2bfloat16_rn(sum);
  const __nv_bfloat16 moe = __float2bfloat16_rn(
      __bfloat162float(routed) + __bfloat162float(shared[index]));
  output[index] = __float2bfloat16_rn(
      __bfloat162float(residual[index]) + __bfloat162float(moe));
}
