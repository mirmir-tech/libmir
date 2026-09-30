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

__device__ float block_sum(float value, float* shared) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0u) shared[warp] = value;
  __syncthreads();
  const unsigned int warps = (blockDim.x + 31u) >> 5;
  value = lane < warps ? shared[lane] : 0.0f;
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  __syncthreads();
  return value;
}

// One block per row; two passes like the host implementations.
extern "C" __global__ void libmir_decision_layer_norm(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int rows, unsigned int width, float epsilon) {
  __shared__ float partial[32];
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  const float* values = input + static_cast<size_t>(row) * width;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    sum += values[column];
  const float mean = block_sum(sum, partial) / static_cast<float>(width);
  float squares = 0.0f;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float centered = values[column] - mean;
    squares += centered * centered;
  }
  const float variance = block_sum(squares, partial) / static_cast<float>(width);
  const float scale = rsqrtf(variance + epsilon);
  float* target = output + static_cast<size_t>(row) * width;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    target[column] = fmaf((values[column] - mean) * scale, weight[column], bias[column]);
}

extern "C" __global__ void libmir_decision_add_bias(
    float* values, const float* bias, unsigned int elements, unsigned int width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) values[index] += bias[index % width];
}

// Rotate-half RoPE of the first `rotated_heads` heads of every token.
extern "C" __global__ void libmir_decision_rope(
    float* values, unsigned int tokens, unsigned int length, unsigned int width,
    unsigned int rotated_heads, unsigned int head_dim, float theta) {
  const unsigned int half_dim = head_dim / 2u;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * rotated_heads * half_dim) return;
  const unsigned int pair = index % half_dim;
  const unsigned int head = (index / half_dim) % rotated_heads;
  const unsigned int token = index / (half_dim * rotated_heads);
  const float inverse =
      1.0f / powf(theta, static_cast<float>(2u * pair) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(token % length) * inverse;
  float* base = values + static_cast<size_t>(token) * width + head * head_dim;
  const float x = base[pair];
  const float y = base[pair + half_dim];
  const float c = cosf(angle);
  const float s = sinf(angle);
  base[pair] = fmaf(x, c, -(y * s));
  base[pair + half_dim] = fmaf(y, c, x * s);
}

extern "C" __global__ void libmir_decision_geglu(
    const float* input, float* output, unsigned int rows, unsigned int width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  const unsigned int row = index / width;
  const unsigned int column = index % width;
  const float x = input[static_cast<size_t>(row) * 2u * width + column];
  const float gate = input[static_cast<size_t>(row) * 2u * width + width + column];
  output[index] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)) * gate;
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
