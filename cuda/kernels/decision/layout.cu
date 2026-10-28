// Kernels writing the input of the next matrix product, in f32 or bf16.
// Head-major attention operands live in one buffer laid out as
// `[pad][queries | keys | values][pad]`, each part `[heads][tokens][dim]`, so a
// strided batched product reaches every head and block with one stride; the
// zeroed pads keep windows at the edges inside the buffer.
#include <cuda_bf16.h>

__device__ inline void put(float* target, float value) { *target = value; }
__device__ inline void put(__nv_bfloat16* target, float value) {
  *target = __float2bfloat16(value);
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
template <typename Out>
__device__ void layer_norm(
    const float* input, const float* weight, const float* bias, Out* output,
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
  Out* target = output + static_cast<size_t>(row) * width;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x)
    put(target + column, fmaf((values[column] - mean) * scale, weight[column], bias[column]));
}

template <typename Out>
__device__ void geglu(const float* input, Out* output, unsigned int rows, unsigned int width) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(rows) * width) return;
  const size_t row = index / width;
  const size_t column = index % width;
  const float x = input[row * 2u * width + column];
  const float gate = input[row * 2u * width + width + column];
  put(output + index, 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)) * gate);
}

// One thread per rotate-half pair of one output row. `cosines` and `sines`
// are `[positions][dim / 2]` tables.
template <typename Out>
__device__ void split_heads(
    const float* qkv, const float* cosines, const float* sines, Out* output,
    unsigned int tokens, unsigned int length, unsigned int heads, unsigned int dim,
    unsigned int pad) {
  const unsigned int half = dim / 2u;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t parts = 3ull * heads * tokens;
  if (index >= (parts + 2u * pad) * half) return;
  const size_t row = index / half;
  const unsigned int pair = static_cast<unsigned int>(index % half);
  Out* target = output + row * dim;
  if (row < pad || row >= pad + parts) {
    put(target + pair, 0.0f);
    put(target + pair + half, 0.0f);
    return;
  }
  const size_t inner = row - pad;
  const unsigned int token = static_cast<unsigned int>(inner % tokens);
  const size_t part_head = inner / tokens;
  const float* source = qkv + static_cast<size_t>(token) * 3u * heads * dim + part_head * dim;
  float x = source[pair];
  float y = source[pair + half];
  if (part_head < 2u * heads) {
    const size_t angle = static_cast<size_t>(token % length) * half + pair;
    const float c = cosines[angle], s = sines[angle];
    const float rotated = fmaf(x, c, -(y * s));
    y = fmaf(y, c, x * s);
    x = rotated;
  }
  put(target + pair, x);
  put(target + pair + half, y);
}

// Token-major copy of a fused `[tokens, 3, heads, dim]` projection with its
// queries and keys rotated, one thread per rotate-half pair.
template <typename Out>
__device__ void rotate(
    const float* qkv, const float* cosines, const float* sines, Out* output,
    unsigned int tokens, unsigned int length, unsigned int heads, unsigned int dim) {
  const unsigned int half = dim / 2u;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(tokens) * 3u * heads * half) return;
  const unsigned int pair = static_cast<unsigned int>(index % half);
  const size_t part_head = (index / half) % (3u * heads);
  const size_t token = index / (half * 3u * heads);
  const size_t offset = (token * 3u * heads + part_head) * dim;
  float x = qkv[offset + pair];
  float y = qkv[offset + pair + half];
  if (part_head < 2u * heads) {
    const size_t angle = (token % length) * half + pair;
    const float c = cosines[angle], s = sines[angle];
    const float rotated = fmaf(x, c, -(y * s));
    y = fmaf(y, c, x * s);
    x = rotated;
  }
  put(output + offset + pair, x);
  put(output + offset + pair + half, y);
}

// `[heads][tokens][dim]` attention outputs to token-major `[tokens][heads × dim]`.
template <typename Out>
__device__ void merge_heads(
    const float* input, Out* output, unsigned int tokens, unsigned int heads, unsigned int dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(tokens) * heads * dim) return;
  const size_t token = index / (heads * dim);
  const size_t head = (index / dim) % heads;
  put(output + index, input[(head * tokens + token) * dim + index % dim]);
}

#define LIBMIR_DECISION_LAYOUT(SUFFIX, OUT)                                                  \
  extern "C" __global__ void libmir_decision_layer_norm_##SUFFIX(                            \
      const float* input, const float* weight, const float* bias, OUT* output,               \
      unsigned int rows, unsigned int width, float epsilon) {                                \
    layer_norm(input, weight, bias, output, rows, width, epsilon);                           \
  }                                                                                          \
  extern "C" __global__ void libmir_decision_geglu_##SUFFIX(                                 \
      const float* input, OUT* output, unsigned int rows, unsigned int width) {              \
    geglu(input, output, rows, width);                                                       \
  }                                                                                          \
  extern "C" __global__ void libmir_decision_split_rope_##SUFFIX(                            \
      const float* qkv, const float* cosines, const float* sines, OUT* output,               \
      unsigned int tokens, unsigned int length, unsigned int heads, unsigned int dim,        \
      unsigned int pad) {                                                                    \
    split_heads(qkv, cosines, sines, output, tokens, length, heads, dim, pad);               \
  }                                                                                          \
  extern "C" __global__ void libmir_decision_rotate_##SUFFIX(                               \
      const float* qkv, const float* cosines, const float* sines, OUT* output,               \
      unsigned int tokens, unsigned int length, unsigned int heads, unsigned int dim) {      \
    rotate(qkv, cosines, sines, output, tokens, length, heads, dim);                         \
  }                                                                                          \
  extern "C" __global__ void libmir_decision_merge_##SUFFIX(                                 \
      const float* input, OUT* output, unsigned int tokens, unsigned int heads,              \
      unsigned int dim) {                                                                    \
    merge_heads(input, output, tokens, heads, dim);                                          \
  }

LIBMIR_DECISION_LAYOUT(f32, float)
LIBMIR_DECISION_LAYOUT(bf16, __nv_bfloat16)
