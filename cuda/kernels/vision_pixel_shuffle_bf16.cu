#include <cuda_bf16.h>

// Prefix tokens, then each patch plus its position: the square table of
// `grid`² positions resized bilinearly (half-pixel centres, clamped at the
// first texel) to the larger patch-grid side and read at the patch.
extern "C" __global__ void libmir_cuda_pixel_shuffle_embed_bf16(
    const __nv_bfloat16* patches, const __nv_bfloat16* table,
    const __nv_bfloat16* prefix, __nv_bfloat16* output, unsigned int rows,
    unsigned int cols, unsigned int grid, unsigned int prefix_tokens,
    unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tokens = prefix_tokens + rows * cols;
  if (index >= tokens * hidden) return;
  const unsigned int token = index / hidden;
  const unsigned int feature = index % hidden;
  if (token < prefix_tokens) {
    output[index] = prefix[token * hidden + feature];
    return;
  }
  const unsigned int patch = token - prefix_tokens;
  const unsigned int row = patch / cols;
  const unsigned int col = patch % cols;
  const unsigned int side = rows > cols ? rows : cols;
  const float scale = static_cast<float>(grid) / static_cast<float>(side);
  const float source_y = fmaxf((row + 0.5f) * scale - 0.5f, 0.0f);
  const float source_x = fmaxf((col + 0.5f) * scale - 0.5f, 0.0f);
  const unsigned int y0 = static_cast<unsigned int>(source_y);
  const unsigned int x0 = static_cast<unsigned int>(source_x);
  const unsigned int y1 = y0 + 1 < grid ? y0 + 1 : y0;
  const unsigned int x1 = x0 + 1 < grid ? x0 + 1 : x0;
  const float dy = source_y - y0;
  const float dx = source_x - x0;
  const float top =
      (1.0f - dx) * __bfloat162float(table[(y0 * grid + x0) * hidden + feature]) +
      dx * __bfloat162float(table[(y0 * grid + x1) * hidden + feature]);
  const float bottom =
      (1.0f - dx) * __bfloat162float(table[(y1 * grid + x0) * hidden + feature]) +
      dx * __bfloat162float(table[(y1 * grid + x1) * hidden + feature]);
  const float position = (1.0f - dy) * top + dy * bottom;
  output[index] = __float2bfloat16_rn(
      __bfloat162float(patches[patch * hidden + feature]) + position);
}

// Folds each `factor`×`factor` block of patches (after the prefix) into one
// token: channel blocks run over the block's rows, then its columns.
extern "C" __global__ void libmir_cuda_pixel_shuffle_fold_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int rows,
    unsigned int cols, unsigned int factor, unsigned int prefix_tokens,
    unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int folded_cols = cols / factor;
  const unsigned int width = factor * factor * hidden;
  if (index >= (rows / factor) * folded_cols * width) return;
  const unsigned int token = index / width;
  const unsigned int channel = index % width;
  const unsigned int block = channel / hidden;
  const unsigned int feature = channel % hidden;
  const unsigned int row = (token / folded_cols) * factor + block / factor;
  const unsigned int col = (token % folded_cols) * factor + block % factor;
  output[index] = input[(prefix_tokens + row * cols + col) * hidden + feature];
}

// Root-mean-square norm with a learned scale and no bias; one block of
// 256 threads per row.
extern "C" __global__ void libmir_cuda_pixel_shuffle_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    float epsilon) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float squares = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]);
    squares = fmaf(value, value, squares);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    squares += __shfl_down_sync(0xffffffffu, squares, offset);
  }
  __shared__ float warp_squares[8];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x / 32u;
  if (lane == 0u) warp_squares[warp] = squares;
  __syncthreads();
  if (warp == 0u) {
    squares = lane < blockDim.x / 32u ? warp_squares[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      squares += __shfl_down_sync(0xffffffffu, squares, offset);
    }
    if (lane == 0u) warp_squares[0] = rsqrtf(squares / columns + epsilon);
  }
  __syncthreads();
  const float inverse = warp_squares[0];
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse * __bfloat162float(weight[column]));
  }
}

// max(x, 0)².
extern "C" __global__ void libmir_cuda_pixel_shuffle_relu_square_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = fmaxf(__bfloat162float(input[index]), 0.0f);
  output[index] = __float2bfloat16_rn(value * value);
}
