#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_spatial_merge_interpolate_bf16(
    const __nv_bfloat16* table, __nv_bfloat16* output,
    unsigned int grid_height, unsigned int grid_width,
    unsigned int source_side, unsigned int merge, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tokens = grid_height * grid_width;
  if (index >= tokens * hidden) return;
  const unsigned int token = index / hidden;
  const unsigned int feature = index % hidden;
  const unsigned int unit = merge * merge;
  const unsigned int block = token / unit;
  const unsigned int local = token % unit;
  const unsigned int blocks_wide = grid_width / merge;
  const unsigned int y = (block / blocks_wide) * merge + local / merge;
  const unsigned int x = (block % blocks_wide) * merge + local % merge;
  const float source_y = grid_height == 1 ? 0.0f
      : static_cast<float>(y * (source_side - 1)) / (grid_height - 1);
  const float source_x = grid_width == 1 ? 0.0f
      : static_cast<float>(x * (source_side - 1)) / (grid_width - 1);
  const unsigned int y0 = static_cast<unsigned int>(source_y);
  const unsigned int x0 = static_cast<unsigned int>(source_x);
  const unsigned int y1 = min(y0 + 1, source_side - 1);
  const unsigned int x1 = min(x0 + 1, source_side - 1);
  const float dy = source_y - y0;
  const float dx = source_x - x0;
  const unsigned int stride = source_side * hidden;
  const float top = (1.0f - dx) * __bfloat162float(table[y0 * stride + x0 * hidden + feature])
      + dx * __bfloat162float(table[y0 * stride + x1 * hidden + feature]);
  const float bottom = (1.0f - dx) * __bfloat162float(table[y1 * stride + x0 * hidden + feature])
      + dx * __bfloat162float(table[y1 * stride + x1 * hidden + feature]);
  output[index] = __float2bfloat16_rn((1.0f - dy) * top + dy * bottom);
}

extern "C" __global__ void libmir_cuda_spatial_merge_qkv_split_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* value,
    unsigned int tokens, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int token = index / hidden;
  const unsigned int feature = index % hidden;
  const unsigned int source = token * 3 * hidden + feature;
  query[index] = input[source];
  key[index] = input[source + hidden];
  value[index] = input[source + 2 * hidden];
}

extern "C" __global__ void libmir_cuda_spatial_merge_rope_bf16(
    const __nv_bfloat16* input, const unsigned int* positions,
    __nv_bfloat16* output, unsigned int tokens, unsigned int heads,
    unsigned int head_dim) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = tokens * heads * head_dim;
  if (index >= elements) return;
  const unsigned int dimension = index % head_dim;
  const unsigned int token = index / (heads * head_dim);
  const unsigned int half = head_dim / 2;
  const unsigned int quarter = head_dim / 4;
  const unsigned int spatial = dimension % half;
  const unsigned int axis = spatial < quarter ? 0 : 1;
  const unsigned int frequency = spatial % quarter;
  const float exponent = -2.0f * frequency / half;
  const float angle = static_cast<float>(positions[token * 2 + axis])
      * powf(10000.0f, exponent);
  const unsigned int base = index - dimension;
  const unsigned int pair = dimension < half ? dimension + half : dimension - half;
  float rotated = __bfloat162float(input[base + pair]);
  if (dimension < half) rotated = -rotated;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(value * cosf(angle) + rotated * sinf(angle));
}
