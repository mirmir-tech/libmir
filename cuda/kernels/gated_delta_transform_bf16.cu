#include <cuda_bf16.h>

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

extern "C" __global__ void libmir_cuda_gated_delta_split_normalize_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* normalized_query, __nv_bfloat16* normalized_key,
    __nv_bfloat16* value, unsigned int tokens, unsigned int key_heads,
    unsigned int value_heads, unsigned int key_dim, unsigned int value_dim,
    float epsilon) {
  const unsigned int row = blockIdx.x;
  const unsigned int rows = tokens * key_heads;
  if (row >= rows) return;
  const unsigned int token = row / key_heads;
  const unsigned int head = row % key_heads;
  const unsigned int key_width = key_heads * key_dim;
  const unsigned int value_width = value_heads * value_dim;
  const unsigned int input_width = 2 * key_width + value_width;
  const unsigned int query_offset = token * input_width + head * key_dim;
  const unsigned int key_offset = query_offset + key_width;
  __shared__ float query_warps[8];
  __shared__ float key_warps[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  float query_sum = 0.0f;
  float key_sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < key_dim; column += blockDim.x) {
    const float query_value = __bfloat162float(input[query_offset + column]);
    const float key_value = __bfloat162float(input[key_offset + column]);
    query_sum = fmaf(query_value, query_value, query_sum);
    key_sum = fmaf(key_value, key_value, key_sum);
  }
  const unsigned int heads_per_key = value_heads / key_heads;
  const unsigned int head_values = heads_per_key * value_dim;
  const unsigned int value_column = head * head_values;
  for (unsigned int column = threadIdx.x; column < head_values; column += blockDim.x) {
    value[token * value_width + value_column + column] =
        input[token * input_width + 2 * key_width + value_column + column];
  }
  query_sum = warp_sum(query_sum);
  key_sum = warp_sum(key_sum);
  if (lane == 0) {
    query_warps[warp] = query_sum;
    key_warps[warp] = key_sum;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int active_warps = blockDim.x / 32u;
    query_sum = lane < active_warps ? query_warps[lane] : 0.0f;
    key_sum = lane < active_warps ? key_warps[lane] : 0.0f;
    query_sum = warp_sum(query_sum);
    key_sum = warp_sum(key_sum);
    if (lane == 0) {
      query_warps[0] = rsqrtf(query_sum / key_dim + epsilon);
      key_warps[0] = rsqrtf(key_sum / key_dim + epsilon);
    }
  }
  __syncthreads();
  const float query_scale = 1.0f / static_cast<float>(key_dim);
  const float key_scale = rsqrtf(static_cast<float>(key_dim));
  for (unsigned int column = threadIdx.x; column < key_dim; column += blockDim.x) {
    const unsigned int index = row * key_dim + column;
    const __nv_bfloat16 query_unit = __float2bfloat16_rn(
        __bfloat162float(input[query_offset + column]) * query_warps[0]);
    const __nv_bfloat16 key_unit = __float2bfloat16_rn(
        __bfloat162float(input[key_offset + column]) * key_warps[0]);
    normalized_query[index] = __float2bfloat16_rn(
        __bfloat162float(query_unit) * query_scale);
    normalized_key[index] = __float2bfloat16_rn(
        __bfloat162float(key_unit) * key_scale);
  }
}

extern "C" __global__ void libmir_cuda_gated_delta_norm_gate_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, float epsilon,
    float weight_shift, unsigned int value_heads,
    unsigned int gate_stride, unsigned int gate_offset) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float warps[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[row * columns + column]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) warps[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < blockDim.x / 32u ? warps[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) warps[0] = rsqrtf(sum / columns + epsilon);
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float scale = __bfloat162float(weight[column]) + weight_shift;
    const __nv_bfloat16 normalized = __float2bfloat16_rn(
        __bfloat162float(input[index]) * warps[0] * scale);
    const unsigned int token = row / value_heads;
    const unsigned int head = row % value_heads;
    const unsigned int gate_index = token * gate_stride + gate_offset + head * columns + column;
    const float gate_value = __bfloat162float(gate[gate_index]);
    const float activated = gate_value / (1.0f + __expf(-gate_value));
    output[index] = __float2bfloat16_rn(
        activated * __bfloat162float(normalized));
  }
}
