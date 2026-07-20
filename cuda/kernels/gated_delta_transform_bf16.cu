#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_gated_delta_split_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* value, unsigned int tokens,
    unsigned int key_width, unsigned int value_width) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int width = 2 * key_width + value_width;
  if (index >= tokens * width) return;
  const unsigned int token = index / width;
  const unsigned int column = index % width;
  if (column < key_width) {
    query[token * key_width + column] = input[index];
  } else if (column < 2 * key_width) {
    key[token * key_width + column - key_width] = input[index];
  } else {
    value[token * value_width + column - 2 * key_width] = input[index];
  }
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

extern "C" __global__ void libmir_cuda_gated_delta_normalize_qk_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    __nv_bfloat16* normalized_query, __nv_bfloat16* normalized_key,
    unsigned int rows, unsigned int columns, float epsilon) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float query_warps[8];
  __shared__ float key_warps[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  float query_sum = 0.0f;
  float key_sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float query_value = __bfloat162float(query[row * columns + column]);
    const float key_value = __bfloat162float(key[row * columns + column]);
    query_sum = fmaf(query_value, query_value, query_sum);
    key_sum = fmaf(key_value, key_value, key_sum);
  }
  query_sum = warp_sum(query_sum);
  key_sum = warp_sum(key_sum);
  if (lane == 0) {
    query_warps[warp] = query_sum;
    key_warps[warp] = key_sum;
  }
  __syncthreads();
  if (warp == 0) {
    query_sum = lane < 8 ? query_warps[lane] : 0.0f;
    key_sum = lane < 8 ? key_warps[lane] : 0.0f;
    query_sum = warp_sum(query_sum);
    key_sum = warp_sum(key_sum);
    if (lane == 0) {
      query_warps[0] = rsqrtf(query_sum / columns + epsilon);
      key_warps[0] = rsqrtf(key_sum / columns + epsilon);
    }
  }
  __syncthreads();
  const float query_scale = 1.0f / static_cast<float>(columns);
  const float key_scale = rsqrtf(static_cast<float>(columns));
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const __nv_bfloat16 query_unit = __float2bfloat16_rn(
        __bfloat162float(query[index]) * query_warps[0]);
    const __nv_bfloat16 key_unit = __float2bfloat16_rn(
        __bfloat162float(key[index]) * key_warps[0]);
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
    float weight_shift) {
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
    sum = lane < 8 ? warps[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) warps[0] = rsqrtf(sum / columns + epsilon);
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int index = row * columns + column;
    const float scale = __bfloat162float(weight[column]) + weight_shift;
    const __nv_bfloat16 normalized = __float2bfloat16_rn(
        __bfloat162float(input[index]) * warps[0] * scale);
    const float gate_value = __bfloat162float(gate[index]);
    const float activated = gate_value / (1.0f + expf(-gate_value));
    output[index] = __float2bfloat16_rn(
        activated * __bfloat162float(normalized));
  }
}
