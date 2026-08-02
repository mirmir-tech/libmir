#include <cuda_bf16.h>

__device__ float libmir_qkv_inverse_rms(
    const __nv_bfloat16* row, unsigned int columns, float epsilon) {
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float value = __bfloat162float(row[column]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warps[8];
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x / 32u] = sum;
  __syncthreads();
  if (threadIdx.x < 32u) {
    sum = threadIdx.x < blockDim.x / 32u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0u) warps[0] = rsqrtf(sum / columns + epsilon);
  }
  __syncthreads();
  return warps[0];
}

__device__ float libmir_qkv_normalized(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    unsigned int column, float inverse_rms, unsigned int normalize) {
  const float value = __bfloat162float(input[column]);
  if (!normalize) return value;
  const float scale = __bfloat162float(weight[column]);
  return __bfloat162float(__float2bfloat16_rn(value * inverse_rms * scale));
}

__device__ void libmir_qkv_transform(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned int head_dim, unsigned int rotary_dim,
    unsigned int pairing_dim, unsigned int position, float theta,
    float inverse_rms, unsigned int normalize) {
  const unsigned int half = pairing_dim / 2u;
  const unsigned int rotary_pairs = rotary_dim / 2u;
  for (unsigned int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const float first =
        libmir_qkv_normalized(input, weight, pair, inverse_rms, normalize);
    const float second =
        libmir_qkv_normalized(input, weight, pair + half, inverse_rms, normalize);
    if (pair >= rotary_pairs) {
      output[pair] = __float2bfloat16_rn(first);
      output[pair + half] = __float2bfloat16_rn(second);
      continue;
    }
    const float frequency = powf(theta, -2.0f * pair / pairing_dim);
    float sine;
    float cosine;
    sincosf(position * frequency, &sine, &cosine);
    output[pair] = __float2bfloat16_rn(
        fmaf(-second, sine, first * cosine));
    output[pair + half] = __float2bfloat16_rn(
        fmaf(first, sine, second * cosine));
  }
  for (unsigned int dimension = pairing_dim + threadIdx.x;
       dimension < head_dim; dimension += blockDim.x) {
    output[dimension] = __float2bfloat16_rn(
        libmir_qkv_normalized(
            input, weight, dimension, inverse_rms, normalize));
  }
}

extern "C" __global__ void libmir_cuda_qkv_postprocess_bf16(
    const __nv_bfloat16* query_input, const __nv_bfloat16* key_input,
    const __nv_bfloat16* value_input, const __nv_bfloat16* query_weight,
    const __nv_bfloat16* key_weight, __nv_bfloat16* query_output,
    __nv_bfloat16* key_output, __nv_bfloat16* value_output,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int value_head_dim,
    unsigned int rotary_dim, unsigned int pairing_dim,
    unsigned int start_position, float theta, float epsilon,
    unsigned int separate_inputs, unsigned int normalize_query, unsigned int normalize_key,
    unsigned int normalize_value) {
  const unsigned int heads_per_token = query_heads + 2u * kv_heads;
  const unsigned int logical_row = blockIdx.x;
  if (logical_row >= tokens * heads_per_token) return;
  const unsigned int token = logical_row / heads_per_token;
  const unsigned int local_head = logical_row % heads_per_token;
  const unsigned int query_width = query_heads * head_dim;
  const unsigned int key_width = kv_heads * head_dim;
  const unsigned int value_width = kv_heads * value_head_dim;
  const unsigned int packed_width = query_width + key_width + value_width;

  if (local_head >= query_heads + kv_heads) {
    const unsigned int head = local_head - query_heads - kv_heads;
    const __nv_bfloat16* input = value_input
        + token * (separate_inputs ? value_width : packed_width)
        + (separate_inputs ? 0u : query_width + key_width)
        + head * value_head_dim;
    __nv_bfloat16* output = value_output
        + (token * kv_heads + head) * value_head_dim;
    const float inverse_rms = normalize_value
        ? libmir_qkv_inverse_rms(input, value_head_dim, epsilon) : 1.0f;
    for (unsigned int column = threadIdx.x; column < value_head_dim;
         column += blockDim.x) {
      const float value = __bfloat162float(input[column]);
      output[column] = __float2bfloat16_rn(value * inverse_rms);
    }
    return;
  }

  const bool query = local_head < query_heads;
  const unsigned int head = query ? local_head : local_head - query_heads;
  const __nv_bfloat16* base = query ? query_input : key_input;
  const __nv_bfloat16* input = base
      + token * (separate_inputs ? (query ? query_width : key_width) : packed_width)
      + (separate_inputs || query ? 0u : query_width) + head * head_dim;
  const __nv_bfloat16* weight = query ? query_weight : key_weight;
  const unsigned int normalize = query ? normalize_query : normalize_key;
  __nv_bfloat16* output = (query ? query_output : key_output)
      + (token * (query ? query_heads : kv_heads) + head) * head_dim;
  const float inverse_rms = normalize
      ? libmir_qkv_inverse_rms(input, head_dim, epsilon) : 1.0f;
  libmir_qkv_transform(
      input, weight, output, head_dim, rotary_dim, pairing_dim,
      start_position + token, theta, inverse_rms, normalize);
}

extern "C" __global__ void libmir_cuda_qkv_postprocess_batch_bf16(
    const __nv_bfloat16* query_input, const __nv_bfloat16* key_input,
    const __nv_bfloat16* value_input, const __nv_bfloat16* query_weight,
    const __nv_bfloat16* key_weight, const unsigned int* positions,
    __nv_bfloat16* query_output, __nv_bfloat16* key_output,
    __nv_bfloat16* value_output, unsigned int tokens,
    unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int value_head_dim,
    unsigned int rotary_dim, unsigned int pairing_dim,
    float theta, float epsilon, unsigned int separate_inputs,
    unsigned int normalize_query,
    unsigned int normalize_key, unsigned int normalize_value) {
  const unsigned int heads_per_token = query_heads + 2u * kv_heads;
  const unsigned int logical_row = blockIdx.x;
  if (logical_row >= tokens * heads_per_token) return;
  const unsigned int token = logical_row / heads_per_token;
  const unsigned int local_head = logical_row % heads_per_token;
  const unsigned int query_width = query_heads * head_dim;
  const unsigned int key_width = kv_heads * head_dim;
  const unsigned int value_width = kv_heads * value_head_dim;
  const unsigned int packed_width = query_width + key_width + value_width;

  if (local_head >= query_heads + kv_heads) {
    const unsigned int head = local_head - query_heads - kv_heads;
    const __nv_bfloat16* input = value_input
        + token * (separate_inputs ? value_width : packed_width)
        + (separate_inputs ? 0u : query_width + key_width)
        + head * value_head_dim;
    __nv_bfloat16* output = value_output
        + (token * kv_heads + head) * value_head_dim;
    const float inverse_rms = normalize_value
        ? libmir_qkv_inverse_rms(input, value_head_dim, epsilon) : 1.0f;
    for (unsigned int column = threadIdx.x; column < value_head_dim;
         column += blockDim.x) {
      const float value = __bfloat162float(input[column]);
      output[column] = __float2bfloat16_rn(value * inverse_rms);
    }
    return;
  }

  const bool query = local_head < query_heads;
  const unsigned int head = query ? local_head : local_head - query_heads;
  const __nv_bfloat16* base = query ? query_input : key_input;
  const __nv_bfloat16* input = base
      + token * (separate_inputs ? (query ? query_width : key_width) : packed_width)
      + (separate_inputs || query ? 0u : query_width) + head * head_dim;
  const __nv_bfloat16* weight = query ? query_weight : key_weight;
  const unsigned int normalize = query ? normalize_query : normalize_key;
  __nv_bfloat16* output = (query ? query_output : key_output)
      + (token * (query ? query_heads : kv_heads) + head) * head_dim;
  const float inverse_rms = normalize
      ? libmir_qkv_inverse_rms(input, head_dim, epsilon) : 1.0f;
  const unsigned int position = positions[token];
  libmir_qkv_transform(
      input, weight, output, head_dim, rotary_dim, pairing_dim, position,
      theta, inverse_rms, normalize);
}
