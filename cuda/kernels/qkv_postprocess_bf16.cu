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

// The inverse RMS a query or key head scales by, and the norm weight of its
// columns: per head, or with LIBMIR_QK_NORM_ACROSS_HEADS over the whole
// projection of the token (`OLMo` 2), whose weight spans every head.
__device__ float libmir_qk_inverse_rms(
    const __nv_bfloat16* input, const __nv_bfloat16** weight, unsigned int head,
    unsigned int heads, unsigned int head_dim, float epsilon) {
#ifdef LIBMIR_QK_NORM_ACROSS_HEADS
  *weight += head * head_dim;
  return libmir_qkv_inverse_rms(input - head * head_dim, heads * head_dim, epsilon);
#else
  (void)head;
  (void)heads;
  return libmir_qkv_inverse_rms(input, head_dim, epsilon);
#endif
}

__device__ float libmir_qkv_normalized(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    unsigned int column, float inverse_rms, unsigned int normalize) {
  const float value = __bfloat162float(input[column]);
  if (!normalize) return value;
  const float scale = __bfloat162float(weight[column]);
  return __bfloat162float(__float2bfloat16_rn(value * inverse_rms * scale));
}

#ifdef LIBMIR_ROPE_PAIR_FACTORS
__constant__ float libmir_rope_pair_factors[] = {LIBMIR_ROPE_PAIR_FACTORS};
#endif

// Each pair's rotary frequency with long-context scaling: `smooth` blends
// from the full `factor` (0) to none (1) linearly in the frequency, which
// covers no scaling (factor 1), linear scaling (slope and offset 0) and
// Llama 3's piecewise scaling. With LIBMIR_ROPE_PAIR_FACTORS (Phi-3
// LongRoPE) each pair's period is further multiplied by its factor.
__device__ __forceinline__ float libmir_rope_frequency(
    unsigned int pair, unsigned int pairing_dim, float theta, float factor, float slope,
    float offset) {
  const float frequency = powf(theta, -2.0f * pair / pairing_dim);
  const float smooth = fminf(fmaxf(fmaf(slope, frequency, -offset), 0.0f), 1.0f);
  const float scaled = frequency * ((1.0f - smooth) / factor + smooth);
#ifdef LIBMIR_ROPE_PAIR_FACTORS
  return scaled / libmir_rope_pair_factors[pair];
#else
  return scaled;
#endif
}

// Llama 4 attention temperature: the query at `position` scales by
// `1 + beta · ln(1 + floor((position + offset) / window))`; keys and values
// and layers compiled without it keep a unit scale.
__device__ __forceinline__ float libmir_query_temperature(unsigned int position) {
#ifdef LIBMIR_QUERY_TEMPERATURE_WINDOW
  const unsigned int periods =
      (position + LIBMIR_QUERY_TEMPERATURE_OFFSET) / LIBMIR_QUERY_TEMPERATURE_WINDOW;
  return fmaf(LIBMIR_QUERY_TEMPERATURE_BETA, log1pf(float(periods)), 1.0f);
#else
  return 1.0f;
#endif
}

__device__ void libmir_qkv_transform(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, unsigned int head_dim, unsigned int rotary_dim,
    unsigned int pairing_dim, unsigned int position, float theta, float factor,
    float slope, float offset, float inverse_rms, unsigned int normalize,
    float output_scale) {
  const unsigned int half = pairing_dim / 2u;
  const unsigned int rotary_pairs = rotary_dim / 2u;
  for (unsigned int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const float first =
        libmir_qkv_normalized(input, weight, pair, inverse_rms, normalize) * output_scale;
    const float second =
        libmir_qkv_normalized(input, weight, pair + half, inverse_rms, normalize)
        * output_scale;
    if (pair >= rotary_pairs) {
      output[pair] = __float2bfloat16_rn(first);
      output[pair + half] = __float2bfloat16_rn(second);
      continue;
    }
    const float frequency =
        libmir_rope_frequency(pair, pairing_dim, theta, factor, slope, offset);
    float sine;
    float cosine;
    sincosf(position * frequency, &sine, &cosine);
#ifdef LIBMIR_ROPE_ATTENTION_FACTOR
    sine *= LIBMIR_ROPE_ATTENTION_FACTOR;
    cosine *= LIBMIR_ROPE_ATTENTION_FACTOR;
#endif
    output[pair] = __float2bfloat16_rn(
        fmaf(-second, sine, first * cosine));
    output[pair + half] = __float2bfloat16_rn(
        fmaf(first, sine, second * cosine));
  }
  for (unsigned int dimension = pairing_dim + threadIdx.x;
       dimension < head_dim; dimension += blockDim.x) {
    output[dimension] = __float2bfloat16_rn(
        libmir_qkv_normalized(
            input, weight, dimension, inverse_rms, normalize) * output_scale);
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
    unsigned int start_position, float theta, float rope_factor, float rope_slope,
    float rope_offset, float epsilon,
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
      ? libmir_qk_inverse_rms(
            input, &weight, head, query ? query_heads : kv_heads, head_dim, epsilon)
      : 1.0f;
  const unsigned int position = start_position + token;
  libmir_qkv_transform(
      input, weight, output, head_dim, rotary_dim, pairing_dim,
      position, theta, rope_factor, rope_slope, rope_offset, inverse_rms,
      normalize, query ? libmir_query_temperature(position) : 1.0f);
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
    float theta, float rope_factor, float rope_slope, float rope_offset, float epsilon,
    unsigned int separate_inputs, unsigned int normalize_query,
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
      ? libmir_qk_inverse_rms(
            input, &weight, head, query ? query_heads : kv_heads, head_dim, epsilon)
      : 1.0f;
  const unsigned int position = positions[token];
  libmir_qkv_transform(
      input, weight, output, head_dim, rotary_dim, pairing_dim, position,
      theta, rope_factor, rope_slope, rope_offset, inverse_rms, normalize,
      query ? libmir_query_temperature(position) : 1.0f);
}
