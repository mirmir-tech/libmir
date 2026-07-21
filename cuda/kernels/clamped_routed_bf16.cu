#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __constant__ float clamped_routed_mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ float clamped_routed_mxfp4(unsigned char packed, unsigned int high) {
  const unsigned int code = high ? packed >> 4 : packed & 15u;
  return clamped_routed_mxfp4_values[code];
}

__device__ float clamped_routed_page(const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

extern "C" __global__ void libmir_cuda_clamped_routed_qkv_bf16(
    const __nv_bfloat16* packed, const __nv_bfloat16* q_bias,
    const __nv_bfloat16* k_bias, const __nv_bfloat16* v_bias,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int start_position, float theta,
    float factor, float initial_context, float beta_fast, float beta_slow) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q_width = query_heads * head_dim;
  const unsigned int kv_width = kv_heads * head_dim;
  const unsigned int row_width = q_width + 2u * kv_width;
  const unsigned int total = tokens * row_width;
  if (index >= total) return;
  const unsigned int token = index / row_width;
  const unsigned int column = index % row_width;
  if (column >= q_width + kv_width) {
    const unsigned int local = column - q_width - kv_width;
    value[token * kv_width + local] = __float2bfloat16_rn(
        __bfloat162float(packed[index]) + __bfloat162float(v_bias[local]));
    return;
  }
  const bool is_query = column < q_width;
  const unsigned int local = is_query ? column : column - q_width;
  const unsigned int dimension = local % head_dim;
  const unsigned int half = head_dim / 2u;
  const unsigned int pair_dim = dimension % half;
  const unsigned int peer = dimension < half ? column + half : column - half;
  const __nv_bfloat16* bias = is_query ? q_bias : k_bias;
  const float current = __bfloat162float(packed[index]) + __bfloat162float(bias[local]);
  const unsigned int peer_local = dimension < half ? local + half : local - half;
  const float paired = __bfloat162float(packed[token * row_width + peer]) +
      __bfloat162float(bias[peer_local]);
  const float frequency = powf(theta, 2.0f * float(pair_dim) / float(head_dim));
  const float low = float(half) * logf(initial_context / (beta_fast * 6.28318530718f)) /
      logf(theta);
  const float high = float(half) * logf(initial_context / (beta_slow * 6.28318530718f)) /
      logf(theta);
  const float ramp = fminf(1.0f, fmaxf(0.0f, (float(pair_dim) - low) / (high - low)));
  const float inv = (1.0f - ramp) / frequency + ramp / (factor * frequency);
  const float angle = float(start_position + token) * inv;
  const float concentration = 0.1f * logf(factor) + 1.0f;
  const float rotated = dimension < half
      ? (current * cosf(angle) - paired * sinf(angle)) * concentration
      : (current * cosf(angle) + paired * sinf(angle)) * concentration;
  if (is_query) query[token * q_width + local] = __float2bfloat16_rn(rotated);
  else key[token * kv_width + local] = __float2bfloat16_rn(rotated);
}

extern "C" __global__ void libmir_cuda_clamped_routed_qkv_split_bf16(
    const __nv_bfloat16* q_input, const __nv_bfloat16* k_input,
    const __nv_bfloat16* v_input, const __nv_bfloat16* q_bias,
    const __nv_bfloat16* k_bias, const __nv_bfloat16* v_bias,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int start_position, float theta,
    float factor, float initial_context, float beta_fast, float beta_slow) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q_width = query_heads * head_dim;
  const unsigned int kv_width = kv_heads * head_dim;
  const unsigned int row_width = q_width + 2u * kv_width;
  if (index >= tokens * row_width) return;
  const unsigned int token = index / row_width;
  const unsigned int column = index % row_width;
  if (column >= q_width + kv_width) {
    const unsigned int local = column - q_width - kv_width;
    value[token * kv_width + local] = __float2bfloat16_rn(
        __bfloat162float(v_input[token * kv_width + local]) +
        __bfloat162float(v_bias[local]));
    return;
  }
  const bool is_query = column < q_width;
  const unsigned int local = is_query ? column : column - q_width;
  const unsigned int width = is_query ? q_width : kv_width;
  const __nv_bfloat16* source = is_query ? q_input : k_input;
  const __nv_bfloat16* bias = is_query ? q_bias : k_bias;
  const unsigned int dimension = local % head_dim;
  const unsigned int half = head_dim / 2u;
  const unsigned int pair_dim = dimension % half;
  const unsigned int peer_local = dimension < half ? local + half : local - half;
  const float current = __bfloat162float(source[token * width + local]) +
      __bfloat162float(bias[local]);
  const float paired = __bfloat162float(source[token * width + peer_local]) +
      __bfloat162float(bias[peer_local]);
  const float frequency = powf(theta, 2.0f * float(pair_dim) / float(head_dim));
  const float low = float(half) * logf(initial_context / (beta_fast * 6.28318530718f)) /
      logf(theta);
  const float high = float(half) * logf(initial_context / (beta_slow * 6.28318530718f)) /
      logf(theta);
  const float ramp = fminf(1.0f, fmaxf(0.0f, (float(pair_dim) - low) / (high - low)));
  const float inv = (1.0f - ramp) / frequency + ramp / (factor * frequency);
  const float angle = float(start_position + token) * inv;
  const float concentration = 0.1f * logf(factor) + 1.0f;
  const float rotated = dimension < half
      ? (current * cosf(angle) - paired * sinf(angle)) * concentration
      : (current * cosf(angle) + paired * sinf(angle)) * concentration;
  if (is_query) query[token * q_width + local] = __float2bfloat16_rn(rotated);
  else key[token * kv_width + local] = __float2bfloat16_rn(rotated);
}

extern "C" __global__ void libmir_cuda_clamped_routed_add_bias_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, unsigned int rows, unsigned int columns) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * columns) output[index] = __float2bfloat16_rn(
      __bfloat162float(input[index]) + __bfloat162float(bias[index % columns]));
}

extern "C" __global__ void libmir_cuda_clamped_routed_mxfp4_gate_up_bf16(
    const __nv_bfloat16* input, const unsigned char* blocks,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, __nv_bfloat16* output,
    unsigned int tokens, unsigned int top_k, unsigned int hidden,
    unsigned int intermediate, float limit) {
  const unsigned int unit = blockIdx.x % intermediate;
  const unsigned int assignment = blockIdx.x / intermediate;
  if (unit >= intermediate || assignment >= tokens * top_k) return;
  const unsigned int expert = selected[assignment];
  const unsigned int groups = hidden / 32u;
  const unsigned int gate_row = 2u * unit;
  const unsigned int up_row = gate_row + 1u;
  float gate = threadIdx.x == 0u
      ? __bfloat162float(bias[expert * intermediate * 2u + gate_row]) : 0.0f;
  float up = threadIdx.x == 0u
      ? __bfloat162float(bias[expert * intermediate * 2u + up_row]) : 0.0f;
  for (unsigned int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const unsigned int group = column / 32u;
    const unsigned int nibble = column % 32u;
    const unsigned int gate_scale = (expert * intermediate * 2u + gate_row) * groups + group;
    const unsigned int up_scale = gate_scale + groups;
    const float x = __bfloat162float(input[(assignment / top_k) * hidden + column]);
    gate += x * ldexpf(clamped_routed_mxfp4(blocks[gate_scale * 16u + nibble / 2u], nibble & 1u),
                       int(scales[gate_scale]) - 127);
    up += x * ldexpf(clamped_routed_mxfp4(blocks[up_scale * 16u + nibble / 2u], nibble & 1u),
                     int(scales[up_scale]) - 127);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset);
    up += __shfl_down_sync(0xffffffffu, up, offset);
  }
  if (threadIdx.x == 0u) {
    gate = fminf(gate, limit);
    up = fminf(limit, fmaxf(-limit, up));
    output[assignment * intermediate + unit] = __float2bfloat16_rn(
        gate / (1.0f + expf(-1.702f * gate)) * (up + 1.0f));
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_mxfp4_down_bf16(
    const __nv_bfloat16* input, const unsigned char* blocks,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, const __nv_bfloat16* routing,
    __nv_bfloat16* output, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int intermediate) {
  const unsigned int column = blockIdx.x % hidden;
  const unsigned int token = blockIdx.x / hidden;
  if (column >= hidden || token >= tokens) return;
  const unsigned int groups = intermediate / 32u;
  float total = 0.0f;
  for (unsigned int route = 0u; route < top_k; ++route) {
    const unsigned int assignment = token * top_k + route;
    const unsigned int expert = selected[assignment];
    float sum = threadIdx.x == 0u ? __bfloat162float(bias[expert * hidden + column]) : 0.0f;
    for (unsigned int feature = threadIdx.x; feature < intermediate; feature += blockDim.x) {
      const unsigned int group = feature / 32u;
      const unsigned int nibble = feature % 32u;
      const unsigned int scale_index = (expert * hidden + column) * groups + group;
      const float weight = ldexpf(
          clamped_routed_mxfp4(blocks[scale_index * 16u + nibble / 2u], nibble & 1u),
          int(scales[scale_index]) - 127);
      sum += __bfloat162float(input[assignment * intermediate + feature]) * weight;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (threadIdx.x == 0u) total += sum * __bfloat162float(routing[assignment]);
  }
  if (threadIdx.x == 0u) output[token * hidden + column] = __float2bfloat16_rn(total);
}

extern "C" __global__ void libmir_cuda_clamped_routed_mlx_mxfp4_gate_up_bf16(
    const __nv_bfloat16* input, const unsigned int* gate_blocks,
    const unsigned char* gate_scales, const __nv_bfloat16* gate_bias,
    const unsigned int* up_blocks, const unsigned char* up_scales,
    const __nv_bfloat16* up_bias, const unsigned int* selected,
    __nv_bfloat16* output, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int intermediate, float limit) {
  const unsigned int unit = blockIdx.x % intermediate;
  const unsigned int assignment = blockIdx.x / intermediate;
  if (unit >= intermediate || assignment >= tokens * top_k) return;
  const unsigned int expert = selected[assignment];
  const unsigned int words = hidden / 8u;
  const unsigned int groups = hidden / 32u;
  const unsigned int row = expert * intermediate + unit;
  float gate = threadIdx.x == 0u ? __bfloat162float(gate_bias[row]) : 0.0f;
  float up = threadIdx.x == 0u ? __bfloat162float(up_bias[row]) : 0.0f;
  for (unsigned int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const unsigned int word = column / 8u;
    const unsigned int shift = (column % 8u) * 4u;
    const unsigned int group = column / 32u;
    const float x = __bfloat162float(input[(assignment / top_k) * hidden + column]);
    const float gate_weight = ldexpf(
        clamped_routed_mxfp4((gate_blocks[row * words + word] >> shift) & 15u, 0u),
        int(gate_scales[row * groups + group]) - 127);
    const float up_weight = ldexpf(
        clamped_routed_mxfp4((up_blocks[row * words + word] >> shift) & 15u, 0u),
        int(up_scales[row * groups + group]) - 127);
    gate += x * gate_weight;
    up += x * up_weight;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset);
    up += __shfl_down_sync(0xffffffffu, up, offset);
  }
  if (threadIdx.x == 0u) {
    gate = fminf(gate, limit);
    up = fminf(limit, fmaxf(-limit, up));
    output[assignment * intermediate + unit] = __float2bfloat16_rn(
        gate / (1.0f + expf(-1.702f * gate)) * (up + 1.0f));
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_mlx_mxfp4_down_bf16(
    const __nv_bfloat16* input, const unsigned int* blocks,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, const __nv_bfloat16* routing,
    __nv_bfloat16* output, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int intermediate) {
  const unsigned int column = blockIdx.x % hidden;
  const unsigned int token = blockIdx.x / hidden;
  if (column >= hidden || token >= tokens) return;
  const unsigned int words = intermediate / 8u;
  const unsigned int groups = intermediate / 32u;
  float total = 0.0f;
  for (unsigned int route = 0u; route < top_k; ++route) {
    const unsigned int assignment = token * top_k + route;
    const unsigned int expert = selected[assignment];
    const unsigned int row = expert * hidden + column;
    float sum = threadIdx.x == 0u ? __bfloat162float(bias[row]) : 0.0f;
    for (unsigned int feature = threadIdx.x; feature < intermediate; feature += blockDim.x) {
      const unsigned int word = feature / 8u;
      const unsigned int shift = (feature % 8u) * 4u;
      const unsigned int group = feature / 32u;
      const float weight = ldexpf(
          clamped_routed_mxfp4((blocks[row * words + word] >> shift) & 15u, 0u),
          int(scales[row * groups + group]) - 127);
      sum += __bfloat162float(input[assignment * intermediate + feature]) * weight;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (threadIdx.x == 0u) total += sum * __bfloat162float(routing[assignment]);
  }
  if (threadIdx.x == 0u) output[token * hidden + column] = __float2bfloat16_rn(total);
}
