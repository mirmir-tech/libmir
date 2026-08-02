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
    unsigned int head_dim, const float* rope_sines,
    const float* rope_cosines, float concentration) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q_width = query_heads * head_dim;
  const unsigned int kv_width = kv_heads * head_dim;
  const unsigned int row_width = q_width + 2u * kv_width;
  const unsigned int half = head_dim / 2u;
  const unsigned int rotary_heads = query_heads + kv_heads;
  const unsigned int rotary_values = tokens * rotary_heads * half;
  const unsigned int total = rotary_values + tokens * kv_width;
  if (index >= total) return;
  if (index >= rotary_values) {
    const unsigned int value_index = index - rotary_values;
    const unsigned int token = value_index / kv_width;
    const unsigned int local = value_index % kv_width;
    const unsigned int packed_index =
        token * row_width + q_width + kv_width + local;
    value[token * kv_width + local] = __float2bfloat16_rn(
        __bfloat162float(packed[packed_index]) +
        __bfloat162float(v_bias[local]));
    return;
  }

  const unsigned int rotary_row = index / half;
  const unsigned int pair_dim = index % half;
  const unsigned int token = rotary_row / rotary_heads;
  const unsigned int local_head = rotary_row % rotary_heads;
  const bool is_query = local_head < query_heads;
  const unsigned int head =
      is_query ? local_head : local_head - query_heads;
  const unsigned int local = head * head_dim + pair_dim;
  const unsigned int column = (is_query ? 0u : q_width) + local;
  const __nv_bfloat16* bias = is_query ? q_bias : k_bias;
  const float first = __bfloat162float(packed[token * row_width + column]) +
      __bfloat162float(bias[local]);
  const float second =
      __bfloat162float(packed[token * row_width + column + half]) +
      __bfloat162float(bias[local + half]);
  const float sine = rope_sines[token * half + pair_dim];
  const float cosine = rope_cosines[token * half + pair_dim];
  __nv_bfloat16* output = is_query ? query + token * q_width
                                  : key + token * kv_width;
  output[local] = __float2bfloat16_rn(
      (first * cosine - second * sine) * concentration);
  output[local + half] = __float2bfloat16_rn(
      (second * cosine + first * sine) * concentration);
}

extern "C" __global__ void libmir_cuda_clamped_routed_qkv_split_bf16(
    const __nv_bfloat16* q_input, const __nv_bfloat16* k_input,
    const __nv_bfloat16* v_input, const __nv_bfloat16* q_bias,
    const __nv_bfloat16* k_bias, const __nv_bfloat16* v_bias,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    unsigned int tokens, unsigned int query_heads, unsigned int kv_heads,
    unsigned int head_dim, const float* rope_sines,
    const float* rope_cosines, float concentration) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int q_width = query_heads * head_dim;
  const unsigned int kv_width = kv_heads * head_dim;
  const unsigned int half = head_dim / 2u;
  const unsigned int rotary_heads = query_heads + kv_heads;
  const unsigned int rotary_values = tokens * rotary_heads * half;
  const unsigned int total = rotary_values + tokens * kv_width;
  if (index >= total) return;
  if (index >= rotary_values) {
    const unsigned int value_index = index - rotary_values;
    const unsigned int token = value_index / kv_width;
    const unsigned int local = value_index % kv_width;
    value[token * kv_width + local] = __float2bfloat16_rn(
        __bfloat162float(v_input[token * kv_width + local]) +
        __bfloat162float(v_bias[local]));
    return;
  }

  const unsigned int rotary_row = index / half;
  const unsigned int pair_dim = index % half;
  const unsigned int token = rotary_row / rotary_heads;
  const unsigned int local_head = rotary_row % rotary_heads;
  const bool is_query = local_head < query_heads;
  const unsigned int head =
      is_query ? local_head : local_head - query_heads;
  const unsigned int local = head * head_dim + pair_dim;
  const unsigned int width = is_query ? q_width : kv_width;
  const __nv_bfloat16* source = is_query ? q_input : k_input;
  const __nv_bfloat16* bias = is_query ? q_bias : k_bias;
  const float first = __bfloat162float(source[token * width + local]) +
      __bfloat162float(bias[local]);
  const float second = __bfloat162float(source[token * width + local + half]) +
      __bfloat162float(bias[local + half]);
  const float sine = rope_sines[token * half + pair_dim];
  const float cosine = rope_cosines[token * half + pair_dim];
  __nv_bfloat16* output = is_query ? query + token * q_width
                                  : key + token * kv_width;
  output[local] = __float2bfloat16_rn(
      (first * cosine - second * sine) * concentration);
  output[local + half] = __float2bfloat16_rn(
      (second * cosine + first * sine) * concentration);
}

extern "C" __global__ void libmir_cuda_clamped_routed_rope_angles(
    const unsigned int* positions, const float* inverse_frequencies,
    float* sines, float* cosines, unsigned int tokens,
    unsigned int half_head_dim) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * half_head_dim) return;
  const unsigned int token = index / half_head_dim;
  const unsigned int pair_dim = index % half_head_dim;
  const float angle =
      float(positions[token]) * inverse_frequencies[pair_dim];
  sincosf(angle, &sines[index], &cosines[index]);
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

extern "C" __global__ void libmir_cuda_clamped_routed_mxfp4_down_routes_bf16(
    const __nv_bfloat16* input, const unsigned char* blocks,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, const __nv_bfloat16* routing,
    float* partial, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int intermediate) {
  const unsigned int column = blockIdx.x % hidden;
  const unsigned int assignment = blockIdx.x / hidden;
  if (column >= hidden || assignment >= tokens * top_k) return;
  const unsigned int expert = selected[assignment];
  const unsigned int groups = intermediate / 32u;
  float sum = threadIdx.x == 0u
      ? __bfloat162float(bias[expert * hidden + column]) : 0.0f;
  for (unsigned int feature = threadIdx.x; feature < intermediate;
       feature += blockDim.x) {
    const unsigned int group = feature / 32u;
    const unsigned int nibble = feature % 32u;
    const unsigned int scale_index =
        (expert * hidden + column) * groups + group;
    const float weight = ldexpf(
        clamped_routed_mxfp4(
            blocks[scale_index * 16u + nibble / 2u], nibble & 1u),
        int(scales[scale_index]) - 127);
    sum += __bfloat162float(input[assignment * intermediate + feature]) *
           weight;
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  if (threadIdx.x == 0u) {
    partial[assignment * hidden + column] =
        sum * __bfloat162float(routing[assignment]);
  }
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

extern "C" __global__ void libmir_cuda_clamped_routed_mlx_mxfp4_down_routes_bf16(
    const __nv_bfloat16* input, const unsigned int* blocks,
    const unsigned char* scales, const __nv_bfloat16* bias,
    const unsigned int* selected, const __nv_bfloat16* routing,
    float* partial, unsigned int tokens, unsigned int top_k,
    unsigned int hidden, unsigned int intermediate) {
  const unsigned int column = blockIdx.x % hidden;
  const unsigned int assignment = blockIdx.x / hidden;
  if (column >= hidden || assignment >= tokens * top_k) return;
  const unsigned int expert = selected[assignment];
  const unsigned int row = expert * hidden + column;
  const unsigned int words = intermediate / 8u;
  const unsigned int groups = intermediate / 32u;
  float sum = threadIdx.x == 0u ? __bfloat162float(bias[row]) : 0.0f;
  for (unsigned int feature = threadIdx.x; feature < intermediate;
       feature += blockDim.x) {
    const unsigned int word = feature / 8u;
    const unsigned int shift = (feature % 8u) * 4u;
    const unsigned int group = feature / 32u;
    const float weight = ldexpf(
        clamped_routed_mxfp4(
            (blocks[row * words + word] >> shift) & 15u, 0u),
        int(scales[row * groups + group]) - 127);
    sum += __bfloat162float(input[assignment * intermediate + feature]) *
           weight;
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  if (threadIdx.x == 0u) {
    partial[assignment * hidden + column] =
        sum * __bfloat162float(routing[assignment]);
  }
}

extern "C" __global__ void libmir_cuda_clamped_routed_reduce_routes_bf16(
    const float* partial, __nv_bfloat16* output, unsigned int tokens,
    unsigned int top_k, unsigned int hidden) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  const unsigned int token = index / hidden;
  const unsigned int column = index - token * hidden;
  float total = 0.0f;
  for (unsigned int route = 0u; route < top_k; ++route) {
    total += partial[(token * top_k + route) * hidden + column];
  }
  output[index] = __float2bfloat16_rn(total);
}
