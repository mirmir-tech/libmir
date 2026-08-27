#include <cuda_bf16.h>

template <unsigned int key_slots, unsigned int value_slots>
__device__ __forceinline__ void gated_delta_value_tiled_recurrence(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ decay,
    const float* __restrict__ update,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  const unsigned int lane = threadIdx.x;
  const unsigned int value_base =
      (blockIdx.y * blockDim.y + threadIdx.y) * value_slots;
  const unsigned int value_head = blockIdx.z;
  if (value_base >= value_dim || value_head >= value_heads) return;
  const unsigned int key_head = value_head / (value_heads / key_heads);
  float memory[value_slots][key_slots];

#pragma unroll
  for (unsigned int row = 0; row < value_slots; ++row) {
    const unsigned int value_index = value_base + row;
#pragma unroll
    for (unsigned int slot = 0; slot < key_slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      memory[row][slot] = value_index < value_dim
          ? state[(value_head * value_dim + value_index) * key_dim + dimension]
          : 0.0f;
    }
  }

  for (unsigned int time = 0; time < tokens; ++time) {
    const unsigned int gate_index = time * value_heads + value_head;
    const unsigned int key_base = (time * key_heads + key_head) * key_dim;
    float keys[key_slots];
    float queries[key_slots];
#pragma unroll
    for (unsigned int slot = 0; slot < key_slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      keys[slot] = __bfloat162float(key[key_base + dimension]);
      queries[slot] = __bfloat162float(query[key_base + dimension]);
    }

    float projections[value_slots] = {};
#pragma unroll
    for (unsigned int row = 0; row < value_slots; ++row) {
#pragma unroll
      for (unsigned int slot = 0; slot < key_slots; ++slot) {
        memory[row][slot] *= decay[gate_index];
        projections[row] += memory[row][slot] * keys[slot];
      }
    }
#pragma unroll
    for (unsigned int row = 0; row < value_slots; ++row) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        projections[row] += __shfl_down_sync(0xffffffffu, projections[row], offset);
      }
    }

    float deltas[value_slots];
#pragma unroll
    for (unsigned int row = 0; row < value_slots; ++row) {
      const unsigned int value_index = value_base + row;
      const float target = value_index < value_dim
          ? __bfloat162float(
                value[(time * value_heads + value_head) * value_dim + value_index])
          : 0.0f;
      deltas[row] = __shfl_sync(
          0xffffffffu, (target - projections[row]) * update[gate_index], 0);
    }

    float results[value_slots] = {};
#pragma unroll
    for (unsigned int row = 0; row < value_slots; ++row) {
#pragma unroll
      for (unsigned int slot = 0; slot < key_slots; ++slot) {
        memory[row][slot] += keys[slot] * deltas[row];
        results[row] += memory[row][slot] * queries[slot];
      }
    }
#pragma unroll
    for (unsigned int row = 0; row < value_slots; ++row) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        results[row] += __shfl_down_sync(0xffffffffu, results[row], offset);
      }
      const unsigned int value_index = value_base + row;
      if (lane == 0 && value_index < value_dim) {
        output[(time * value_heads + value_head) * value_dim + value_index]
            = __float2bfloat16_rn(results[row]);
      }
    }
  }

#pragma unroll
  for (unsigned int row = 0; row < value_slots; ++row) {
    const unsigned int value_index = value_base + row;
    if (value_index >= value_dim) continue;
#pragma unroll
    for (unsigned int slot = 0; slot < key_slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      state[(value_head * value_dim + value_index) * key_dim + dimension]
          = memory[row][slot];
    }
  }
}

#define GATED_DELTA_VALUE_TILED_KERNEL(name, value_slots)                         \
extern "C" __global__ void name(                                                 \
    const __nv_bfloat16* query, const __nv_bfloat16* key,                        \
    const __nv_bfloat16* value, const float* decay, const float* update,          \
    float* state, __nv_bfloat16* output,                                          \
    unsigned int tokens, unsigned int key_heads, unsigned int value_heads,        \
    unsigned int key_dim, unsigned int value_dim) {                               \
  switch (key_dim) {                                                              \
    case 32: return gated_delta_value_tiled_recurrence<1, value_slots>(           \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 64: return gated_delta_value_tiled_recurrence<2, value_slots>(           \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 96: return gated_delta_value_tiled_recurrence<3, value_slots>(           \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 128: return gated_delta_value_tiled_recurrence<4, value_slots>(          \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 160: return gated_delta_value_tiled_recurrence<5, value_slots>(          \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 192: return gated_delta_value_tiled_recurrence<6, value_slots>(          \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 224: return gated_delta_value_tiled_recurrence<7, value_slots>(          \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
    case 256: return gated_delta_value_tiled_recurrence<8, value_slots>(          \
        query, key, value, decay, update, state, output,                          \
        tokens, key_heads, value_heads, key_dim, value_dim);                      \
  }                                                                               \
}

GATED_DELTA_VALUE_TILED_KERNEL(
    libmir_cuda_gated_delta_recurrence_value_tiled_2_bf16, 2)
GATED_DELTA_VALUE_TILED_KERNEL(
    libmir_cuda_gated_delta_recurrence_value_tiled_4_bf16, 4)
GATED_DELTA_VALUE_TILED_KERNEL(
    libmir_cuda_gated_delta_recurrence_value_tiled_8_bf16, 8)
