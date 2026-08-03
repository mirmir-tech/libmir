#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_gated_delta_convolution_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* history, __nv_bfloat16* output,
    unsigned int tokens, unsigned int channels, unsigned int kernel_size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * channels) return;
  const unsigned int token = index / channels;
  const unsigned int channel = index % channels;
  float sum = 0.0f;
  for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
    const unsigned int position = token + kernel;
    const float source = position < kernel_size - 1
        ? __bfloat162float(history[position * channels + channel])
        : __bfloat162float(input[(position - kernel_size + 1) * channels + channel]);
    sum += source * __bfloat162float(weight[channel * kernel_size + kernel]);
  }
  output[index] = __float2bfloat16_rn(sum / (1.0f + expf(-sum)));
}

extern "C" __global__ void libmir_cuda_gated_delta_history_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* history,
    __nv_bfloat16* next_history, unsigned int tokens,
    unsigned int channels, unsigned int kernel_size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int history_tokens = kernel_size - 1;
  if (index >= history_tokens * channels) return;
  const unsigned int history_token = index / channels;
  const unsigned int channel = index % channels;
  const unsigned int combined = tokens + history_token;
  next_history[index] = combined < history_tokens
      ? history[combined * channels + channel]
      : input[(combined - history_tokens) * channels + channel];
}

extern "C" __global__ void libmir_cuda_gated_delta_batch_convolution_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* history, __nv_bfloat16* next_history,
    __nv_bfloat16* output, unsigned int rows, unsigned int channels,
    unsigned int kernel_size, unsigned int tokens) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * tokens * channels) return;
  const unsigned int token = (index / channels) % tokens;
  const unsigned int row = index / (tokens * channels);
  const unsigned int channel = index % channels;
  const unsigned int history_tokens = kernel_size - 1u;
  const unsigned int history_base = row * history_tokens * channels;
  float sum = 0.0f;
  for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
    const unsigned int position = token + kernel;
    const float source = position < history_tokens
        ? __bfloat162float(history[history_base + position * channels + channel])
        : __bfloat162float(input[
            (row * tokens + position - history_tokens) * channels + channel]);
    sum += source * __bfloat162float(weight[channel * kernel_size + kernel]);
  }
  output[index] = __float2bfloat16_rn(sum / (1.0f + expf(-sum)));
  if (token == 0) {
    for (unsigned int history_token = 0; history_token < history_tokens; ++history_token) {
      const unsigned int combined = tokens + history_token;
      const unsigned int target = history_base + history_token * channels + channel;
      next_history[target] = combined < history_tokens
          ? history[history_base + combined * channels + channel]
          : input[(row * tokens + combined - history_tokens) * channels + channel];
    }
  }
}

extern "C" __global__ void libmir_cuda_gated_delta_parameters_bf16(
    const __nv_bfloat16* alpha, const __nv_bfloat16* beta,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* decay, float* update, unsigned int tokens,
    unsigned int value_heads) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * value_heads) return;
  const unsigned int head = index % value_heads;
  const float parameter = __bfloat162float(alpha[index])
      + __bfloat162float(dt_bias[head]);
  const float softplus = fmaxf(parameter, 0.0f)
      + log1pf(expf(-fabsf(parameter)));
  decay[index] = expf(-expf(__bfloat162float(a_log[head])) * softplus);
  update[index] = 1.0f / (1.0f + expf(-__bfloat162float(beta[index])));
}

template <unsigned int slots>
__device__ __forceinline__ void gated_delta_register_recurrence(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ decay,
    const float* __restrict__ update,
    float* __restrict__ state, __nv_bfloat16* __restrict__ output,
    unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  const unsigned int lane = threadIdx.x;
  const unsigned int value_index = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int value_head = blockIdx.z;
  if (value_index >= value_dim || value_head >= value_heads) return;
  const unsigned int key_head = value_head / (value_heads / key_heads);
  const unsigned int state_base = (value_head * value_dim + value_index) * key_dim;
  float memory[slots];
#pragma unroll
  for (unsigned int slot = 0; slot < slots; ++slot) {
    memory[slot] = state[state_base + lane + slot * 32];
  }
  for (unsigned int time = 0; time < tokens; ++time) {
    const unsigned int gate_index = time * value_heads + value_head;
    float step_decay;
    float step_update;
    if (tokens == 1) {
      const float parameter = __bfloat162float(alpha[gate_index])
          + __bfloat162float(dt_bias[value_head]);
      const float softplus = fmaxf(parameter, 0.0f)
          + log1pf(expf(-fabsf(parameter)));
      step_decay = expf(
          -expf(__bfloat162float(a_log[value_head])) * softplus);
      step_update = 1.0f /
          (1.0f + expf(-__bfloat162float(beta[gate_index])));
    } else {
      step_decay = decay[gate_index];
      step_update = update[gate_index];
    }
    const unsigned int key_base = (time * key_heads + key_head) * key_dim;
    float projection = 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      memory[slot] *= step_decay;
      projection += memory[slot] * __bfloat162float(key[key_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      projection += __shfl_down_sync(0xffffffffu, projection, offset);
    }
    const float target = __bfloat162float(
        value[(time * value_heads + value_head) * value_dim + value_index]);
    const float delta = __shfl_sync(
        0xffffffffu, (target - projection) * step_update, 0);
    float result = 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      memory[slot] += __bfloat162float(key[key_base + dimension]) * delta;
      const unsigned int query_base = (time * key_heads + key_head) * key_dim;
      result += memory[slot] * __bfloat162float(query[query_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane == 0) {
      output[(time * value_heads + value_head) * value_dim + value_index]
          = __float2bfloat16_rn(result);
    }
  }
#pragma unroll
  for (unsigned int slot = 0; slot < slots; ++slot) {
    state[state_base + lane + slot * 32] = memory[slot];
  }
}

extern "C" __global__ void libmir_cuda_gated_delta_recurrence_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* alpha,
    const __nv_bfloat16* beta, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, const float* decay, const float* update,
    float* state, __nv_bfloat16* output,
    unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  switch (key_dim) {
    case 32: return gated_delta_register_recurrence<1>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 64: return gated_delta_register_recurrence<2>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 96: return gated_delta_register_recurrence<3>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 128: return gated_delta_register_recurrence<4>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 160: return gated_delta_register_recurrence<5>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 192: return gated_delta_register_recurrence<6>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 224: return gated_delta_register_recurrence<7>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
    case 256: return gated_delta_register_recurrence<8>(
        query, key, value, alpha, beta, a_log, dt_bias, decay, update, state, output,
        tokens, key_heads, value_heads, key_dim, value_dim);
  }
}

template <unsigned int slots>
__device__ __forceinline__ void gated_delta_register_batch_recurrence(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ state, __nv_bfloat16* __restrict__ output,
    unsigned int rows, unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  const unsigned int row = blockIdx.x;
  const unsigned int lane = threadIdx.x;
  const unsigned int value_index = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int value_head = blockIdx.z;
  if (row >= rows || value_index >= value_dim || value_head >= value_heads) return;
  const unsigned int key_head = value_head / (value_heads / key_heads);
  const unsigned int row_state = value_heads * value_dim * key_dim;
  const unsigned int state_base = row * row_state
      + (value_head * value_dim + value_index) * key_dim;
  float memory[slots];
#pragma unroll
  for (unsigned int slot = 0; slot < slots; ++slot) {
    const unsigned int dimension = lane + slot * 32;
    memory[slot] = state[state_base + dimension];
  }
  for (unsigned int time = 0; time < tokens; ++time) {
    const unsigned int packed_token = row * tokens + time;
    const unsigned int gate_index = packed_token * value_heads + value_head;
    const float parameter = __bfloat162float(alpha[gate_index])
        + __bfloat162float(dt_bias[value_head]);
    const float softplus = fmaxf(parameter, 0.0f)
        + log1pf(expf(-fabsf(parameter)));
    const float step_decay = expf(
        -expf(__bfloat162float(a_log[value_head])) * softplus);
    const float step_update = 1.0f /
        (1.0f + expf(-__bfloat162float(beta[gate_index])));
    const unsigned int key_base = (packed_token * key_heads + key_head) * key_dim;
    float projection = 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      memory[slot] *= step_decay;
      projection += memory[slot] * __bfloat162float(key[key_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      projection += __shfl_down_sync(0xffffffffu, projection, offset);
    }
    const float target = __bfloat162float(
        value[(packed_token * value_heads + value_head) * value_dim + value_index]);
    const float delta = __shfl_sync(
        0xffffffffu, (target - projection) * step_update, 0);
    float result = 0.0f;
#pragma unroll
    for (unsigned int slot = 0; slot < slots; ++slot) {
      const unsigned int dimension = lane + slot * 32;
      memory[slot] += __bfloat162float(key[key_base + dimension]) * delta;
      result += memory[slot] * __bfloat162float(query[key_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane == 0) {
      output[(packed_token * value_heads + value_head) * value_dim + value_index]
          = __float2bfloat16_rn(result);
    }
  }
#pragma unroll
  for (unsigned int slot = 0; slot < slots; ++slot) {
    const unsigned int dimension = lane + slot * 32;
    state[state_base + dimension] = memory[slot];
  }
}

extern "C" __global__ void libmir_cuda_gated_delta_batch_recurrence_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* alpha,
    const __nv_bfloat16* beta, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, float* state, __nv_bfloat16* output,
    unsigned int rows, unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  switch (key_dim) {
    case 32: return gated_delta_register_batch_recurrence<1>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 64: return gated_delta_register_batch_recurrence<2>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 96: return gated_delta_register_batch_recurrence<3>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 128: return gated_delta_register_batch_recurrence<4>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 160: return gated_delta_register_batch_recurrence<5>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 192: return gated_delta_register_batch_recurrence<6>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 224: return gated_delta_register_batch_recurrence<7>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
    case 256: return gated_delta_register_batch_recurrence<8>(
        query, key, value, alpha, beta, a_log, dt_bias, state, output,
        rows, tokens, key_heads, value_heads, key_dim, value_dim);
  }
}
