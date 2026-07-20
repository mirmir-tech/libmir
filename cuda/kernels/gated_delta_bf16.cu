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

extern "C" __global__ void libmir_cuda_gated_delta_recurrence_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* alpha,
    const __nv_bfloat16* beta, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, float* state, __nv_bfloat16* output,
    unsigned int tokens, unsigned int key_heads, unsigned int value_heads,
    unsigned int key_dim, unsigned int value_dim) {
  const unsigned int lane = threadIdx.x;
  const unsigned int value_index = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int value_head = blockIdx.z;
  if (value_index >= value_dim || value_head >= value_heads) return;
  const unsigned int key_head = value_head / (value_heads / key_heads);
  const unsigned int state_base = (value_head * value_dim + value_index) * key_dim;
  for (unsigned int time = 0; time < tokens; ++time) {
    const unsigned int gate_index = time * value_heads + value_head;
    const float parameter = __bfloat162float(alpha[gate_index])
        + __bfloat162float(dt_bias[value_head]);
    const float softplus = fmaxf(parameter, 0.0f) + log1pf(expf(-fabsf(parameter)));
    const float decay = expf(-expf(__bfloat162float(a_log[value_head])) * softplus);
    const float update = 1.0f / (1.0f + expf(-__bfloat162float(beta[gate_index])));
    const unsigned int key_base = (time * key_heads + key_head) * key_dim;
    float projection = 0.0f;
    for (unsigned int dimension = lane; dimension < key_dim; dimension += 32) {
      const unsigned int index = state_base + dimension;
      const float memory = state[index] * decay;
      state[index] = memory;
      projection += memory * __bfloat162float(key[key_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      projection += __shfl_down_sync(0xffffffffu, projection, offset);
    }
    const float target = __bfloat162float(
        value[(time * value_heads + value_head) * value_dim + value_index]);
    const float delta = __shfl_sync(
        0xffffffffu, (target - projection) * update, 0);
    float result = 0.0f;
    for (unsigned int dimension = lane; dimension < key_dim; dimension += 32) {
      const unsigned int index = state_base + dimension;
      const float memory = state[index]
          + __bfloat162float(key[key_base + dimension]) * delta;
      state[index] = memory;
      const unsigned int query_base = (time * key_heads + key_head) * key_dim;
      result += memory * __bfloat162float(query[query_base + dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane == 0) {
      output[(time * value_heads + value_head) * value_dim + value_index]
          = __float2bfloat16_rn(result);
    }
  }
}
