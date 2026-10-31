#include <cuda_bf16.h>

// Mamba-2 selective state-space mixer kernels.
//
// The fused input projection of one token is laid out as
//   z[inner] | x[inner] | B[groups * state] | C[groups * state] | dt[heads]
// where `x | B | C` pass through the causal depthwise convolution. The
// recurrent state of one head is a [head_dim, state] FP32 matrix:
//   S = exp(dt * A) * S + dt * (x outer B),   y = S C + D x.

#define LIBMIR_MAMBA2_SCAN_ROWS 16u

// One thread per (token, channel): causal depthwise convolution over the
// previous `kernel_size - 1` history tokens and the current chunk, plus an
// optional bias, followed by SiLU.
extern "C" __global__ void libmir_cuda_mamba2_convolution_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, const __nv_bfloat16* history,
    __nv_bfloat16* output, unsigned int tokens, unsigned int channels,
    unsigned int kernel_size, unsigned int input_stride,
    unsigned int input_offset, unsigned int has_bias) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * channels) return;
  const unsigned int token = index / channels;
  const unsigned int channel = index % channels;
  float sum = has_bias != 0u ? __bfloat162float(bias[channel]) : 0.0f;
  for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
    const unsigned int position = token + kernel;
    const float source = position < kernel_size - 1u
        ? __bfloat162float(history[position * channels + channel])
        : __bfloat162float(input[(position - kernel_size + 1u) * input_stride +
                                input_offset + channel]);
    sum += source * __bfloat162float(weight[channel * kernel_size + kernel]);
  }
  output[index] = __float2bfloat16_rn(sum / (1.0f + __expf(-sum)));
}

// Keeps the last `kernel_size - 1` convolution inputs for the next chunk.
extern "C" __global__ void libmir_cuda_mamba2_history_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* history,
    __nv_bfloat16* next_history, unsigned int tokens, unsigned int channels,
    unsigned int kernel_size, unsigned int input_stride,
    unsigned int input_offset) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int history_tokens = kernel_size - 1u;
  if (index >= history_tokens * channels) return;
  const unsigned int history_token = index / channels;
  const unsigned int channel = index % channels;
  const unsigned int combined = tokens + history_token;
  next_history[index] = combined < history_tokens
      ? history[combined * channels + channel]
      : input[(combined - history_tokens) * input_stride + input_offset +
              channel];
}

__device__ __forceinline__ float libmir_mamba2_softplus(float value) {
  return value > 20.0f ? value : log1pf(__expf(value));
}

// Sequential selective scan of one sequence. Block (head, head_dim / ROWS)
// comes from blockIdx.x/y; one thread per state column (`state_size`
// threads, a multiple of 32). Each thread keeps ROWS state rows of its column
// in registers for all tokens.
__device__ __forceinline__ void libmir_mamba2_scan(
    const __nv_bfloat16* convolved, const __nv_bfloat16* projection,
    const __nv_bfloat16* a_log, const __nv_bfloat16* skip,
    const __nv_bfloat16* dt_bias, float* state, __nv_bfloat16* output,
    unsigned int tokens, unsigned int heads, unsigned int head_dim,
    unsigned int state_size, unsigned int groups,
    unsigned int projection_stride, unsigned int dt_offset,
    float dt_minimum, float dt_maximum) {
  constexpr unsigned int rows = LIBMIR_MAMBA2_SCAN_ROWS;
  const unsigned int head = blockIdx.x;
  const unsigned int row_base = blockIdx.y * rows;
  const unsigned int column = threadIdx.x;
  const unsigned int lane = column % 32u;
  const unsigned int warp = column / 32u;
  const unsigned int warps = blockDim.x / 32u;
  const unsigned int inner = heads * head_dim;
  const unsigned int convolved_width = inner + 2u * groups * state_size;
  const unsigned int group = head / (heads / groups);
  const unsigned int b_offset = inner + group * state_size + column;
  const unsigned int c_offset = inner + (groups + group) * state_size + column;
  __shared__ float x_rows[rows];
  __shared__ float step[2];
  __shared__ float partial[32][rows];

  float values[rows];
  float* head_state = state + (static_cast<size_t>(head) * head_dim + row_base) *
                                  state_size + column;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    values[row] = head_state[static_cast<size_t>(row) * state_size];
  }
  const float decay_rate = -__expf(__bfloat162float(a_log[head]));
  const float skip_scale = __bfloat162float(skip[head]);
  const float bias = __bfloat162float(dt_bias[head]);

  for (unsigned int token = 0; token < tokens; ++token) {
    const __nv_bfloat16* token_values = convolved + static_cast<size_t>(token) * convolved_width;
    if (column < rows) {
      x_rows[column] = __bfloat162float(token_values[head * head_dim + row_base + column]);
    }
    if (column == 0u) {
      const float raw = __bfloat162float(
          projection[static_cast<size_t>(token) * projection_stride + dt_offset + head]);
      const float dt = fminf(fmaxf(libmir_mamba2_softplus(raw + bias), dt_minimum), dt_maximum);
      step[0] = dt;
      step[1] = __expf(dt * decay_rate);
    }
    __syncthreads();
    const float dt = step[0];
    const float decay = step[1];
    const float b = __bfloat162float(token_values[b_offset]);
    const float c = __bfloat162float(token_values[c_offset]);
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
      values[row] = values[row] * decay + dt * x_rows[row] * b;
      float sum = values[row] * c;
      for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
      }
      if (lane == 0u) partial[warp][row] = sum;
    }
    __syncthreads();
    if (column < rows) {
      float sum = 0.0f;
      for (unsigned int index = 0; index < warps; ++index) sum += partial[index][column];
      output[static_cast<size_t>(token) * inner + head * head_dim + row_base + column] =
          __float2bfloat16_rn(sum + skip_scale * x_rows[column]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    head_state[static_cast<size_t>(row) * state_size] = values[row];
  }
}

// Grid: (heads, head_dim / ROWS); see `libmir_mamba2_scan`.
extern "C" __global__ void libmir_cuda_mamba2_scan_bf16(
    const __nv_bfloat16* convolved, const __nv_bfloat16* projection,
    const __nv_bfloat16* a_log, const __nv_bfloat16* skip,
    const __nv_bfloat16* dt_bias, float* state, __nv_bfloat16* output,
    unsigned int tokens, unsigned int heads, unsigned int head_dim,
    unsigned int state_size, unsigned int groups,
    unsigned int projection_stride, unsigned int dt_offset,
    float dt_minimum, float dt_maximum) {
  libmir_mamba2_scan(convolved, projection, a_log, skip, dt_bias, state, output,
                     tokens, heads, head_dim, state_size, groups,
                     projection_stride, dt_offset, dt_minimum, dt_maximum);
}

// One decode token for each of `gridDim.z` independent sequences whose
// states are packed contiguously, one row per sequence.
// Grid: (heads, head_dim / ROWS, rows).
extern "C" __global__ void libmir_cuda_mamba2_decode_scan_bf16(
    const __nv_bfloat16* convolved, const __nv_bfloat16* projection,
    const __nv_bfloat16* a_log, const __nv_bfloat16* skip,
    const __nv_bfloat16* dt_bias, float* states, __nv_bfloat16* output,
    unsigned int heads, unsigned int head_dim, unsigned int state_size,
    unsigned int groups, unsigned int projection_stride,
    unsigned int dt_offset, float dt_minimum, float dt_maximum) {
  const size_t row = blockIdx.z;
  const unsigned int inner = heads * head_dim;
  libmir_mamba2_scan(
      convolved + row * (inner + 2u * groups * state_size),
      projection + row * projection_stride, a_log, skip, dt_bias,
      states + row * inner * state_size, output + row * inner, 1u, heads,
      head_dim, state_size, groups, projection_stride, dt_offset, dt_minimum,
      dt_maximum);
}

// One decode token for each of `rows` independent sequences. One thread per
// (row, channel): convolves the row's history with its current input in the
// same order as `libmir_cuda_mamba2_convolution_bf16`, then shifts the
// history in place.
extern "C" __global__ void libmir_cuda_mamba2_decode_convolution_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* histories,
    __nv_bfloat16* output, unsigned int rows, unsigned int channels,
    unsigned int kernel_size, unsigned int input_stride,
    unsigned int input_offset, unsigned int has_bias) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * channels) return;
  const unsigned int row = index / channels;
  const unsigned int channel = index % channels;
  const unsigned int history_tokens = kernel_size - 1u;
  __nv_bfloat16* history =
      histories + static_cast<size_t>(row) * history_tokens * channels;
  const __nv_bfloat16 current =
      input[static_cast<size_t>(row) * input_stride + input_offset + channel];
  float sum = has_bias != 0u ? __bfloat162float(bias[channel]) : 0.0f;
  for (unsigned int kernel = 0; kernel < kernel_size; ++kernel) {
    const float source = kernel < history_tokens
        ? __bfloat162float(history[kernel * channels + channel])
        : __bfloat162float(current);
    sum += source * __bfloat162float(weight[channel * kernel_size + kernel]);
  }
  for (unsigned int kernel = 0; kernel + 1u < history_tokens; ++kernel) {
    history[kernel * channels + channel] = history[(kernel + 1u) * channels + channel];
  }
  history[(history_tokens - 1u) * channels + channel] = current;
  output[index] = __float2bfloat16_rn(sum / (1.0f + __expf(-sum)));
}

// Gated grouped RMSNorm: h = y * silu(z) in FP32, normalized over groups of
// `group_size` channels, rounded to BF16, then scaled by the BF16 weight.
// Grid: tokens * (inner / group_size); block: 256 threads.
extern "C" __global__ void libmir_cuda_mamba2_gated_norm_bf16(
    const __nv_bfloat16* values, const __nv_bfloat16* projection,
    const __nv_bfloat16* weight, __nv_bfloat16* output, unsigned int tokens,
    unsigned int inner, unsigned int group_size,
    unsigned int projection_stride, float epsilon) {
  const unsigned int groups = inner / group_size;
  const unsigned int token = blockIdx.x / groups;
  const unsigned int group = blockIdx.x % groups;
  if (token >= tokens) return;
  const unsigned int base = group * group_size;
  const __nv_bfloat16* row = values + static_cast<size_t>(token) * inner;
  const __nv_bfloat16* gate = projection + static_cast<size_t>(token) * projection_stride;
  float sum = 0.0f;
  for (unsigned int index = threadIdx.x; index < group_size; index += blockDim.x) {
    const float z = __bfloat162float(gate[base + index]);
    const float hidden = __bfloat162float(row[base + index]) * (z / (1.0f + __expf(-z)));
    sum += hidden * hidden;
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, delta);
  }
  __shared__ float warp_sums[32];
  __shared__ float scale;
  if (threadIdx.x % 32u == 0u) warp_sums[threadIdx.x / 32u] = sum;
  __syncthreads();
  if (threadIdx.x == 0u) {
    float total = 0.0f;
    for (unsigned int index = 0; index < blockDim.x / 32u; ++index) total += warp_sums[index];
    scale = rsqrtf(total / static_cast<float>(group_size) + epsilon);
  }
  __syncthreads();
  __nv_bfloat16* destination = output + static_cast<size_t>(token) * inner;
  for (unsigned int index = threadIdx.x; index < group_size; index += blockDim.x) {
    const float z = __bfloat162float(gate[base + index]);
    const float hidden = __bfloat162float(row[base + index]) * (z / (1.0f + __expf(-z)));
    const float normalized = __bfloat162float(__float2bfloat16_rn(hidden * scale));
    destination[base + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(weight[base + index]));
  }
}
