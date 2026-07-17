#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float libmir_fp8_e4m3(unsigned char value) {
  __nv_fp8_e4m3 scale;
  scale.__x = value;
  return static_cast<float>(scale);
}

__device__ __forceinline__ float2 libmir_nvfp4_pair(
    const unsigned char* weight, const unsigned char* scales, float global,
    unsigned int weight_base, unsigned int scale_base, unsigned int pair) {
  const unsigned char packed = weight[weight_base + pair];
  const float scale = libmir_fp8_e4m3(scales[scale_base + pair / 8u]) * global;
  __nv_fp4x2_e2m1 values;
  values.__x = packed;
  const float2 converted = static_cast<float2>(values);
  return make_float2(converted.x * scale, converted.y * scale);
}

__device__ float libmir_activation(float value, unsigned int activation) {
  if (activation == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

extern "C" __global__ void libmir_cuda_selected_nvfp4_gated_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const unsigned char* gate_weight, const unsigned char* gate_scales,
    const float* gate_global, const unsigned char* up_weight,
    const unsigned char* up_scales, const float* up_global,
    __nv_bfloat16* output, unsigned int input_features,
    unsigned int output_features, unsigned int selected_count,
    unsigned int tokens, unsigned int activation) {
  const unsigned int token = blockIdx.z;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int row = blockIdx.x * 8u + warp;
  const unsigned int rank = blockIdx.y;
  if (token >= tokens || row >= output_features || rank >= selected_count) return;
  input += token * input_features;
  selected += token * selected_count;
  output += token * selected_count * output_features;
  const unsigned int expert = selected[rank];
  const unsigned int pairs = input_features / 2u;
  const unsigned int weight_base =
      (expert * output_features + row) * pairs;
  const unsigned int scale_base =
      (expert * output_features + row) * (input_features / 16u);
  const __nv_bfloat162* input_pairs =
      reinterpret_cast<const __nv_bfloat162*>(input);
  float gate = 0.0f;
  float up = 0.0f;
  for (unsigned int pair = lane; pair < pairs; pair += 32u) {
    const float2 value = __bfloat1622float2(input_pairs[pair]);
    const float2 gate_pair = libmir_nvfp4_pair(
        gate_weight, gate_scales, gate_global[expert], weight_base,
        scale_base, pair);
    const float2 up_pair = libmir_nvfp4_pair(
        up_weight, up_scales, up_global[expert], weight_base,
        scale_base, pair);
    gate = fmaf(value.x, gate_pair.x, gate);
    gate = fmaf(value.y, gate_pair.y, gate);
    up = fmaf(value.x, up_pair.x, up);
    up = fmaf(value.y, up_pair.y, up);
  }
  for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
    gate += __shfl_down_sync(0xffffffffu, gate, stride);
    up += __shfl_down_sync(0xffffffffu, up, stride);
  }
  if (lane == 0u) {
    const float gated = libmir_activation(gate, activation) * up;
    output[rank * output_features + row] = __float2bfloat16_rn(gated);
  }
}

extern "C" __global__ void libmir_cuda_selected_nvfp4_reduce_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const __nv_bfloat16* routing, const unsigned char* weight,
    const unsigned char* scales, const float* global_scales,
    __nv_bfloat16* output, unsigned int input_features,
    unsigned int output_features, unsigned int selected_count,
    unsigned int tokens) {
  const unsigned int token = blockIdx.y;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int row = blockIdx.x * 8u + warp;
  if (token >= tokens || row >= output_features) return;
  input += token * selected_count * input_features;
  selected += token * selected_count;
  routing += token * selected_count;
  output += token * output_features;
  float reduced = 0.0f;
  const unsigned int pairs = input_features / 2u;
  const __nv_bfloat162* input_pairs =
      reinterpret_cast<const __nv_bfloat162*>(input);
  for (unsigned int rank = 0; rank < selected_count; ++rank) {
    const unsigned int expert = selected[rank];
    const unsigned int weight_base =
        (expert * output_features + row) * pairs;
    const unsigned int scale_base =
        (expert * output_features + row) * (input_features / 16u);
    float sum = 0.0f;
    const unsigned int input_base = rank * pairs;
    for (unsigned int pair = lane; pair < pairs; pair += 32u) {
      const float2 value = __bfloat1622float2(input_pairs[input_base + pair]);
      const float2 weight_pair = libmir_nvfp4_pair(
          weight, scales, global_scales[expert], weight_base, scale_base,
          pair);
      sum = fmaf(value.x, weight_pair.x, sum);
      sum = fmaf(value.y, weight_pair.y, sum);
    }
    for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
      sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }
    if (lane == 0u) reduced = fmaf(sum, __bfloat162float(routing[rank]), reduced);
  }
  if (lane == 0u) output[row] = __float2bfloat16_rn(reduced);
}
