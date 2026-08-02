#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float libmir_nvfp4_weight_only_scale(
    unsigned char encoded) {
  __nv_fp8_e4m3 scale;
  scale.__x = encoded;
  return static_cast<float>(scale);
}

extern "C" __global__ void libmir_cuda_nvfp4_weight_only_bf16(
    const __nv_bfloat16* input, const unsigned char* weight,
    const unsigned char* block_scales, const float* global_scale,
    __nv_bfloat16* output, unsigned int input_features,
    unsigned int output_features, unsigned int tokens) {
  const unsigned int token = blockIdx.y;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int row = blockIdx.x * 8u + warp;
  if (token >= tokens || row >= output_features) return;

  const unsigned int pairs = input_features / 2u;
  const unsigned int weight_base = row * pairs;
  const unsigned int scale_base = row * (input_features / 16u);
  const __nv_bfloat162* values =
      reinterpret_cast<const __nv_bfloat162*>(input + token * input_features);
  float sum = 0.0f;
  for (unsigned int pair = lane; pair < pairs; pair += 32u) {
    const float2 value = __bfloat1622float2(values[pair]);
    __nv_fp4x2_e2m1 packed;
    packed.__x = weight[weight_base + pair];
    const float2 converted = static_cast<float2>(packed);
    const float scale =
        libmir_nvfp4_weight_only_scale(block_scales[scale_base + pair / 8u]) *
        global_scale[0];
    sum = fmaf(value.x, converted.x * scale, sum);
    sum = fmaf(value.y, converted.y * scale, sum);
  }
  for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
    sum += __shfl_down_sync(0xffffffffu, sum, stride);
  }
  if (lane == 0u) {
    output[token * output_features + row] = __float2bfloat16_rn(sum);
  }
}
