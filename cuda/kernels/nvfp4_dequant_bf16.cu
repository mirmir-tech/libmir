#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

extern "C" __global__ void libmir_cuda_nvfp4_dequant_bf16(
    const unsigned char* packed, const unsigned char* block_scales,
    const float* global_scale, unsigned short* output,
    unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;

  __nv_fp4x2_e2m1 pair;
  pair.__x = packed[index / 2u];
  const float2 values = static_cast<float2>(pair);

  __nv_fp8_e4m3 scale;
  scale.__x = block_scales[index / 16u];
  const float value = (index & 1u) == 0u ? values.x : values.y;
  const __nv_bfloat16 converted =
      __float2bfloat16_rn(value * static_cast<float>(scale) * global_scale[0]);
  output[index] = reinterpret_cast<const unsigned short&>(converted);
}
