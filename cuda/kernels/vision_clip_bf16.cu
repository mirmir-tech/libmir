#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_clip_bounds_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* minimum,
    const __nv_bfloat16* maximum, __nv_bfloat16* output,
    unsigned int elements, unsigned int columns, unsigned int bounds) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int bound = bounds == 1 ? 0 : index % columns;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(
      fminf(fmaxf(value, __bfloat162float(minimum[bound])),
            __bfloat162float(maximum[bound])));
}

extern "C" __global__ void libmir_cuda_vision_clip_bounds_f32(
    const __nv_bfloat16* input, const float* minimum,
    const float* maximum, __nv_bfloat16* output,
    unsigned int elements, unsigned int columns, unsigned int bounds) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int bound = bounds == 1 ? 0 : index % columns;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(
      fminf(fmaxf(value, minimum[bound]), maximum[bound]));
}
