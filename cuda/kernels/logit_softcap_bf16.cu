#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_logit_softcap_bf16(
    __nv_bfloat16* values, unsigned int elements, float cap) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = __bfloat162float(values[index]);
  values[index] = __float2bfloat16_rn(tanhf(value / cap) * cap);
}
