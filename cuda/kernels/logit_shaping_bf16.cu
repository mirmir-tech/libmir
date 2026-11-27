#include <cuda_bf16.h>

// Output logits divided by `divisor` (Granite `logits_scaling`), then, with a
// positive `cap`, soft-capped as `cap * tanh(value / cap)` (Gemma).
extern "C" __global__ void libmir_cuda_logit_shaping_bf16(
    __nv_bfloat16* values, unsigned int elements, float divisor, float cap) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = __bfloat162float(values[index]) / divisor;
  if (cap > 0.0f) value = tanhf(value / cap) * cap;
  values[index] = __float2bfloat16_rn(value);
}
