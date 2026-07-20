#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_mrope_bf16(
    const __nv_bfloat16* input, const unsigned int* positions,
    __nv_bfloat16* output, unsigned int tokens, unsigned int heads,
    unsigned int head_dim, unsigned int rotary_dim,
    unsigned int section_t, unsigned int section_h, unsigned int section_w,
    unsigned int interleaved, float theta) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = tokens * heads * head_dim;
  if (index >= elements) return;
  const unsigned int dimension = index % head_dim;
  if (dimension >= rotary_dim) {
    output[index] = input[index];
    return;
  }
  const unsigned int half = rotary_dim / 2;
  const unsigned int frequency = dimension % half;
  unsigned int axis;
  if (interleaved != 0) {
    axis = frequency % 3 == 1 && frequency < 3 * section_h ? 1
        : frequency % 3 == 2 && frequency < 3 * section_w ? 2 : 0;
  } else {
    axis = frequency < section_t ? 0
        : frequency < section_t + section_h ? 1 : 2;
  }
  const unsigned int token = index / (heads * head_dim);
  const float inverse = powf(theta, -2.0f * frequency / rotary_dim);
  float sine;
  float cosine;
  sincosf(static_cast<float>(positions[axis * tokens + token]) * inverse, &sine, &cosine);
  const unsigned int base = index - dimension;
  const float first = __bfloat162float(input[base + frequency]);
  const float second = __bfloat162float(input[base + frequency + half]);
  const float value = dimension < half
      ? fmaf(-second, sine, first * cosine)
      : fmaf(first, sine, second * cosine);
  output[index] = __float2bfloat16_rn(value);
}
