#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_rope_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int tokens, unsigned int heads, unsigned int head_dim,
    unsigned int rotary_dim, unsigned int pairing_dim,
    unsigned int start_position, float theta) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int elements = tokens * heads * head_dim;
  if (index >= elements) return;
  const unsigned int dimension = index % head_dim;
  const unsigned int half = pairing_dim / 2u;
  const unsigned int pair = dimension % half;
  if (dimension >= pairing_dim || pair >= rotary_dim / 2u) {
    output[index] = input[index];
    return;
  }
  const unsigned int head_offset = index - dimension;
  const float first = __bfloat162float(input[head_offset + pair]);
  const float second = __bfloat162float(input[head_offset + pair + half]);
  const unsigned int token = index / (heads * head_dim);
  const float frequency = powf(theta, -2.0f * pair / pairing_dim);
  float sine;
  float cosine;
  sincosf((start_position + token) * frequency, &sine, &cosine);
  const float value = dimension < half
      ? fmaf(-second, sine, first * cosine)
      : fmaf(first, sine, second * cosine);
  output[index] = __float2bfloat16_rn(value);
}
