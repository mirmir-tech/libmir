#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_activation_fingerprint_bf16(
    const __nv_bfloat16* input, unsigned long long* output,
    unsigned int elements, unsigned int layer) {
  __shared__ unsigned long long sums[256];
  __shared__ unsigned long long weighted[256];
  const unsigned int lane = threadIdx.x;
  const unsigned short* bits = reinterpret_cast<const unsigned short*>(input);
  unsigned long long sum_value = 0;
  unsigned long long weighted_value = 0;
  for (unsigned int index = lane; index < elements; index += blockDim.x) {
    const unsigned long long value = bits[index];
    sum_value += value;
    weighted_value += value * (static_cast<unsigned long long>(index) + 1);
  }
  sums[lane] = sum_value;
  weighted[lane] = weighted_value;
  __syncthreads();
  if (lane == 0) {
    for (unsigned int index = 1; index < blockDim.x; ++index) {
      sums[0] += sums[index];
      weighted[0] += weighted[index];
    }
    output[layer * 2] = sums[0];
    output[layer * 2 + 1] = weighted[0];
  }
}
