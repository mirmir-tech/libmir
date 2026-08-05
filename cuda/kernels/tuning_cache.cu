#include <cuda_runtime.h>

extern "C" __global__ void libmir_cuda_tuning_cache_evict(
    uint4* values, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    uint4 value = values[index];
    value.x ^= 0x9e3779b9u + index;
    value.y ^= 0x7f4a7c15u + index;
    value.z ^= 0xf39cc060u + index;
    value.w ^= 0x106aa070u + index;
    values[index] = value;
  }
}
