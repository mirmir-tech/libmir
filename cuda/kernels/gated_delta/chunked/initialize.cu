#include <cuda_bf16.h>

// The triangular solve writes only its lower block triangle. Its consumers
// load the full matrix, so the remaining entries must be initialized to zero.
extern "C" __global__ void libmir_cuda_gdn_initialize_inverse_bf16(
    __nv_bfloat16* inverse, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) inverse[index] = __float2bfloat16(0.0f);
}
