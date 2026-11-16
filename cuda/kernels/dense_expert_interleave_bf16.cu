#include <cuda_bf16.h>

// Gate and up banks `[rows, columns]` to one fused bank of `2 * rows` rows:
// gate row r to row 2r and up row r to row 2r + 1, which keeps each expert's
// rows together when `rows` spans whole experts. Eight values per thread.
extern "C" __global__ void libmir_cuda_dense_expert_interleave_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long vectors = static_cast<unsigned long long>(columns) / 8u;
  if (index >= static_cast<unsigned long long>(rows) * vectors * 2u) return;
  const unsigned long long source = index >> 1u;
  const unsigned long long row = source / vectors;
  const unsigned long long vector = source % vectors;
  const __nv_bfloat16* from = (index & 1u) != 0u ? up : gate;
  const uint4 value = reinterpret_cast<const uint4*>(from)[source];
  reinterpret_cast<uint4*>(output)[(row * 2u + (index & 1u)) * vectors + vector] = value;
}
