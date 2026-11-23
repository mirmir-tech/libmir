// GGML MXFP4 expert banks to the OCP MXFP4 layout of Hugging Face gpt-oss
// checkpoints, which the routed-expert kernels read. Lossless: both store
// e2m1 codes and an E8M0 scale per 32 values; only the code order differs.
// GGML `block_mxfp4` (17 B): e, then qs[16] with value j in the low nibble of
// qs[j] and value j + 16 in its high nibble. OCP: blocks[16] with values 2k
// and 2k + 1 in the low and high nibble of byte k, scales apart.
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ unsigned int ggml_mxfp4_code(const unsigned char* qs, unsigned int j) {
  return j < 16u ? qs[j] & 0xFu : qs[j - 16u] >> 4u;
}

}  // namespace

// One thread per block. Source row r becomes output row r * stride + offset,
// which interleaves separate gate and up banks (stride 2, offsets 0 and 1)
// into the fused gate/up bank: per expert, gate and up rows alternate.
extern "C" __global__ void libmir_cuda_ggml_mxfp4_to_ocp(
    const unsigned char* source, unsigned char* blocks, unsigned char* scales, unsigned int rows,
    unsigned int blocks_per_row, unsigned int stride, unsigned int offset) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(rows) * blocks_per_row) return;
  const unsigned long long row = index / blocks_per_row;
  const unsigned int block = static_cast<unsigned int>(index % blocks_per_row);
  const unsigned char* from = source + index * 17u;
  const unsigned long long target = (row * stride + offset) * blocks_per_row + block;
  scales[target] = from[0];
  unsigned char* to = blocks + target * 16u;
#pragma unroll
  for (unsigned int k = 0; k < 16u; ++k) {
    const unsigned int low = ggml_mxfp4_code(from + 1u, 2u * k);
    const unsigned int high = ggml_mxfp4_code(from + 1u, 2u * k + 1u);
    to[k] = static_cast<unsigned char>(low | (high << 4u));
  }
}

// Bias rows in the same order: element i becomes i * stride + offset.
extern "C" __global__ void libmir_cuda_ggml_interleave_bf16(
    const __nv_bfloat16* source, __nv_bfloat16* output, unsigned int count, unsigned int stride,
    unsigned int offset) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[static_cast<unsigned long long>(index) * stride + offset] = source[index];
}
