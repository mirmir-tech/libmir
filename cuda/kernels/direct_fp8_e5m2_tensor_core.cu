#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

namespace {
constexpr unsigned int kTileM = 16;
constexpr unsigned int kTileN = 16;
constexpr unsigned int kTileK = 16;
constexpr unsigned int kWarps = 8;
constexpr unsigned int kBlockN = kWarps * kTileN;

__device__ __forceinline__ __nv_bfloat16 decode_e5m2(unsigned char raw) {
  __nv_fp8_e5m2 value;
  value.__x = raw;
  return __float2bfloat16_rn(static_cast<float>(value));
}
}  // namespace

extern "C" __global__ void libmir_cuda_e5m2_bf16_tensor_core_linear(
    const __nv_bfloat16* input, const unsigned char* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int tokens, unsigned int rows, unsigned int columns,
    unsigned int has_bias) {
  using namespace nvcuda;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int token_base = blockIdx.y * kTileM;
  const unsigned int row_base = blockIdx.x * kBlockN;
  __shared__ __nv_bfloat16 tile_a[kTileM * kTileK];
  __shared__ __nv_bfloat16 tile_b[kBlockN * kTileK];
  __shared__ float tile_c[kBlockN * kTileM];
  wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK,
                 __nv_bfloat16, wmma::row_major> fragment_a;
  wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK,
                 __nv_bfloat16, wmma::col_major> fragment_b;
  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (unsigned int column_base = 0; column_base < columns;
       column_base += kTileK) {
    for (unsigned int index = threadIdx.x; index < kTileM * kTileK;
         index += blockDim.x) {
      const unsigned int token = token_base + index / kTileK;
      const unsigned int column = column_base + index % kTileK;
      tile_a[index] = token < tokens && column < columns
                          ? input[token * columns + column]
                          : __float2bfloat16(0.0f);
    }
    for (unsigned int index = threadIdx.x; index < kBlockN * kTileK;
         index += blockDim.x) {
      const unsigned int row = row_base + index / kTileK;
      const unsigned int column = column_base + index % kTileK;
      tile_b[index] = row < rows && column < columns
                          ? decode_e5m2(weight[row * columns + column])
                          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::load_matrix_sync(fragment_a, tile_a, kTileK);
    wmma::load_matrix_sync(fragment_b, tile_b + warp * kTileN * kTileK,
                           kTileK);
    wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    __syncthreads();
  }
  wmma::store_matrix_sync(tile_c + warp * kTileM * kTileN, accumulator,
                          kTileN, wmma::mem_row_major);
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < kBlockN * kTileM;
       index += blockDim.x) {
    const unsigned int local_row = index / kTileM;
    const unsigned int local_token = index % kTileM;
    const unsigned int row = row_base + local_row;
    const unsigned int token = token_base + local_token;
    if (row < rows && token < tokens) {
      const unsigned int warp_row = local_row / kTileN;
      const unsigned int warp_column = local_row % kTileN;
      const float value = tile_c[warp_row * kTileM * kTileN +
                                 local_token * kTileN + warp_column] +
                          (has_bias == 0u ? 0.0f
                                          : __bfloat162float(bias[row]));
      output[token * rows + row] = __float2bfloat16_rn(value);
    }
  }
}
