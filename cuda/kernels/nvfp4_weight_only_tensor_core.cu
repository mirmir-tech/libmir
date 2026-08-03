#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mma.h>

namespace {
constexpr unsigned int kTile = 16;
constexpr unsigned int kWarps = 8;
constexpr unsigned int kBlockRows = kWarps * kTile;

__device__ __forceinline__ float decode_scale(unsigned char raw) {
  __nv_fp8_e4m3 value;
  value.__x = raw;
  return static_cast<float>(value);
}
}  // namespace

extern "C" __global__ void libmir_cuda_nvfp4_weight_only_tensor_core_bf16(
    const __nv_bfloat16* input, const unsigned char* weight,
    const unsigned char* block_scales, const float* global_scale,
    __nv_bfloat16* output, unsigned int input_features,
    unsigned int output_features, unsigned int tokens) {
  using namespace nvcuda;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int token_base = blockIdx.y * kTile;
  const unsigned int row_base = blockIdx.x * kBlockRows;
  __shared__ __nv_bfloat16 tile_a[kTile * kTile];
  __shared__ __nv_bfloat16 tile_b[kBlockRows * kTile];
  __shared__ float tile_c[kBlockRows * kTile];
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::row_major> fragment_a;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::col_major> fragment_b;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  const float global = global_scale[0];
  for (unsigned int column_base = 0; column_base < input_features;
       column_base += kTile) {
    const unsigned int input_elements = kTile * kTile;
    for (unsigned int index = threadIdx.x; index < input_elements;
         index += blockDim.x) {
      const unsigned int token = token_base + index / kTile;
      const unsigned int column = column_base + index % kTile;
      tile_a[index] = token < tokens
          ? input[token * input_features + column]
          : __float2bfloat16(0.0f);
    }
    const unsigned int pairs = kBlockRows * kTile / 2u;
    for (unsigned int index = threadIdx.x; index < pairs;
         index += blockDim.x) {
      const unsigned int local_row = index / (kTile / 2u);
      const unsigned int local_pair = index % (kTile / 2u);
      const unsigned int row = row_base + local_row;
      const unsigned int column = column_base + local_pair * 2u;
      float2 values = make_float2(0.0f, 0.0f);
      if (row < output_features) {
        __nv_fp4x2_e2m1 packed;
        packed.__x = weight[row * input_features / 2u + column / 2u];
        values = static_cast<float2>(packed);
        const float scale = decode_scale(
            block_scales[row * input_features / 16u + column / 16u]) * global;
        values.x *= scale;
        values.y *= scale;
      }
      const unsigned int target = local_row * kTile + local_pair * 2u;
      tile_b[target] = __float2bfloat16_rn(values.x);
      tile_b[target + 1u] = __float2bfloat16_rn(values.y);
    }
    __syncthreads();
    wmma::load_matrix_sync(fragment_a, tile_a, kTile);
    wmma::load_matrix_sync(fragment_b, tile_b + warp * kTile * kTile, kTile);
    wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    __syncthreads();
  }
  wmma::store_matrix_sync(tile_c + warp * kTile * kTile, accumulator,
                          kTile, wmma::mem_row_major);
  __syncthreads();
  for (unsigned int index = threadIdx.x; index < kBlockRows * kTile;
       index += blockDim.x) {
    const unsigned int local_row = index / kTile;
    const unsigned int local_token = index % kTile;
    const unsigned int row = row_base + local_row;
    const unsigned int token = token_base + local_token;
    if (row < output_features && token < tokens) {
      const unsigned int output_warp = local_row / kTile;
      const unsigned int output_column = local_row % kTile;
      const float value = tile_c[output_warp * kTile * kTile +
                                 local_token * kTile + output_column];
      output[token * output_features + row] = __float2bfloat16_rn(value);
    }
  }
}
