#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mma.h>

namespace {
constexpr unsigned int kTile = 16;
constexpr unsigned int kWarps = 8;
constexpr unsigned int kColumns = kTile * kWarps;

__device__ __forceinline__ float decode_scale(unsigned char raw) {
  __nv_fp8_e4m3 value;
  value.__x = raw;
  return static_cast<float>(value);
}

__device__ __forceinline__ float activation(float value,
                                             unsigned int kind) {
  if (kind == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

__device__ __forceinline__ void load_input_tile(
    const __nv_bfloat16* input, __nv_bfloat16* tile,
    unsigned int column_base, unsigned int input_features) {
  const unsigned int index = threadIdx.x;
  const unsigned int row = index / kTile;
  const unsigned int column = index % kTile;
  tile[index] = row == 0u
      ? input[column_base + column]
      : __float2bfloat16(0.0f);
}

__device__ __forceinline__ void load_weight_tile(
    const unsigned char* weight, const unsigned char* scales,
    __nv_bfloat16* tile, unsigned int expert, unsigned int row_base,
    unsigned int column_base, unsigned int input_features,
    unsigned int output_features, float global) {
  constexpr unsigned int pairs = kColumns * kTile / 2u;
  for (unsigned int index = threadIdx.x; index < pairs;
       index += blockDim.x) {
    const unsigned int local_row = index / (kTile / 2u);
    const unsigned int local_pair = index % (kTile / 2u);
    const unsigned int row = row_base + local_row;
    const unsigned int column = column_base + local_pair * 2u;
    float2 values = make_float2(0.0f, 0.0f);
    if (row < output_features) {
      const unsigned int weight_base =
          (expert * output_features + row) * input_features / 2u;
      const unsigned int scale_base =
          (expert * output_features + row) * input_features / 16u;
      __nv_fp4x2_e2m1 packed;
      packed.__x = weight[weight_base + column / 2u];
      values = static_cast<float2>(packed);
      const float scale = decode_scale(scales[scale_base + column / 16u]) * global;
      values.x *= scale;
      values.y *= scale;
    }
    const unsigned int target = local_row * kTile + local_pair * 2u;
    tile[target] = __float2bfloat16_rn(values.x);
    tile[target + 1u] = __float2bfloat16_rn(values.y);
  }
}
}  // namespace

extern "C" __global__ void libmir_cuda_selected_nvfp4_tensor_core_gated_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const unsigned char* gate_weight, const unsigned char* gate_scales,
    const float* gate_global, const unsigned char* up_weight,
    const unsigned char* up_scales, const float* up_global,
    __nv_bfloat16* output, unsigned int input_features,
    unsigned int output_features, unsigned int selected_count,
    unsigned int routes, unsigned int activation_kind) {
  using namespace nvcuda;
  const unsigned int route = blockIdx.y;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int row_base = blockIdx.x * kColumns;
  if (route >= routes) return;
  const unsigned int expert = selected[route];
  const unsigned int token = route / selected_count;
  input += token * input_features;
  output += route * output_features;
  __shared__ __nv_bfloat16 tile_a[kTile * kTile];
  __shared__ __nv_bfloat16 tile_b[kColumns * kTile];
  __shared__ float gate_result[kColumns * kTile];
  __shared__ float up_result[kColumns * kTile];
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::row_major> fragment_a;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::col_major> fragment_b;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> gate_acc;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> up_acc;
  wmma::fill_fragment(gate_acc, 0.0f);
  wmma::fill_fragment(up_acc, 0.0f);
  for (unsigned int column = 0; column < input_features; column += kTile) {
    load_input_tile(input, tile_a, column, input_features);
    load_weight_tile(gate_weight, gate_scales, tile_b, expert, row_base,
                     column, input_features, output_features,
                     gate_global[expert]);
    __syncthreads();
    wmma::load_matrix_sync(fragment_a, tile_a, kTile);
    wmma::load_matrix_sync(fragment_b, tile_b + warp * kTile * kTile, kTile);
    wmma::mma_sync(gate_acc, fragment_a, fragment_b, gate_acc);
    __syncthreads();
    load_weight_tile(up_weight, up_scales, tile_b, expert, row_base,
                     column, input_features, output_features,
                     up_global[expert]);
    __syncthreads();
    wmma::load_matrix_sync(fragment_b, tile_b + warp * kTile * kTile, kTile);
    wmma::mma_sync(up_acc, fragment_a, fragment_b, up_acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(gate_result + warp * kTile * kTile, gate_acc,
                          kTile, wmma::mem_row_major);
  wmma::store_matrix_sync(up_result + warp * kTile * kTile, up_acc,
                          kTile, wmma::mem_row_major);
  __syncthreads();
  if (threadIdx.x < kColumns) {
    const unsigned int row = row_base + threadIdx.x;
    if (row < output_features) {
      const unsigned int offset =
          (threadIdx.x / kTile) * kTile * kTile + threadIdx.x % kTile;
      output[row] = __float2bfloat16_rn(
          activation(gate_result[offset], activation_kind) * up_result[offset]);
    }
  }
}

extern "C" __global__ void libmir_cuda_selected_nvfp4_tensor_core_linear_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const unsigned char* weight, const unsigned char* scales,
    const float* global_scales, __nv_bfloat16* output,
    unsigned int input_features, unsigned int output_features,
    unsigned int routes) {
  using namespace nvcuda;
  const unsigned int route = blockIdx.y;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int row_base = blockIdx.x * kColumns;
  if (route >= routes) return;
  const unsigned int expert = selected[route];
  input += route * input_features;
  output += route * output_features;
  __shared__ __nv_bfloat16 tile_a[kTile * kTile];
  __shared__ __nv_bfloat16 tile_b[kColumns * kTile];
  __shared__ float result[kColumns * kTile];
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::row_major> fragment_a;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                 __nv_bfloat16, wmma::col_major> fragment_b;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (unsigned int column = 0; column < input_features; column += kTile) {
    load_input_tile(input, tile_a, column, input_features);
    load_weight_tile(weight, scales, tile_b, expert, row_base, column,
                     input_features, output_features, global_scales[expert]);
    __syncthreads();
    wmma::load_matrix_sync(fragment_a, tile_a, kTile);
    wmma::load_matrix_sync(fragment_b, tile_b + warp * kTile * kTile, kTile);
    wmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    __syncthreads();
  }
  wmma::store_matrix_sync(result + warp * kTile * kTile, accumulator,
                          kTile, wmma::mem_row_major);
  __syncthreads();
  if (threadIdx.x < kColumns) {
    const unsigned int row = row_base + threadIdx.x;
    if (row < output_features) {
      const unsigned int offset =
          (threadIdx.x / kTile) * kTile * kTile + threadIdx.x % kTile;
      output[row] = __float2bfloat16_rn(result[offset]);
    }
  }
}
