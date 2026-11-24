#include <cuda_bf16.h>

// Gemma 4's per-layer inputs (Hugging Face `Gemma4TextModel`'s
// `project_per_layer_inputs`): for token row `t` and layer `l`,
//   output[l][t] = (rms_norm(projected[t][l] * projection_scale) * weight
//                   + embedded[t][l]) * input_scale,
// where `projected` and `embedded` are `[rows, layers * width]` and each
// layer's slice is `width` wide. The output is layer-major,
// `[layers, rows, width]`, so a layer reads its rows contiguously. One block
// per (layer, row).
namespace {

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warps[32];
  __shared__ float total;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x >> 5u] = value;
  __syncthreads();
  if (threadIdx.x < 32u) {
    value = threadIdx.x < (blockDim.x + 31u) / 32u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (threadIdx.x == 0u) total = value;
  }
  __syncthreads();
  return total;
}

}  // namespace

extern "C" __global__ void libmir_cuda_per_layer_input_bf16(
    const __nv_bfloat16* projected, const __nv_bfloat16* embedded,
    const __nv_bfloat16* weight, __nv_bfloat16* output, unsigned int rows,
    unsigned int layers, unsigned int width, float projection_scale,
    float input_scale, float epsilon) {
  const unsigned int layer = blockIdx.x;
  const unsigned int row = blockIdx.y;
  if (layer >= layers || row >= rows) return;
  const unsigned long long source =
      (static_cast<unsigned long long>(row) * layers + layer) * width;
  const unsigned long long target =
      (static_cast<unsigned long long>(layer) * rows + row) * width;
  float squares = 0.0f;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(projected[source + column]) * projection_scale;
    squares = fmaf(value, value, squares);
  }
  const float inverse = rsqrtf(block_sum(squares) / static_cast<float>(width) + epsilon);
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(projected[source + column]) * projection_scale;
    const float normalized = value * inverse * __bfloat162float(weight[column]);
    output[target + column] = __float2bfloat16_rn(
        (normalized + __bfloat162float(embedded[source + column])) * input_scale);
  }
}
