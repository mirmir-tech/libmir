// Output scales of GGML projections (llama.cpp `.scale`, `build_lora_mm` and
// the `*_s` factors of `build_ffn`): an F32 factor per matrix, or one per
// expert of a bank, multiplied into the BF16 products after the matmul.
#include <cuda_bf16.h>

// `data[i] *= scale[0]` for `i < elements`.
extern "C" __global__ void libmir_cuda_ggml_scale_bf16(
    __nv_bfloat16* data, const float* scale, unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  data[index] = __float2bfloat16_rn(__bfloat162float(data[index]) * scale[0]);
}

// Row `r` of `rows` selection rows, columns `j < columns` at `r * ld + j *
// stride + offset`, times the scale of its expert `experts[r]`; rows of no
// expert (`>= expert_count`) stay as they are.
extern "C" __global__ void libmir_cuda_ggml_scale_expert_rows_bf16(
    __nv_bfloat16* data, const float* scales, const unsigned int* experts,
    unsigned int rows, unsigned int columns, unsigned int ld, unsigned int stride,
    unsigned int offset, unsigned int expert_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(rows) * columns) return;
  const unsigned int row = index / columns;
  const unsigned int column = index % columns;
  const unsigned int expert = experts[row];
  if (expert >= expert_count) return;
  const unsigned long long element =
      static_cast<unsigned long long>(row) * ld + static_cast<unsigned long long>(column) * stride +
      offset;
  data[element] = __float2bfloat16_rn(__bfloat162float(data[element]) * scales[expert]);
}

// `data[r * columns + c] += bias[c]`: a projection's output bias after its
// matmul (Qwen2, GLM-4 query/key/value).
extern "C" __global__ void libmir_cuda_ggml_bias_bf16(
    __nv_bfloat16* data, const __nv_bfloat16* bias, unsigned int rows, unsigned int columns) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(rows) * columns) return;
  const unsigned int column = index % columns;
  data[index] =
      __float2bfloat16_rn(__bfloat162float(data[index]) + __bfloat162float(bias[column]));
}
