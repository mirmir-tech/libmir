#include <cuda_bf16.h>

// Llama 4's router scales the selected expert's input, not its output: each
// row of `input` is multiplied by its one routing weight into `output`, and
// the weight becomes one, so the experts then combine with unit weights.
// One block per row; every thread reads the weight before any rewrites it.
extern "C" __global__ void libmir_cuda_router_input_scale_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* weights, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  const float weight = __bfloat162float(weights[row]);
  __syncthreads();
  if (threadIdx.x == 0) weights[row] = __float2bfloat16_rn(1.0f);
  const unsigned long long base = static_cast<unsigned long long>(row) * columns;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    output[base + column] =
        __float2bfloat16_rn(__bfloat162float(input[base + column]) * weight);
  }
}
