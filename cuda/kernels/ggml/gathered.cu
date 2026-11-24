// Routed-expert projections over stacked GGML expert banks, composed after
// `ggml/vector.cu`. A bank holds `experts` matrices of `rows` rows back to
// back. Grid row y is one (token, expert) selection: it reads the selected
// expert's matrix and the activations of token `selection / group`, so gate
// and up (group = top-k) share each token's quantized input while down
// (group 1) takes each selection's own intermediate.
namespace {

template <typename Block>
__device__ void ggml_gemv_gathered(
    const unsigned int* quantized, const float4* scales, const unsigned char* weight,
    const unsigned int* selected, __nv_bfloat16* output, unsigned int selections,
    unsigned int group, unsigned int rows, unsigned int columns, unsigned int expert_rows,
    unsigned int first_row) {
  const unsigned int selection = blockIdx.y;
  if (selection >= selections) return;
  const unsigned long long subs = columns / kSub;
  const unsigned long long token = selection / group;
  // The expert's matrix holds `expert_rows` rows, this projection's from
  // `first_row` (gate then up halves of a fused bank).
  const unsigned long long row_bytes = (columns / Block::elements) * Block::bytes;
  const unsigned char* matrix =
      weight + (static_cast<unsigned long long>(selected[selection]) * expert_rows + first_row) *
                   row_bytes;
  ggml_gemv<Block, 1u>(
      quantized + token * subs * 8u, scales + token * subs, matrix,
      output + static_cast<unsigned long long>(selection) * rows, 1u, rows, columns, 0u, 1u);
}

}  // namespace

// The routing-weighted sum of each token's selected expert outputs, in
// float; one thread per token and hidden column.
extern "C" __global__ void libmir_cuda_ggml_combine_bf16(
    const __nv_bfloat16* partial, const __nv_bfloat16* routing, __nv_bfloat16* output,
    unsigned int tokens, unsigned int top_k, unsigned int hidden) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(tokens) * hidden) return;
  const unsigned long long token = index / hidden;
  const unsigned long long column = index % hidden;
  float total = 0.0f;
  for (unsigned int k = 0; k < top_k; ++k) {
    const unsigned long long selection = token * top_k + k;
    total = fmaf(
        __bfloat162float(routing[selection]), __bfloat162float(partial[selection * hidden + column]),
        total);
  }
  output[index] = __float2bfloat16_rn(total);
}

#define LIBMIR_CUDA_GGML_GATHERED(SUFFIX, BLOCK)                                               \
  extern "C" __global__ void libmir_cuda_ggml_gemv_gathered_bf16_##SUFFIX(                    \
      const unsigned int* quantized, const float4* scales, const unsigned char* weight,        \
      const unsigned int* selected, __nv_bfloat16* output, unsigned int selections,            \
      unsigned int group, unsigned int rows, unsigned int columns, unsigned int expert_rows,   \
      unsigned int first_row) {                                                                \
    ggml_gemv_gathered<BLOCK>(quantized, scales, weight, selected, output, selections, group,  \
                              rows, columns, expert_rows, first_row);                          \
  }

LIBMIR_CUDA_GGML_GATHERED(q4_0, Q4_0)
LIBMIR_CUDA_GGML_GATHERED(q4_1, Q4_1)
LIBMIR_CUDA_GGML_GATHERED(q5_0, Q5_0)
LIBMIR_CUDA_GGML_GATHERED(q5_1, Q5_1)
LIBMIR_CUDA_GGML_GATHERED(q8_0, Q8_0)
LIBMIR_CUDA_GGML_GATHERED(q2_k, Q2_K)
LIBMIR_CUDA_GGML_GATHERED(q3_k, Q3_K)
LIBMIR_CUDA_GGML_GATHERED(q4_k, Q4_K)
LIBMIR_CUDA_GGML_GATHERED(q5_k, Q5_K)
LIBMIR_CUDA_GGML_GATHERED(q6_k, Q6_K)
LIBMIR_CUDA_GGML_GATHERED(iq4_nl, IQ4_NL)
LIBMIR_CUDA_GGML_GATHERED(iq4_xs, IQ4_XS)
LIBMIR_CUDA_GGML_GATHERED(iq2_xxs, IQ2_XXS)
LIBMIR_CUDA_GGML_GATHERED(iq2_xs, IQ2_XS)
LIBMIR_CUDA_GGML_GATHERED(iq2_s, IQ2_S)
LIBMIR_CUDA_GGML_GATHERED(iq3_xxs, IQ3_XXS)
LIBMIR_CUDA_GGML_GATHERED(iq3_s, IQ3_S)
LIBMIR_CUDA_GGML_GATHERED(iq1_s, IQ1_S)
LIBMIR_CUDA_GGML_GATHERED(iq1_m, IQ1_M)
LIBMIR_CUDA_GGML_GATHERED(mxfp4, MXFP4)
LIBMIR_CUDA_GGML_GATHERED(q1_0, Q1_0)
LIBMIR_CUDA_GGML_GATHERED(q2_0, Q2_0)
LIBMIR_CUDA_GGML_GATHERED(tq1_0, TQ1_0)
LIBMIR_CUDA_GGML_GATHERED(tq2_0, TQ2_0)
LIBMIR_CUDA_GGML_GATHERED(nvfp4, NVFP4)
