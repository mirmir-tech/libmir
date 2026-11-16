// GGML matrices, composed after the block decoders: prefill dequantizes a
// projection to BF16 for the dense GEMM; embeddings decode the selected
// vocabulary rows.
namespace {

__device__ __forceinline__ void store_sub(__nv_bfloat16* output, const float* w, float scale) {
  uint4* vectors = reinterpret_cast<uint4*>(output);
#pragma unroll
  for (unsigned int v = 0; v < 4u; ++v) {
    unsigned int words[4];
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) {
      const __nv_bfloat162 pair = __floats2bfloat162_rn(
          w[v * 8u + i * 2u] * scale, w[v * 8u + i * 2u + 1u] * scale);
      words[i] = *reinterpret_cast<const unsigned int*>(&pair);
    }
    vectors[v] = make_uint4(words[0], words[1], words[2], words[3]);
  }
}

// The whole matrix as row-major BF16, one sub-block per thread; source row r
// lands in output row `r * stride + offset`, so two matrices can interleave.
template <typename Block>
__device__ void ggml_dequantize(
    const unsigned char* weight, __nv_bfloat16* output, unsigned int rows, unsigned int columns,
    unsigned int stride, unsigned int offset) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned int subs = columns / kSub;
  if (index >= static_cast<unsigned long long>(rows) * subs) return;
  const unsigned long long row = index / subs;
  const unsigned int s = static_cast<unsigned int>(index % subs);
  IntSub sub;
  float w[kSub];
  decode_sub<Block>(weight, row, s, columns, sub);
  unpack(sub, w);
  store_sub(output + (row * stride + offset) * columns + s * kSub, w, 1.0f);
}

// Selected vocabulary rows, scaled for embeddings with a multiplier.
template <typename Block>
__device__ void ggml_embedding(
    const unsigned char* weight, const unsigned int* selected, __nv_bfloat16* output,
    unsigned int selected_start, unsigned int tokens, unsigned int vocab,
    unsigned int hidden, float output_scale) {
  const unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int token = blockIdx.y;
  if (s >= hidden / kSub || token >= tokens) return;
  const unsigned int row = selected[selected_start + token];
  if (row >= vocab) return;
  IntSub sub;
  float w[kSub];
  decode_sub<Block>(weight, row, s, hidden, sub);
  unpack(sub, w);
  store_sub(output + static_cast<unsigned long long>(token) * hidden + s * kSub, w, output_scale);
}

}  // namespace

#define LIBMIR_CUDA_GGML_MATRIX(SUFFIX, BLOCK)                                             \
  extern "C" __global__ void libmir_cuda_ggml_dequantize_bf16_##SUFFIX(                   \
      const unsigned char* weight, __nv_bfloat16* output, unsigned int rows,               \
      unsigned int columns, unsigned int stride, unsigned int offset) {                    \
    ggml_dequantize<BLOCK>(weight, output, rows, columns, stride, offset);                 \
  }                                                                                        \
  extern "C" __global__ void libmir_cuda_ggml_embedding_bf16_##SUFFIX(                    \
      const unsigned char* weight, const unsigned int* selected, __nv_bfloat16* output,    \
      unsigned int selected_start, unsigned int tokens, unsigned int vocab,                \
      unsigned int hidden, float output_scale) {                                           \
    ggml_embedding<BLOCK>(weight, selected, output, selected_start, tokens, vocab, hidden, \
                          output_scale);                                                   \
  }

LIBMIR_CUDA_GGML_MATRIX(q4_0, Q4_0)
LIBMIR_CUDA_GGML_MATRIX(q4_1, Q4_1)
LIBMIR_CUDA_GGML_MATRIX(q5_0, Q5_0)
LIBMIR_CUDA_GGML_MATRIX(q5_1, Q5_1)
LIBMIR_CUDA_GGML_MATRIX(q8_0, Q8_0)
LIBMIR_CUDA_GGML_MATRIX(q2_k, Q2_K)
LIBMIR_CUDA_GGML_MATRIX(q3_k, Q3_K)
LIBMIR_CUDA_GGML_MATRIX(q4_k, Q4_K)
LIBMIR_CUDA_GGML_MATRIX(q5_k, Q5_K)
LIBMIR_CUDA_GGML_MATRIX(q6_k, Q6_K)
LIBMIR_CUDA_GGML_MATRIX(iq4_nl, IQ4_NL)
LIBMIR_CUDA_GGML_MATRIX(iq4_xs, IQ4_XS)
LIBMIR_CUDA_GGML_MATRIX(iq2_xxs, IQ2_XXS)
LIBMIR_CUDA_GGML_MATRIX(iq2_xs, IQ2_XS)
LIBMIR_CUDA_GGML_MATRIX(iq2_s, IQ2_S)
LIBMIR_CUDA_GGML_MATRIX(iq3_xxs, IQ3_XXS)
LIBMIR_CUDA_GGML_MATRIX(iq3_s, IQ3_S)
LIBMIR_CUDA_GGML_MATRIX(iq1_s, IQ1_S)
LIBMIR_CUDA_GGML_MATRIX(iq1_m, IQ1_M)
LIBMIR_CUDA_GGML_MATRIX(mxfp4, MXFP4)
