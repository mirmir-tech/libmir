// GGML projections for decode batches, composed after `ggml/staged.cu`.
// Activations are quantized once to int8 per 32-value sub-block, with a
// scale and the exact sums of both halves (llama.cpp's Q8_1 sum and Q8_K
// bsums). Each lane decodes a weight sub-block to packed int8 and takes
// `dp4a` dot products with a pass of up to `Tokens` tokens; the narrowest
// pass that fits a batch keeps registers free.
namespace {


// One thread per token and sub-block: 32 BF16 values to int8 words, with the
// scale and the exact float sums of both halves beside them.
__device__ void ggml_quantize(
    const __nv_bfloat16* input, unsigned int* quantized, float4* scales, unsigned int tokens,
    unsigned int columns) {
  const unsigned int subs = columns / kSub;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(tokens) * subs) return;
  const uint4* vectors = reinterpret_cast<const uint4*>(input + index * kSub);
  float x[kSub];
#pragma unroll
  for (unsigned int v = 0; v < 4u; ++v) {
    const uint4 packed = vectors[v];
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) {
      const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
      x[v * 8u + i * 2u] = pair.x;
      x[v * 8u + i * 2u + 1u] = pair.y;
    }
  }
  float maximum = 0.0f;
  float sums[2] = {0.0f, 0.0f};
#pragma unroll
  for (unsigned int i = 0; i < kSub; ++i) {
    maximum = fmaxf(maximum, fabsf(x[i]));
    sums[i >> 4u] += x[i];
  }
  const float scale = maximum / 127.0f;
  const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
  unsigned int words[8];
#pragma unroll
  for (unsigned int w = 0; w < 8u; ++w) {
    unsigned int word = 0u;
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) {
      const int q = __float2int_rn(x[w * 4u + i] * inverse);
      word |= (static_cast<unsigned int>(q) & 0xFFu) << (i * 8u);
    }
    words[w] = word;
  }
  uint4* output = reinterpret_cast<uint4*>(quantized + index * 8u);
  output[0] = make_uint4(words[0], words[1], words[2], words[3]);
  output[1] = make_uint4(words[4], words[5], words[6], words[7]);
  scales[index] = make_float4(scale, sums[0], sums[1], 0.0f);
}

// Warps own `rows_per_warp` output rows; pass `pass` of `passes` takes every
// `passes`-th group of `kTokens` tokens. each lane decodes a unit of
// sub-blocks sharing bytes once per row and applies it to every token of the
// pass before the next.
template <typename Block, unsigned int kTokens>
__device__ void ggml_gemv(
    const unsigned int* quantized, const float4* scales, const unsigned char* weight,
    __nv_bfloat16* output, unsigned int tokens, unsigned int rows, unsigned int columns,
    unsigned int pass, unsigned int passes) {
  if constexpr (Block::staged) {
    if (stageable<Block>(columns)) {
      ggml_gemv_staged<Block, kTokens>(
          quantized, scales, weight, output, tokens, rows, columns, pass, passes);
      return;
    }
  }
  constexpr unsigned int kRows = rows_per_warp<kTokens>();
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int row = (blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u)) * kRows;
  if (row >= rows) return;
  const unsigned int subs = columns / kSub;
  const unsigned int blocks = columns / Block::elements;
  const unsigned int units = blocks * Block::units;
  const unsigned char* weights[kRows];
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
    // A missing last row repeats the first and is not written.
    const unsigned int owned = row + r < rows ? row + r : row;
    weights[r] = weight + static_cast<unsigned long long>(owned) * blocks * Block::bytes;
  }
  for (unsigned int first = pass * kTokens; first < tokens; first += passes * kTokens) {
    const unsigned int count = min(kTokens, tokens - first);
    float total[kRows][kTokens];
#pragma unroll
    for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
      for (unsigned int t = 0; t < kTokens; ++t) total[r][t] = 0.0f;
    }
#pragma unroll Block::unroll
    for (unsigned int u = lane; u < units; u += 32u) {
      IntSub sub[kRows][Block::width];
      unsigned int index[Block::width];
      const unsigned int block = u / Block::units;
#pragma unroll
      for (unsigned int r = 0; r < kRows; ++r) {
        Block::decode_unit(weights[r] + block * Block::bytes, u % Block::units, sub[r], index);
      }
      const unsigned int base = block * (Block::elements / kSub);
#pragma unroll
      for (unsigned int t = 0; t < kTokens; ++t) {
        if (t < count) {
          const unsigned long long token = static_cast<unsigned long long>(first + t) * subs;
#pragma unroll
          for (unsigned int w = 0; w < Block::width; ++w) {
            const Activation x = load_activation(quantized, scales, token + base + index[w]);
#pragma unroll
            for (unsigned int r = 0; r < kRows; ++r) total[r][t] += dot_sub(sub[r][w], x);
          }
        }
      }
    }
    store_rows<kRows, kTokens>(total, output, lane, row, rows, first, count);
  }
}

}  // namespace

extern "C" __global__ void libmir_cuda_ggml_quantize_q8(
    const __nv_bfloat16* input, unsigned int* quantized, float4* scales, unsigned int tokens,
    unsigned int columns) {
  ggml_quantize(input, quantized, scales, tokens, columns);
}

#define LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, TOKENS)                                     \
  extern "C" __global__ void libmir_cuda_ggml_gemv_bf16_##SUFFIX##_t##TOKENS(             \
      const unsigned int* quantized, const float4* scales, const unsigned char* weight,    \
      __nv_bfloat16* output, unsigned int tokens, unsigned int rows, unsigned int columns) { \
    ggml_gemv<BLOCK, TOKENS>(                                                              \
        quantized, scales, weight, output, tokens, rows, columns, blockIdx.y, gridDim.y);  \
  }
#define LIBMIR_CUDA_GGML_VECTORS(SUFFIX, BLOCK) \
  LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, 1)     \
  LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, 2)     \
  LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, 4)     \
  LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, 8)     \
  LIBMIR_CUDA_GGML_VECTOR(SUFFIX, BLOCK, 16)

LIBMIR_CUDA_GGML_VECTORS(q4_0, Q4_0)
LIBMIR_CUDA_GGML_VECTORS(q4_1, Q4_1)
LIBMIR_CUDA_GGML_VECTORS(q5_0, Q5_0)
LIBMIR_CUDA_GGML_VECTORS(q5_1, Q5_1)
LIBMIR_CUDA_GGML_VECTORS(q8_0, Q8_0)
LIBMIR_CUDA_GGML_VECTORS(q2_k, Q2_K)
LIBMIR_CUDA_GGML_VECTORS(q3_k, Q3_K)
LIBMIR_CUDA_GGML_VECTORS(q4_k, Q4_K)
LIBMIR_CUDA_GGML_VECTORS(q5_k, Q5_K)
LIBMIR_CUDA_GGML_VECTORS(q6_k, Q6_K)
LIBMIR_CUDA_GGML_VECTORS(iq4_nl, IQ4_NL)
LIBMIR_CUDA_GGML_VECTORS(iq4_xs, IQ4_XS)
LIBMIR_CUDA_GGML_VECTORS(iq2_xxs, IQ2_XXS)
LIBMIR_CUDA_GGML_VECTORS(iq2_xs, IQ2_XS)
LIBMIR_CUDA_GGML_VECTORS(iq2_s, IQ2_S)
LIBMIR_CUDA_GGML_VECTORS(iq3_xxs, IQ3_XXS)
LIBMIR_CUDA_GGML_VECTORS(iq3_s, IQ3_S)
LIBMIR_CUDA_GGML_VECTORS(iq1_s, IQ1_S)
LIBMIR_CUDA_GGML_VECTORS(iq1_m, IQ1_M)
LIBMIR_CUDA_GGML_VECTORS(mxfp4, MXFP4)
