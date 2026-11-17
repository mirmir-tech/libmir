// Grouped GGML GEMM for routed prefill and short dense prefill chunks,
// composed after the block decoders and `ggml/mma.cuh`. Input rows are
// grouped by expert (`offsets[e]..offsets[e] + counts[e]`) and quantized like
// decode activations (int8 per 32 values, a scale and both halves' sums).
// A block computes a 64 x 64 output tile of one expert: it stages 128 input
// columns of 64 activation rows, decodes those columns of 64 weight rows to
// int8 with their scales and minima, takes `mma.sync` int8 products of each
// 32-value sub-block (16 for per-16 scales) and folds each into float sums:
// d_x * scale_w * dot - minimum_w * sum_x. llama.cpp's MMQ takes this route.
namespace {

// output[row * ld + column * stride + offset] for each expert row and
// weight row of the tile.
template <typename Block>
__device__ void ggml_mmq(
    const unsigned int* quantized, const float4* scales, const unsigned int* offsets,
    const unsigned int* counts, const unsigned int* tile_experts, const unsigned int* tile_rows,
    const unsigned int* tile_count, const unsigned char* weight, __nv_bfloat16* output,
    unsigned int rows, unsigned int columns, unsigned int ld, unsigned int stride,
    unsigned int offset) {
  __shared__ unsigned int a_words[kTileRows][kRowWords];
  __shared__ float4 a_scales[kTileRows][kStageSubs];
  __shared__ unsigned int b_words[kTileColumns][kRowWords];
  __shared__ float4 b_scales[kTileColumns][kStageSubs];
  // Whole-scale types fold each sub-block with one pair per row:
  // (scale, sum of the 32 values) and (scale, minimum).
  __shared__ float2 a_pairs[kTileRows][kStageSubs];
  __shared__ float2 b_pairs[kTileColumns][kStageSubs];
  constexpr bool kWhole = WholeScales<Block>::value;
  const unsigned int tile = blockIdx.y;
  if (tile >= *tile_count) return;
  const unsigned int expert = tile_experts[tile];
  const unsigned int first = tile_rows[tile];
  const unsigned int last = min(first + kTileRows, offsets[expert] + counts[expert]);
  const unsigned int n0 = blockIdx.x * kTileColumns;
  const unsigned int subs = columns / kSub;
  constexpr unsigned int per_block = Block::elements / kSub;
  const unsigned long long row_bytes =
      static_cast<unsigned long long>(columns / Block::elements) * Block::bytes;
  const unsigned char* bank =
      weight + static_cast<unsigned long long>(expert) * rows * row_bytes;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int group = lane >> 2u;
  const unsigned int quad = lane & 3u;
  // Warps take 32 x 32 quarters: two 16-row by four 8-column mma tiles.
  const unsigned int wm = (warp >> 1u) * 32u;
  const unsigned int wn = (warp & 1u) * 32u;
  float total[2][4][4] = {};
  for (unsigned int s0 = 0u; s0 < subs; s0 += kStageSubs) {
    // 64 rows x 4 sub-blocks of activations and of weights; 128 threads
    // take two of each.
    for (unsigned int item = threadIdx.x; item < kTileRows * kStageSubs; item += blockDim.x) {
      const unsigned int r = item / kStageSubs;
      const unsigned int s = item % kStageSubs;
      const unsigned int row = first + r;
      uint4 low = make_uint4(0u, 0u, 0u, 0u);
      uint4 high = low;
      float4 scale = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (row < last && s0 + s < subs) {
        const unsigned long long at = static_cast<unsigned long long>(row) * subs + s0 + s;
        const uint4* x = reinterpret_cast<const uint4*>(quantized + at * 8u);
        low = __ldg(x);
        high = __ldg(x + 1u);
        scale = __ldg(scales + at);
      }
      unsigned int* words = &a_words[r][s * 8u];
      words[0] = low.x; words[1] = low.y; words[2] = low.z; words[3] = low.w;
      words[4] = high.x; words[5] = high.y; words[6] = high.z; words[7] = high.w;
      if constexpr (kWhole) {
        a_pairs[r][s] = make_float2(scale.x, scale.y + scale.z);
      } else {
        a_scales[r][s] = scale;
      }
    }
    for (unsigned int item = threadIdx.x; item < kTileColumns * kStageSubs; item += blockDim.x) {
      const unsigned int n = item / kStageSubs;
      const unsigned int s = item % kStageSubs;
      IntSub sub;
      if (n0 + n < rows && s0 + s < subs) {
        const unsigned int at = s0 + s;
        Block::decode(
            bank + static_cast<unsigned long long>(n0 + n) * row_bytes +
                static_cast<unsigned long long>(at / per_block) * Block::bytes,
            at % per_block, sub);
      } else {
#pragma unroll
        for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = 0u;
        sub.scale[0] = sub.scale[1] = sub.minimum[0] = sub.minimum[1] = 0.0f;
      }
      unsigned int* words = &b_words[n][s * 8u];
#pragma unroll
      for (unsigned int i = 0; i < 8u; ++i) words[i] = sub.q[i];
      if constexpr (kWhole) {
        b_pairs[n][s] = make_float2(sub.scale[0], sub.minimum[0]);
      } else {
        b_scales[n][s] = make_float4(sub.scale[0], sub.scale[1], sub.minimum[0], sub.minimum[1]);
      }
    }
    __syncthreads();
    if constexpr (WholeScales<Block>::value) {
      // One int8 product of all 32 values per sub-block, folded once.
#pragma unroll
      for (unsigned int s = 0u; s < kStageSubs; ++s) {
        const unsigned int word = s * 8u + quad;
        int dot[2][4][4] = {};
#pragma unroll
        for (unsigned int m = 0u; m < 2u; ++m) {
          const unsigned int* low = a_words[wm + m * 16u + group];
          const unsigned int* high = a_words[wm + m * 16u + group + 8u];
#pragma unroll
          for (unsigned int n = 0u; n < 4u; ++n) {
            const unsigned int* b = b_words[wn + n * 8u + group];
            mma_s8_16x8x32(
                dot[m][n], low[word], high[word], low[word + 4u], high[word + 4u], b[word],
                b[word + 4u]);
          }
        }
        float w_scale[4][2];
        float w_minimum[4][2];
#pragma unroll
        for (unsigned int n = 0u; n < 4u; ++n) {
#pragma unroll
          for (unsigned int c = 0u; c < 2u; ++c) {
            const float2 w = b_pairs[wn + n * 8u + quad * 2u + c][s];
            w_scale[n][c] = w.x;
            w_minimum[n][c] = w.y;
          }
        }
#pragma unroll
        for (unsigned int m = 0u; m < 2u; ++m) {
#pragma unroll
          for (unsigned int half = 0u; half < 2u; ++half) {
            const float2 x = a_pairs[wm + m * 16u + group + half * 8u][s];
            const float sum = x.y;
#pragma unroll
            for (unsigned int n = 0u; n < 4u; ++n) {
#pragma unroll
              for (unsigned int c = 0u; c < 2u; ++c) {
                float& out = total[m][n][half * 2u + c];
                out = fmaf(x.x * w_scale[n][c], static_cast<float>(dot[m][n][half * 2u + c]),
                           fmaf(-w_minimum[n][c], sum, out));
              }
            }
          }
        }
      }
    } else {
#pragma unroll
      for (unsigned int s = 0u; s < kStageSubs; ++s) {
  #pragma unroll
        for (unsigned int h = 0u; h < 2u; ++h) {
          const unsigned int word = s * 8u + h * 4u + quad;
          int dot[2][4][4] = {};
  #pragma unroll
          for (unsigned int m = 0u; m < 2u; ++m) {
            const unsigned int a0 = a_words[wm + m * 16u + group][word];
            const unsigned int a1 = a_words[wm + m * 16u + group + 8u][word];
  #pragma unroll
            for (unsigned int n = 0u; n < 4u; ++n) {
              mma_s8_16x8x16(dot[m][n], a0, a1, b_words[wn + n * 8u + group][word]);
            }
          }
  #pragma unroll
          for (unsigned int m = 0u; m < 2u; ++m) {
  #pragma unroll
            for (unsigned int half = 0u; half < 2u; ++half) {
              const float4 x = a_scales[wm + m * 16u + group + half * 8u][s];
              const float sum = h == 0u ? x.y : x.z;
  #pragma unroll
              for (unsigned int n = 0u; n < 4u; ++n) {
  #pragma unroll
                for (unsigned int c = 0u; c < 2u; ++c) {
                  const float4 w = b_scales[wn + n * 8u + quad * 2u + c][s];
                  const float scale = h == 0u ? w.x : w.y;
                  const float minimum = h == 0u ? w.z : w.w;
                  float& out = total[m][n][half * 2u + c];
                  out = fmaf(x.x * scale, static_cast<float>(dot[m][n][half * 2u + c]),
                             fmaf(-minimum, sum, out));
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned int m = 0u; m < 2u; ++m) {
#pragma unroll
    for (unsigned int half = 0u; half < 2u; ++half) {
      const unsigned int row = first + wm + m * 16u + group + half * 8u;
      if (row >= last) continue;
#pragma unroll
      for (unsigned int n = 0u; n < 4u; ++n) {
#pragma unroll
        for (unsigned int c = 0u; c < 2u; ++c) {
          const unsigned int column = n0 + wn + n * 8u + quad * 2u + c;
          if (column < rows) {
            output[static_cast<unsigned long long>(row) * ld + column * stride + offset] =
                __float2bfloat16_rn(total[m][n][half * 2u + c]);
          }
        }
      }
    }
  }
}

}  // namespace

extern "C" __global__ void libmir_cuda_ggml_mmq_tiles(
    const unsigned int* offsets, const unsigned int* counts, unsigned int* tile_experts,
    unsigned int* tile_rows, unsigned int* tile_count, unsigned int experts,
    unsigned int max_tiles) {
  ggml_mmq_tiles(offsets, counts, tile_experts, tile_rows, tile_count, experts, max_tiles);
}

#define LIBMIR_CUDA_GGML_MMQ(SUFFIX, BLOCK)                                                 \
  extern "C" __global__ void __launch_bounds__(128) libmir_cuda_ggml_mmq_bf16_##SUFFIX(    \
      const unsigned int* quantized, const float4* scales, const unsigned int* offsets,     \
      const unsigned int* counts, const unsigned int* tile_experts,                         \
      const unsigned int* tile_rows, const unsigned int* tile_count,                        \
      const unsigned char* weight, __nv_bfloat16* output, unsigned int rows,                \
      unsigned int columns, unsigned int ld, unsigned int stride, unsigned int offset) {    \
    ggml_mmq<BLOCK>(quantized, scales, offsets, counts, tile_experts, tile_rows, tile_count, \
                    weight, output, rows, columns, ld, stride, offset);                     \
  }

LIBMIR_CUDA_GGML_MMQ(q4_0, Q4_0)
LIBMIR_CUDA_GGML_MMQ(q4_1, Q4_1)
LIBMIR_CUDA_GGML_MMQ(q5_0, Q5_0)
LIBMIR_CUDA_GGML_MMQ(q5_1, Q5_1)
LIBMIR_CUDA_GGML_MMQ(q8_0, Q8_0)
LIBMIR_CUDA_GGML_MMQ(q2_k, Q2_K)
LIBMIR_CUDA_GGML_MMQ(q3_k, Q3_K)
LIBMIR_CUDA_GGML_MMQ(q4_k, Q4_K)
LIBMIR_CUDA_GGML_MMQ(q5_k, Q5_K)
LIBMIR_CUDA_GGML_MMQ(q6_k, Q6_K)
LIBMIR_CUDA_GGML_MMQ(iq4_nl, IQ4_NL)
LIBMIR_CUDA_GGML_MMQ(iq4_xs, IQ4_XS)
LIBMIR_CUDA_GGML_MMQ(iq2_xxs, IQ2_XXS)
LIBMIR_CUDA_GGML_MMQ(iq2_xs, IQ2_XS)
LIBMIR_CUDA_GGML_MMQ(iq2_s, IQ2_S)
LIBMIR_CUDA_GGML_MMQ(iq3_xxs, IQ3_XXS)
LIBMIR_CUDA_GGML_MMQ(iq3_s, IQ3_S)
LIBMIR_CUDA_GGML_MMQ(iq1_s, IQ1_S)
LIBMIR_CUDA_GGML_MMQ(iq1_m, IQ1_M)
LIBMIR_CUDA_GGML_MMQ(mxfp4, MXFP4)
