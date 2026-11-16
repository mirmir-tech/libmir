// The GEMV for 2-byte aligned GGML blocks (`Block::staged`), composed after
// `ggml/dot.cuh`. A warp copies a 64-sub-block chunk of each of its rows
// into shared memory with 16-byte loads; lanes then decode two sub-blocks
// each from there. Rows must hold whole chunks, which keeps every chunk
// 16-byte aligned (whole rows of eight blocks of 210 or 34 bytes are).
namespace {

constexpr unsigned int kWarps = 4u;
constexpr unsigned int kChunkSubs = 64u;

template <typename Block>
__device__ constexpr unsigned int chunk_vectors() {
  return kChunkSubs / (Block::elements / kSub) * Block::bytes / 16u;
}

template <typename Block>
__device__ __forceinline__ bool stageable(unsigned int columns) {
  constexpr unsigned int chunk_blocks = kChunkSubs / (Block::elements / kSub);
  return (columns / Block::elements) % chunk_blocks == 0u;
}

template <unsigned int kRows, unsigned int kPerLane, unsigned int kVectors>
__device__ __forceinline__ void prefetch(
    uint4 (&next)[kRows][kPerLane], const uint4* const (&weights)[kRows], unsigned int offset,
    unsigned int lane) {
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
    for (unsigned int i = 0; i < kPerLane; ++i) {
      const unsigned int v = lane + 32u * i;
      if (v < kVectors) next[r][i] = __ldg(weights[r] + offset + v);
    }
  }
}

template <typename Block, unsigned int kTokens>
__device__ void ggml_gemv_staged(
    const unsigned int* quantized, const float4* scales, const unsigned char* weight,
    __nv_bfloat16* output, unsigned int tokens, unsigned int rows, unsigned int columns,
    unsigned int pass, unsigned int passes) {
  constexpr unsigned int kRows = rows_per_warp<kTokens>();
  constexpr unsigned int per_block = Block::elements / kSub;
  constexpr unsigned int vectors = chunk_vectors<Block>();
  constexpr unsigned int kPerLane = (vectors + 31u) / 32u;
  // One spare vector: decoders read up to three bytes past a block's end.
  __shared__ uint4 stage[kWarps][kRows][vectors + 1u];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int row = (blockIdx.x * kWarps + warp) * kRows;
  if (row >= rows) return;
  const unsigned int subs = columns / kSub;
  const unsigned int row_vectors = columns / Block::elements * Block::bytes / 16u;
  const uint4* weights[kRows];
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
    const unsigned int owned = row + r < rows ? row + r : row;
    weights[r] = reinterpret_cast<const uint4*>(weight) +
                 static_cast<unsigned long long>(owned) * row_vectors;
  }
  for (unsigned int first = pass * kTokens; first < tokens; first += passes * kTokens) {
    const unsigned int count = min(kTokens, tokens - first);
    float total[kRows][kTokens];
#pragma unroll
    for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
      for (unsigned int t = 0; t < kTokens; ++t) total[r][t] = 0.0f;
    }
    // Each lane carries its share of the next chunk in registers while the
    // warp computes on the current one in shared memory.
    uint4 next[kRows][kPerLane];
    prefetch<kRows, kPerLane, vectors>(next, weights, 0u, lane);
    for (unsigned int chunk = 0; chunk * vectors < row_vectors; ++chunk) {
      __syncwarp();
#pragma unroll
      for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
        for (unsigned int i = 0; i < kPerLane; ++i) {
          if (lane + 32u * i < vectors) stage[warp][r][lane + 32u * i] = next[r][i];
        }
      }
      __syncwarp();
      if ((chunk + 1u) * vectors < row_vectors) {
        prefetch<kRows, kPerLane, vectors>(next, weights, (chunk + 1u) * vectors, lane);
      }
#pragma unroll
      for (unsigned int k = 0; k < kChunkSubs / 32u; ++k) {
        const unsigned int local = lane + 32u * k;
        IntSub sub[kRows];
#pragma unroll
        for (unsigned int r = 0; r < kRows; ++r) {
          const unsigned char* bytes = reinterpret_cast<const unsigned char*>(stage[warp][r]);
          Block::decode(bytes + (local / per_block) * Block::bytes, local % per_block, sub[r]);
        }
        const unsigned int s = chunk * kChunkSubs + local;
#pragma unroll
        for (unsigned int t = 0; t < kTokens; ++t) {
          if (t < count) {
            const Activation x = load_activation(
                quantized, scales, static_cast<unsigned long long>(first + t) * subs + s);
#pragma unroll
            for (unsigned int r = 0; r < kRows; ++r) total[r][t] += dot_sub(sub[r], x);
          }
        }
      }
    }
    store_rows<kRows, kTokens>(total, output, lane, row, rows, first, count);
  }
}

}  // namespace
