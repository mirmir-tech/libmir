// The GEMV for 2-byte aligned GGML blocks (`Block::staged`), composed after
// `ggml/dot.cuh`. A warp copies a 64-sub-block chunk of each of its rows
// into shared memory with aligned 16-byte loads; lanes then decode two
// sub-blocks each from there. A chunk is a whole number of 16-byte vectors,
// so a row's chunks share its misalignment, which the decoders' byte loads
// absorb; the last chunk of a row may be partial. Rows align by their
// absolute address: a routed expert's matrix starts wherever its bank's
// earlier matrices end, which a block of 18, 34 or 110 bytes leaves only
// 2-byte aligned. A row's last vector may run up to 15 bytes past it, past
// the matrix's last row into the allocation's `GgmlBlocks::TAIL_BYTES`.
namespace {

constexpr unsigned int kWarps = 4u;
constexpr unsigned int kChunkSubs = 64u;

template <typename Block>
__device__ constexpr unsigned int chunk_vectors() {
  return kChunkSubs / (Block::elements / kSub) * Block::bytes / 16u;
}

// Vectors [offset, offset + kVectors) of each row's aligned window, short of
// the vector holding the row's last byte (`ends`).
template <unsigned int kRows, unsigned int kPerLane, unsigned int kVectors>
__device__ __forceinline__ void prefetch(
    uint4 (&next)[kRows][kPerLane], const uint4* const (&weights)[kRows],
    const unsigned int (&ends)[kRows], unsigned int offset, unsigned int lane) {
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
    for (unsigned int i = 0; i < kPerLane; ++i) {
      const unsigned int v = lane + 32u * i;
      if (v < kVectors && offset + v < ends[r]) next[r][i] = __ldg(weights[r] + offset + v);
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
  // A chunk's vectors, plus one for a misaligned row start.
  constexpr unsigned int vectors = kChunkSubs / per_block * Block::bytes / 16u;
  constexpr unsigned int window = vectors + 1u;
  constexpr unsigned int kPerLane = (window + 31u) / 32u;
  // One spare vector: decoders read up to three bytes past a block's end.
  __shared__ uint4 stage[kWarps][kRows][window + 1u];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int row = (blockIdx.x * kWarps + warp) * kRows;
  if (row >= rows) return;
  const unsigned int subs = columns / kSub;
  const unsigned long long row_bytes =
      static_cast<unsigned long long>(columns / Block::elements) * Block::bytes;
  const uint4* weights[kRows];
  unsigned int shifts[kRows];
  unsigned int ends[kRows];
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
    const unsigned int owned = row + r < rows ? row + r : row;
    const unsigned long long start = reinterpret_cast<unsigned long long>(weight) +
                                     static_cast<unsigned long long>(owned) * row_bytes;
    weights[r] = reinterpret_cast<const uint4*>(start & ~15ull);
    shifts[r] = static_cast<unsigned int>(start & 15ull);
    ends[r] = static_cast<unsigned int>((start + row_bytes + 15ull) / 16ull - (start >> 4u));
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
    prefetch<kRows, kPerLane, window>(next, weights, ends, 0u, lane);
    for (unsigned int chunk = 0; chunk * kChunkSubs < subs; ++chunk) {
      __syncwarp();
#pragma unroll
      for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
        for (unsigned int i = 0; i < kPerLane; ++i) {
          if (lane + 32u * i < window) stage[warp][r][lane + 32u * i] = next[r][i];
        }
      }
      __syncwarp();
      if ((chunk + 1u) * kChunkSubs < subs) {
        prefetch<kRows, kPerLane, window>(next, weights, ends, (chunk + 1u) * vectors, lane);
      }
#pragma unroll
      for (unsigned int k = 0; k < kChunkSubs / 32u; ++k) {
        const unsigned int local = lane + 32u * k;
        const unsigned int s = chunk * kChunkSubs + local;
        if (s < subs) {
          IntSub sub[kRows];
#pragma unroll
          for (unsigned int r = 0; r < kRows; ++r) {
            const unsigned char* bytes =
                reinterpret_cast<const unsigned char*>(stage[warp][r]) + shifts[r];
            Block::decode(bytes + (local / per_block) * Block::bytes, local % per_block, sub[r]);
          }
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
    }
    store_rows<kRows, kTokens>(total, output, lane, row, rows, first, count);
  }
}

}  // namespace
