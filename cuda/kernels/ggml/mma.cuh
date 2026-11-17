// Pieces of the grouped GGML GEMM (`ggml/mmq.cu`): its tile layout, the
// scale traits of block types, and int8 `mma.sync` wrappers.
namespace {

constexpr unsigned int kTileRows = 64u;
constexpr unsigned int kTileColumns = 64u;

// The expert and first row of each 64-row tile, from the expert offsets.
__device__ void ggml_mmq_tiles(
    const unsigned int* offsets, const unsigned int* counts, unsigned int* tile_experts,
    unsigned int* tile_rows,
    unsigned int* tile_count, unsigned int experts, unsigned int max_tiles) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  unsigned int tiles = 0u;
  for (unsigned int expert = 0u; expert < experts; ++expert) {
    const unsigned int end = offsets[expert] + counts[expert];
    for (unsigned int row = offsets[expert]; row < end && tiles < max_tiles;
         row += kTileRows) {
      tile_experts[tiles] = expert;
      tile_rows[tiles] = row;
      ++tiles;
    }
  }
  *tile_count = tiles;
}

// Whether a block type keeps one scale and minimum per 32-value sub-block
// (`Block::whole_scales`); others scale each 16 values apart.
template <typename Block, typename = void>
struct WholeScales {
  static constexpr bool value = false;
};
template <typename Block>
struct WholeScales<Block, decltype(void(Block::whole_scales))> {
  static constexpr bool value = Block::whole_scales;
};

__device__ __forceinline__ void mma_s8_16x8x32(
    int (&c)[4], unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8_16x8x16(
    int (&c)[4], unsigned int a0, unsigned int a1, unsigned int b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

}  // namespace
