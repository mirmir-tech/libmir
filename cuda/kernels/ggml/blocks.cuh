// Shared pieces of the GGML block decoders, composed before
// `ggml/legacy.cuh` and `ggml/k_quants.cuh`. Dequantization follows llama.cpp
// v1.0.0 `ggml-quants.c` (`dequantize_row_*`) value for value. Every kernel
// works on 32-value sub-blocks, decoded either to floats or to packed int8
// with per-16 scales and a minimum for `dp4a`. Blocks that are only 2-byte
// aligned have their bytes funnel-shifted out of aligned words.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr unsigned int kSub = 32u;

__device__ __forceinline__ float ggml_half(const unsigned char* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8u);
  return __half2float(__ushort_as_half(bits));
}

// 32 bytes from any byte address, global or shared, into eight
// little-endian words.
__device__ __forceinline__ void load32(const unsigned char* bytes, unsigned int* words) {
  const unsigned long long address = reinterpret_cast<unsigned long long>(bytes);
  const unsigned int* aligned = reinterpret_cast<const unsigned int*>(address & ~3ull);
  const unsigned int shift = static_cast<unsigned int>(address & 3ull) * 8u;
  unsigned int previous = aligned[0];
#pragma unroll
  for (unsigned int i = 0; i < 8u; ++i) {
    const unsigned int next = shift == 0u ? 0u : aligned[i + 1u];
    words[i] = shift == 0u ? aligned[i] : __funnelshift_r(previous, next, shift);
    previous = next;
  }
}

// 16 bytes from any byte address, global or shared, into four words.
__device__ __forceinline__ void load16(const unsigned char* bytes, unsigned int* words) {
  const unsigned long long address = reinterpret_cast<unsigned long long>(bytes);
  const unsigned int* aligned = reinterpret_cast<const unsigned int*>(address & ~3ull);
  const unsigned int shift = static_cast<unsigned int>(address & 3ull) * 8u;
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    words[i] = shift == 0u ? aligned[i] : __funnelshift_r(aligned[i], aligned[i + 1u], shift);
  }
}

// Bits 0..3 of `bits` as bit 4 of bytes 0..3: GGML's fifth value bit.
__device__ __forceinline__ unsigned int spread4(unsigned int bits) {
  return ((bits & 0xFu) * 0x00204081u & 0x01010101u) << 4u;
}

// A sub-block as packed signed int8 values: value i is byte i % 4 of
// q[i / 4]; it weighs `scale[i / 16] * q - minimum[i / 16]` before the
// activation.
struct IntSub {
  unsigned int q[8];
  float scale[2];
  float minimum[2];
};

__device__ __forceinline__ void unpack(const IntSub& sub, float* w) {
#pragma unroll
  for (unsigned int i = 0; i < kSub; ++i) {
    const int value = static_cast<signed char>((sub.q[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu);
    w[i] = sub.scale[i >> 4u] * static_cast<float>(value) - sub.minimum[i >> 4u];
  }
}

__device__ __forceinline__ unsigned int byte_at(const unsigned int* words, unsigned int index) {
  return (words[index >> 2u] >> ((index & 3u) * 8u)) & 0xFFu;
}

// Q4_K's 6-bit scale and minimum of sub-block s from scales[12].
__device__ __forceinline__ void scale_min_k4(
    const unsigned int* scales, unsigned int s, unsigned int& scale, unsigned int& minimum) {
  if (s < 4u) {
    scale = byte_at(scales, s) & 63u;
    minimum = byte_at(scales, s + 4u) & 63u;
  } else {
    scale = (byte_at(scales, s + 4u) & 0xFu) | ((byte_at(scales, s - 4u) >> 6u) << 4u);
    minimum = (byte_at(scales, s + 4u) >> 4u) | ((byte_at(scales, s) >> 6u) << 4u);
  }
}

// Block traits: the GEMV reads a block in units of `width` sub-blocks that
// share bytes, `units` per block; `index` receives each sub-block's place in
// the block. `staged`: blocks not 16-byte aligned, which the GEMV first
// copies, a row chunk at a time, into shared memory with 16-byte loads.
template <typename Block>
__device__ __forceinline__ void decode_sub(
    const unsigned char* weight, unsigned long long row, unsigned int s, unsigned int columns,
    IntSub& sub) {
  constexpr unsigned int per_block = Block::elements / kSub;
  const unsigned long long row_bytes =
      static_cast<unsigned long long>(columns / Block::elements) * Block::bytes;
  const unsigned char* block =
      weight + row * row_bytes + static_cast<unsigned long long>(s / per_block) * Block::bytes;
  Block::decode(block, s % per_block, sub);
}

}  // namespace
