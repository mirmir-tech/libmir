// GGML 4-bit table blocks, composed after `ggml/blocks.cuh`: IQ4 indices
// into llama.cpp's non-linear `kvalues_iq4nl`, MXFP4 e2m1 codes into the
// doubled `kvalues_mxfp4`. The values are int8, so the GEMV keeps its `dp4a`
// dot products; `__byte_perm` looks four up at once.
namespace {

// kvalues_iq4nl: -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89 113.
constexpr unsigned int kIq4Table[4] = {0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u};
// kvalues_mxfp4 (e2m1 doubled): 0 1 2 3 4 6 8 12 0 -1 -2 -3 -4 -6 -8 -12.
constexpr unsigned int kMxFp4Table[4] = {0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};

// Table values of four 4-bit indices (bits 0..15), as four int8 bytes.
__device__ __forceinline__ unsigned int iq4_lookup(
    unsigned int indices, const unsigned int (&table)[4]) {
  // `__byte_perm` reads three selector bits per byte (the fourth would mean
  // sign replication), so the table halves are picked with bit 3 afterwards.
  const unsigned int selector = indices & 0x7777u;
  const unsigned int low = __byte_perm(table[0], table[1], selector);
  const unsigned int high = __byte_perm(table[2], table[3], selector);
  const unsigned int upper = (indices >> 3u) & 0x1111u;
  const unsigned int mask = ((upper & 1u) | ((upper >> 4u & 1u) << 8u) |
                             ((upper >> 8u & 1u) << 16u) | ((upper >> 12u & 1u) << 24u)) *
                            0xFFu;
  return (low & ~mask) | (high & mask);
}

// The 32 values of 16 bytes: low nibbles first, then high nibbles.
__device__ __forceinline__ void iq4_values(
    const unsigned char* qs, unsigned int* q, const unsigned int (&table)[4] = kIq4Table) {
  unsigned int words[4];
  load16(qs, words);
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    const unsigned int w = words[i];
    const unsigned int low = (w & 0xFu) | ((w >> 4u) & 0xF0u) | ((w >> 8u) & 0xF00u) |
                             ((w >> 12u) & 0xF000u);
    const unsigned int high = ((w >> 4u) & 0xFu) | ((w >> 8u) & 0xF0u) | ((w >> 12u) & 0xF00u) |
                              ((w >> 16u) & 0xF000u);
    q[i] = iq4_lookup(low, table);
    q[i + 4u] = iq4_lookup(high, table);
  }
}

// IQ4_NL: half d, qs[16]; value = d * kvalues[q].
struct IQ4_NL {
  static constexpr unsigned int elements = 32u;
  static constexpr unsigned int bytes = 18u;
  static constexpr bool staged = true;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 1u;
  static constexpr unsigned int width = 1u;
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    iq4_values(block + 2u, sub.q);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int, IntSub* sub, unsigned int* index) {
    decode(block, 0u, sub[0]);
    index[0] = 0u;
  }
};

// IQ4_XS: half d, u16 scales_h, scales_l[4], qs[128]. Sub-block s has a
// 6-bit scale (low nibble of scales_l[s / 2], two bits of scales_h) offset
// by 32; value = d * (ls - 32) * kvalues[q].
struct IQ4_XS {
  static constexpr unsigned int elements = 256u;
  static constexpr unsigned int bytes = 136u;
  static constexpr bool staged = true;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 8u;
  static constexpr unsigned int width = 1u;
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    iq4_values(block + 8u + 16u * s, sub.q);
    const unsigned int high = static_cast<unsigned int>(block[2]) |
                              (static_cast<unsigned int>(block[3]) << 8u);
    const unsigned int scale =
        ((block[4u + s / 2u] >> (4u * (s & 1u))) & 0xFu) | (((high >> (2u * s)) & 3u) << 4u);
    sub.scale[0] = sub.scale[1] = ggml_half(block) * static_cast<float>(static_cast<int>(scale) - 32);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {
    decode(block, u, sub[0]);
    index[0] = u;
  }
};

// MXFP4: u8 e (E8M0), qs[16]; value = 2^(e - 128) * kvalues_mxfp4[q]
// (`GGML_E8M0_TO_FP32_HALF`, the half undoing the doubled table).
struct MXFP4 {
  static constexpr unsigned int elements = 32u;
  static constexpr unsigned int bytes = 17u;
  static constexpr bool staged = true;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 1u;
  static constexpr unsigned int width = 1u;
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    iq4_values(block + 1u, sub.q, kMxFp4Table);
    const unsigned int e = block[0];
    const unsigned int bits = e < 2u ? 0x00200000u << e : (e - 1u) << 23u;
    sub.scale[0] = sub.scale[1] = __uint_as_float(bits);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int, IntSub* sub, unsigned int* index) {
    decode(block, 0u, sub[0]);
    index[0] = 0u;
  }
};

}  // namespace
