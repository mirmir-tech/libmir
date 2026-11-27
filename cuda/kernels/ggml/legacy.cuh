// Legacy GGML blocks of 32 values, composed after `ggml/blocks.cuh`. Each
// block is one sub-block: values 0..15 come from the low nibbles of qs[16],
// 16..31 from the high nibbles; Q5 adds a fifth bit from qh (bit j for value
// j). Q4_0 and Q5_0 subtract 8 and 16; Q4_1 and Q5_1 add a minimum m.
namespace {

template <bool kFifthBit>
__device__ __forceinline__ void legacy_nibbles(
    const unsigned char* qs, unsigned int high_bits, unsigned int* q) {
  unsigned int words[4];
  load16(qs, words);
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    q[i] = words[i] & 0x0F0F0F0Fu;
    q[i + 4u] = (words[i] >> 4u) & 0x0F0F0F0Fu;
    if constexpr (kFifthBit) {
      q[i] |= spread4(high_bits >> (4u * i));
      q[i + 4u] |= spread4(high_bits >> (16u + 4u * i));
    }
  }
}

__device__ __forceinline__ unsigned int load_u32(const unsigned char* bytes) {
  return static_cast<unsigned int>(bytes[0]) | (static_cast<unsigned int>(bytes[1]) << 8u) |
         (static_cast<unsigned int>(bytes[2]) << 16u) | (static_cast<unsigned int>(bytes[3]) << 24u);
}

template <unsigned int kBytes, unsigned int kStaged>
struct LegacyTraits {
  static constexpr unsigned int elements = 32u;
  static constexpr unsigned int bytes = kBytes;
  static constexpr unsigned int staged_tokens = kStaged;
  static constexpr bool whole_scales = true;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 1u;
  static constexpr unsigned int width = 1u;
};

#define LIBMIR_GGML_LEGACY_UNIT                                                          \
  __device__ static void decode_unit(                                                   \
      const unsigned char* block, unsigned int, IntSub* sub, unsigned int* index) {     \
    decode(block, 0u, sub[0]);                                                          \
    index[0] = 0u;                                                                      \
  }

// Q4_0: half d, qs[16]; value = d * (q - 8).
struct Q4_0 : LegacyTraits<18u, 1u> {
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    legacy_nibbles<false>(block + 2u, 0u, sub.q);
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = __vsub4(sub.q[i], 0x08080808u);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
  LIBMIR_GGML_LEGACY_UNIT
};

// Q4_1: half d, half m, qs[16]; value = d * q + m.
struct Q4_1 : LegacyTraits<20u, 1u> {
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    legacy_nibbles<false>(block + 4u, 0u, sub.q);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = -ggml_half(block + 2u);
  }
  LIBMIR_GGML_LEGACY_UNIT
};

// Q5_0: half d, qh[4], qs[16]; value = d * (q - 16).
struct Q5_0 : LegacyTraits<22u, 1u> {
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    legacy_nibbles<true>(block + 6u, load_u32(block + 2u), sub.q);
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = __vsub4(sub.q[i], 0x10101010u);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
  LIBMIR_GGML_LEGACY_UNIT
};

// Q5_1: half d, half m, qh[4], qs[16]; value = d * q + m.
struct Q5_1 : LegacyTraits<24u, 4u> {
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    legacy_nibbles<true>(block + 8u, load_u32(block + 4u), sub.q);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = -ggml_half(block + 2u);
  }
  LIBMIR_GGML_LEGACY_UNIT
};

struct Q8_0 {
  static constexpr unsigned int staged_tokens = 8u;
  static constexpr bool whole_scales = true;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 1u;
  static constexpr unsigned int width = 1u;
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int, IntSub* sub, unsigned int* index) {
    decode(block, 0u, sub[0]);
    index[0] = 0u;
  }

  static constexpr unsigned int elements = 32u;
  static constexpr unsigned int bytes = 34u;
  // Q8_0: half d, int8 qs[32].
  __device__ static void decode(const unsigned char* block, unsigned int, IntSub& sub) {
    load32(block + 2u, sub.q);
    sub.scale[0] = sub.scale[1] = ggml_half(block);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

#undef LIBMIR_GGML_LEGACY_UNIT

}  // namespace
