// GGML one- and two-bit, ternary and NVFP4 blocks, composed after
// `ggml/blocks.cuh` and `ggml/iq4.cuh` (for the E2M1 table): llama.cpp
// `dequantize_row_q1_0`, `_q2_0`, `_tq1_0`, `_tq2_0` and `_nvfp4`. Their
// values are small integers, so the GEMV keeps its `dp4a` dot products. The
// blocks are at most 2-byte aligned and are read a byte or a word at a time.
namespace {

// One sub-block per unit, read from shared memory like the staged K-quants.
#define LIBMIR_GGML_LOW_BIT(ELEMENTS, BYTES)                                            \
  static constexpr unsigned int elements = ELEMENTS;                                   \
  static constexpr unsigned int bytes = BYTES;                                         \
  static constexpr unsigned int staged_tokens = 1u;                                    \
  static constexpr unsigned int unroll = 1u;                                           \
  static constexpr unsigned int units = ELEMENTS / kSub;                               \
  static constexpr unsigned int width = 1u;                                            \
  __device__ static void decode_unit(                                                  \
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {  \
    decode(block, u, sub[0]);                                                          \
    index[0] = u;                                                                      \
  }

__device__ __forceinline__ void low_bit_scale(IntSub& sub, float d) {
  sub.scale[0] = sub.scale[1] = d;
  sub.minimum[0] = sub.minimum[1] = 0.0f;
}

// Q1_0: half d, qs[16] of 128 sign bits; a set bit is d, a clear bit -d.
struct Q1_0 {
  LIBMIR_GGML_LOW_BIT(128u, 18u)
  static constexpr bool whole_scales = true;
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned char* qs = block + 2u + 4u * s;
    const unsigned int bits = static_cast<unsigned int>(qs[0]) |
                              (static_cast<unsigned int>(qs[1]) << 8u) |
                              (static_cast<unsigned int>(qs[2]) << 16u) |
                              (static_cast<unsigned int>(qs[3]) << 24u);
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      const unsigned int set = ((bits >> (4u * i)) & 0xFu) * 0x00204081u & 0x01010101u;
      sub.q[i] = __vsub4(set << 1u, 0x01010101u);
    }
    low_bit_scale(sub, ggml_half(block));
  }
};

// Q2_0: half d, qs[16]; value j is bits 2 (j % 4) of qs[j / 4], minus 1.
struct Q2_0 {
  LIBMIR_GGML_LOW_BIT(64u, 18u)
  static constexpr bool whole_scales = true;
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned char* qs = block + 2u + 8u * s;
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      const unsigned int b = qs[i];
      const unsigned int value =
          (b & 3u) | (((b >> 2u) & 3u) << 8u) | (((b >> 4u) & 3u) << 16u) | ((b >> 6u) << 24u);
      sub.q[i] = __vsub4(value, 0x01010101u);
    }
    low_bit_scale(sub, ggml_half(block));
  }
};

// TQ2_0: qs[64], half d. Sub-block s is shift 2 (s % 4) of
// qs[32 (s / 4) ..][0..32], minus 1.
struct TQ2_0 {
  LIBMIR_GGML_LOW_BIT(256u, 66u)
  static constexpr bool whole_scales = true;
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    unsigned int q[8];
    load32(block + (s >> 2u) * 32u, q);
    const unsigned int shift = 2u * (s & 3u);
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = __vsub4((q[i] >> shift) & 0x03030303u, 0x01010101u);
    low_bit_scale(sub, ggml_half(block + 64u));
  }
};

// TQ1_0: qs[48] of five base-3 digits, qh[4] of four, half d. Values
// 0..159 are digit s of qs[0..32]; 160..239 digit (p - 160) / 16 of
// qs[32..48]; 240..255 digit (p - 240) / 4 of qh. Digit n of byte b is
// ((b 3^n mod 256) 3) >> 8; the value is that digit minus 1.
struct TQ1_0 {
  LIBMIR_GGML_LOW_BIT(256u, 54u)
  static constexpr bool whole_scales = true;
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      unsigned int word = 0u;
#pragma unroll
      for (unsigned int b = 0; b < 4u; ++b) {
        const unsigned int p = 4u * i + b;
        unsigned int index = p;
        unsigned int digit = s;
        if (s >= 5u && (s < 7u || p < 16u)) {
          index = 32u + (p & 15u);
          digit = 2u * (s - 5u) + (p >> 4u);
        } else if (s == 7u) {
          index = 48u + (p & 3u);
          digit = (p - 16u) >> 2u;
        }
        const unsigned int power =
            digit == 0u ? 1u : digit == 1u ? 3u : digit == 2u ? 9u : digit == 3u ? 27u : 81u;
        const unsigned int shifted = (static_cast<unsigned int>(block[index]) * power) & 0xFFu;
        word |= ((shifted * 3u) >> 8u) << (8u * b);
      }
      sub.q[i] = __vsub4(word, 0x01010101u);
    }
    low_bit_scale(sub, ggml_half(block + 52u));
  }
};

// llama.cpp `ggml_ue4m3_to_fp32`: an unsigned E4M3 scale, halved because the
// E2M1 table is doubled; 0 and 0x7F decode as zero.
__device__ __forceinline__ float ggml_ue4m3(unsigned int x) {
  if (x == 0u || x == 0x7Fu) return 0.0f;
  const unsigned int exponent = (x >> 3u) & 0xFu;
  const float mantissa = static_cast<float>(x & 7u);
  const float raw = exponent == 0u ? ldexpf(mantissa, -9)
                                   : ldexpf(1.0f + mantissa / 8.0f, static_cast<int>(exponent) - 7);
  return 0.5f * raw;
}

// Four 4-bit indices of `word`, from bit `shift` of each byte.
__device__ __forceinline__ unsigned int nvfp4_indices(unsigned int word, unsigned int shift) {
  const unsigned int w = word >> shift;
  return (w & 0xFu) | ((w >> 4u) & 0xF0u) | ((w >> 8u) & 0xF00u) | ((w >> 12u) & 0xF000u);
}

// NVFP4: UE4M3 d[4], one per 16 values, qs[32] of E2M1 indices: values j
// and j + 8 of 16-value group g are the low and high nibbles of qs[8 g + j].
// Each 16 values have their own scale.
struct NVFP4 {
  LIBMIR_GGML_LOW_BIT(64u, 36u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    unsigned int words[4];
    load16(block + 4u + 16u * s, words);
#pragma unroll
    for (unsigned int g = 0; g < 2u; ++g) {
      sub.q[4u * g] = iq4_lookup(nvfp4_indices(words[2u * g], 0u), kMxFp4Table);
      sub.q[4u * g + 1u] = iq4_lookup(nvfp4_indices(words[2u * g + 1u], 0u), kMxFp4Table);
      sub.q[4u * g + 2u] = iq4_lookup(nvfp4_indices(words[2u * g], 4u), kMxFp4Table);
      sub.q[4u * g + 3u] = iq4_lookup(nvfp4_indices(words[2u * g + 1u], 4u), kMxFp4Table);
      sub.scale[g] = ggml_ue4m3(block[2u * s + g]);
      sub.minimum[g] = 0.0f;
    }
  }
};

#undef LIBMIR_GGML_LOW_BIT

}  // namespace
