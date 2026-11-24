// GGML IQ1-IQ3 superblocks, composed after `ggml/grids/iq2.cuh` and
// `ggml/grids/iq13.cuh`. Values are signed grid entries, int8 for `dp4a`:
// IQ2 and IQ3 grids hold magnitudes with separate sign bits; IQ1 grids hold
// -1/0/1 plus a +-1/8 shift, stored here exactly as 8 * value +- 1 with the
// scale divided by 8. Follows `dequantize_row_iq*` in llama.cpp v1.0.0.
namespace {

__device__ __forceinline__ unsigned int byte_u16(const unsigned char* bytes) {
  return static_cast<unsigned int>(bytes[0]) | (static_cast<unsigned int>(bytes[1]) << 8u);
}

__device__ __forceinline__ unsigned int byte_u32(const unsigned char* bytes) {
  return byte_u16(bytes) | (byte_u16(bytes + 2u) << 16u);
}

// Bytes of `values` negated where bits 0..3 of `signs` are set.
__device__ __forceinline__ unsigned int signed4(unsigned int values, unsigned int signs) {
  const unsigned int mask = ((signs & 0xFu) * 0x00204081u & 0x01010101u) * 0xFFu;
  return __vsub4(values ^ mask, mask);
}

// `ksigns_iq2xs`: seven sign bits and their parity as the eighth.
__device__ __forceinline__ unsigned int even_signs(unsigned int bits) {
  return bits | ((__popc(bits) & 1u) << 7u);
}

// Eight grid magnitudes with eight sign bits into two words.
__device__ __forceinline__ void signed8(
    unsigned long long grid, unsigned int signs, unsigned int* q) {
  q[0] = signed4(static_cast<unsigned int>(grid), signs);
  q[1] = signed4(static_cast<unsigned int>(grid >> 32u), signs >> 4u);
}

// A 4-bit IQ2 scale: d * (0.5 + s) / 4.
__device__ __forceinline__ float iq2_scale(float d, unsigned int scale) {
  return d * (0.5f + static_cast<float>(scale)) * 0.25f;
}

#define LIBMIR_GGML_IQ(BYTES, STAGED)                                                   \
  static constexpr unsigned int elements = 256u;                                       \
  static constexpr unsigned int bytes = BYTES;                                         \
  static constexpr unsigned int staged_tokens = STAGED;                                \
  static constexpr unsigned int unroll = 1u;                                           \
  static constexpr unsigned int units = 8u;                                            \
  static constexpr unsigned int width = 1u;                                            \
  __device__ static void decode_unit(                                                  \
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {  \
    decode(block, u, sub[0]);                                                          \
    index[0] = u;                                                                      \
  }

// IQ2_XXS: half d, qs u16[32]; sub-block s is 8 bytes: four grid indices,
// then 4 x 7 sign bits and a 4-bit scale in the top bits.
struct IQ2_XXS {
  LIBMIR_GGML_IQ(66u, 2u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int indices = byte_u32(block + 2u + 8u * s);
    const unsigned int aux = byte_u32(block + 6u + 8u * s);
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned long long grid = __ldg(&k_iq2xxs_grid[(indices >> (8u * l)) & 0xFFu]);
      signed8(grid, even_signs((aux >> (7u * l)) & 127u), sub.q + 2u * l);
    }
    sub.scale[0] = sub.scale[1] = iq2_scale(ggml_half(block), aux >> 28u);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// IQ2_XS: half d, qs u16[32] (9-bit index, 7 sign bits), scales[8] (4 bits
// per 16 values).
struct IQ2_XS {
  LIBMIR_GGML_IQ(74u, 2u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int value = byte_u16(block + 2u + 2u * (4u * s + l));
      signed8(__ldg(&k_iq2xs_grid[value & 511u]), even_signs(value >> 9u), sub.q + 2u * l);
    }
    const unsigned int scales = block[66u + s];
    sub.scale[0] = iq2_scale(ggml_half(block), scales & 0xFu);
    sub.scale[1] = iq2_scale(ggml_half(block), scales >> 4u);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// IQ2_S: half d, qs[64] (32 low index bytes, then 32 sign bytes), qh[8]
// (two high index bits per group), scales[8].
struct IQ2_S {
  LIBMIR_GGML_IQ(82u, 16u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int high = block[66u + s];
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int index = block[2u + 4u * s + l] | ((high << (8u - 2u * l)) & 0x300u);
      signed8(__ldg(&k_iq2s_grid[index]), block[34u + 4u * s + l], sub.q + 2u * l);
    }
    const unsigned int scales = block[74u + s];
    sub.scale[0] = iq2_scale(ggml_half(block), scales & 0xFu);
    sub.scale[1] = iq2_scale(ggml_half(block), scales >> 4u);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// IQ3_XXS: half d, qs[64] (two 4-value grid indices per group), then u32[8]
// of 4 x 7 sign bits and a 4-bit scale; value = d * (0.5 + s) / 2 * grid.
struct IQ3_XXS {
  LIBMIR_GGML_IQ(98u, 8u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int aux = byte_u32(block + 66u + 4u * s);
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int signs = even_signs((aux >> (7u * l)) & 127u);
      sub.q[2u * l] = signed4(__ldg(&k_iq3xxs_grid[block[2u + 8u * s + 2u * l]]), signs);
      sub.q[2u * l + 1u] = signed4(__ldg(&k_iq3xxs_grid[block[3u + 8u * s + 2u * l]]), signs >> 4u);
    }
    sub.scale[0] = sub.scale[1] =
        ggml_half(block) * (0.5f + static_cast<float>(aux >> 28u)) * 0.5f;
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// IQ3_S: half d, qs[64], qh[8] (ninth index bit), signs[32], scales[4] (4 bits
// per sub-block); value = d * (1 + 2 * s) * grid.
struct IQ3_S {
  LIBMIR_GGML_IQ(110u, 16u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int high = block[66u + s];
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int signs = block[74u + 4u * s + l];
      const unsigned int first = block[2u + 8u * s + 2u * l] | ((high << (8u - 2u * l)) & 256u);
      const unsigned int second = block[3u + 8u * s + 2u * l] | ((high << (7u - 2u * l)) & 256u);
      sub.q[2u * l] = signed4(__ldg(&k_iq3s_grid[first]), signs);
      sub.q[2u * l + 1u] = signed4(__ldg(&k_iq3s_grid[second]), signs >> 4u);
    }
    const unsigned int scale = (block[106u + s / 2u] >> (4u * (s & 1u))) & 0xFu;
    sub.scale[0] = sub.scale[1] = ggml_half(block) * static_cast<float>(1u + 2u * scale);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// Eight -1/0/1 grid values shifted by +-1/8, times 8: 8 * g + 1 or 8 * g - 1.
__device__ __forceinline__ void iq1_values(
    unsigned long long grid, bool negative, unsigned int* q) {
#pragma unroll
  for (unsigned int part = 0; part < 2u; ++part) {
    const unsigned int eight = (static_cast<unsigned int>(grid >> (32u * part)) << 3u) & 0xF8F8F8F8u;
    q[part] = negative ? __vsub4(eight, 0x01010101u) : __vadd4(eight, 0x01010101u);
  }
}

// IQ1_S: half d, qs[32] (low index bytes), qh u16[8] (3 high index bits per
// group, a 3-bit scale, the shift's sign); value = d * (2 * s + 1) * (g +- 1/8).
struct IQ1_S {
  LIBMIR_GGML_IQ(50u, 2u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int high = byte_u16(block + 34u + 2u * s);
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int index = block[2u + 4u * s + l] | (((high >> (3u * l)) & 7u) << 8u);
      iq1_values(__ldg(&k_iq1s_grid[index]), (high & 0x8000u) != 0u, sub.q + 2u * l);
    }
    sub.scale[0] = sub.scale[1] =
        ggml_half(block) * static_cast<float>(2u * ((high >> 12u) & 7u) + 1u) * 0.125f;
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

// IQ1_M: qs[32], qh[16] (3 high index bits and the shift's sign per group),
// scales u16[4] (3-bit scales per 16 values; their top nibbles hold d).
struct IQ1_M {
  LIBMIR_GGML_IQ(56u, 16u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    unsigned int scales[4];
#pragma unroll
    for (unsigned int k = 0; k < 4u; ++k) scales[k] = byte_u16(block + 48u + 2u * k);
    const unsigned int bits = (scales[0] >> 12u) | ((scales[1] >> 8u) & 0x00F0u) |
                              ((scales[2] >> 4u) & 0x0F00u) | (scales[3] & 0xF000u);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
    const unsigned int packed = scales[s / 2u] >> (6u * (s & 1u));
#pragma unroll
    for (unsigned int l = 0; l < 4u; ++l) {
      const unsigned int high = block[32u + 2u * s + l / 2u] >> (4u * (l & 1u));
      const unsigned int index = block[4u * s + l] | ((high & 7u) << 8u);
      iq1_values(__ldg(&k_iq1s_grid[index]), (high & 8u) != 0u, sub.q + 2u * l);
    }
    sub.scale[0] = d * static_cast<float>(2u * (packed & 7u) + 1u) * 0.125f;
    sub.scale[1] = d * static_cast<float>(2u * ((packed >> 3u) & 7u) + 1u) * 0.125f;
    sub.minimum[0] = sub.minimum[1] = 0.0f;
  }
};

#undef LIBMIR_GGML_IQ

}  // namespace
