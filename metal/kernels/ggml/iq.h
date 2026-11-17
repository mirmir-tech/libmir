// GGML IQ1-IQ3 superblocks, composed after `ggml/blocks.h` and `ggml/grids.h`
// and before `ggml/decode.h`. Values are signed grid entries packed as bytes:
// IQ2 and IQ3 grids hold magnitudes with separate sign bits; IQ1 grids hold
// -1/0/1 plus a +-1/8 shift, stored exactly as 8 * value +- 1 with the scale
// divided by 8. Follows `dequantize_row_iq*` in llama.cpp v1.0.0, as the CUDA
// decoders in `cuda/kernels/ggml/iq.cuh` do.
#define GGML_IQ_DECODERS 1

inline uint ggml_u16(const device uchar* base, ulong at) {
  return uint(base[at]) | uint(base[at + 1]) << 8;
}

// Bytes of `values` negated where bits 0..3 of `signs` are set.
inline uint ggml_signed4(uint values, uint signs) {
  const char4 value = as_type<char4>(values);
  const bool4 negative = bool4((uint4(signs) >> uint4(0, 1, 2, 3)) & 1u);
  return as_type<uint>(select(value, char4(-int4(value)), negative));
}

// `ksigns_iq2xs`: seven sign bits and their parity as the eighth.
inline uint ggml_even_signs(uint bits) {
  return bits | (popcount(bits) & 1u) << 7;
}

// Eight grid magnitudes with eight sign bits into two words.
inline void ggml_signed8(ulong grid, uint signs, thread uint* q) {
  q[0] = ggml_signed4(uint(grid), signs);
  q[1] = ggml_signed4(uint(grid >> 32), signs >> 4);
}

// A 4-bit IQ2 scale: d * (0.5 + s) / 4.
inline float ggml_iq2_scale(float d, uint scale) {
  return d * (0.5f + float(scale)) * 0.25f;
}

// Eight -1/0/1 grid values shifted by +-1/8, times 8: 8 g + 1 or 8 g - 1.
inline void ggml_iq1_values(ulong grid, bool negative, thread uint* q) {
  for (uint part = 0; part < 2; ++part) {
    const char4 eight = as_type<char4>((uint(grid >> (32 * part)) << 3) & 0xF8F8F8F8u);
    q[part] = as_type<uint>(char4(int4(eight) + (negative ? -1 : 1)));
  }
}

template <uint KIND>
inline void ggml_decode_iq(const device uchar* base, ulong at, uint s, thread GgmlSub& sub) {
  sub.minimum = float2(0.0f);
  if (KIND == GGML_IQ2_XXS) {
    // half d, qs u16[32]; sub-block s is 8 bytes: four grid indices, then
    // 4 x 7 sign bits and a 4-bit scale in the top bits.
    const uint indices = ggml_u32(base, at + 2 + 8 * s);
    const uint aux = ggml_u32(base, at + 6 + 8 * s);
    for (uint l = 0; l < 4; ++l) {
      const ulong grid = k_iq2xxs_grid[(indices >> (8 * l)) & 0xFF];
      ggml_signed8(grid, ggml_even_signs((aux >> (7 * l)) & 127), sub.q + 2 * l);
    }
    sub.scale = float2(ggml_iq2_scale(ggml_half(base, at), aux >> 28));
  } else if (KIND == GGML_IQ2_XS) {
    // half d, qs u16[32] (9-bit index, 7 sign bits), scales[8] (4 bits per 16
    // values).
    for (uint l = 0; l < 4; ++l) {
      const uint value = ggml_u16(base, at + 2 + 2 * (4 * s + l));
      ggml_signed8(k_iq2xs_grid[value & 511], ggml_even_signs(value >> 9), sub.q + 2 * l);
    }
    const uint scales = base[at + 66 + s];
    const float d = ggml_half(base, at);
    sub.scale = float2(ggml_iq2_scale(d, scales & 15), ggml_iq2_scale(d, scales >> 4));
  } else if (KIND == GGML_IQ2_S) {
    // half d, qs[64] (32 low index bytes, then 32 sign bytes), qh[8] (two
    // high index bits per group), scales[8].
    const uint high = base[at + 66 + s];
    for (uint l = 0; l < 4; ++l) {
      const uint index = base[at + 2 + 4 * s + l] | ((high << (8 - 2 * l)) & 0x300);
      ggml_signed8(k_iq2s_grid[index], base[at + 34 + 4 * s + l], sub.q + 2 * l);
    }
    const uint scales = base[at + 74 + s];
    const float d = ggml_half(base, at);
    sub.scale = float2(ggml_iq2_scale(d, scales & 15), ggml_iq2_scale(d, scales >> 4));
  } else if (KIND == GGML_IQ3_XXS) {
    // half d, qs[64] (two 4-value grid indices per group), then u32[8] of
    // 4 x 7 sign bits and a 4-bit scale; value = d * (0.5 + s) / 2 * grid.
    const uint aux = ggml_u32(base, at + 66 + 4 * s);
    for (uint l = 0; l < 4; ++l) {
      const uint signs = ggml_even_signs((aux >> (7 * l)) & 127);
      sub.q[2 * l] = ggml_signed4(k_iq3xxs_grid[base[at + 2 + 8 * s + 2 * l]], signs);
      sub.q[2 * l + 1] = ggml_signed4(k_iq3xxs_grid[base[at + 3 + 8 * s + 2 * l]], signs >> 4);
    }
    sub.scale = float2(ggml_half(base, at) * (0.5f + float(aux >> 28)) * 0.5f);
  } else if (KIND == GGML_IQ3_S) {
    // half d, qs[64], qh[8] (ninth index bit), signs[32], scales[4] (4 bits
    // per sub-block); value = d * (1 + 2 * s) * grid.
    const uint high = base[at + 66 + s];
    for (uint l = 0; l < 4; ++l) {
      const uint signs = base[at + 74 + 4 * s + l];
      const uint first = base[at + 2 + 8 * s + 2 * l] | ((high << (8 - 2 * l)) & 256);
      const uint second = base[at + 3 + 8 * s + 2 * l] | ((high << (7 - 2 * l)) & 256);
      sub.q[2 * l] = ggml_signed4(k_iq3s_grid[first], signs);
      sub.q[2 * l + 1] = ggml_signed4(k_iq3s_grid[second], signs >> 4);
    }
    const uint scale = (base[at + 106 + s / 2] >> (4 * (s & 1))) & 15;
    sub.scale = float2(ggml_half(base, at) * float(1 + 2 * scale));
  } else if (KIND == GGML_IQ1_S) {
    // half d, qs[32] (low index bytes), qh u16[8] (3 high index bits per
    // group, a 3-bit scale, the shift's sign); value = d (2 s + 1) (g +- 1/8).
    const uint high = ggml_u16(base, at + 34 + 2 * s);
    for (uint l = 0; l < 4; ++l) {
      const uint index = base[at + 2 + 4 * s + l] | ((high >> (3 * l)) & 7) << 8;
      ggml_iq1_values(k_iq1s_grid[index], (high & 0x8000) != 0, sub.q + 2 * l);
    }
    sub.scale = float2(ggml_half(base, at) * float(2 * ((high >> 12) & 7) + 1) * 0.125f);
  } else if (KIND == GGML_IQ1_M) {
    // qs[32], qh[16] (3 high index bits and the shift's sign per group),
    // scales u16[4] (3-bit scales per 16 values; their top nibbles hold d).
    uint scales[4];
    for (uint k = 0; k < 4; ++k) scales[k] = ggml_u16(base, at + 48 + 2 * k);
    const uint bits = scales[0] >> 12 | (scales[1] >> 8 & 0x00F0) | (scales[2] >> 4 & 0x0F00) |
                      (scales[3] & 0xF000);
    const float d = float(as_type<half>(ushort(bits)));
    const uint packed = scales[s / 2] >> (6 * (s & 1));
    for (uint l = 0; l < 4; ++l) {
      const uint high = base[at + 32 + 2 * s + l / 2] >> (4 * (l & 1));
      const uint index = base[at + 4 * s + l] | (high & 7) << 8;
      ggml_iq1_values(k_iq1s_grid[index], (high & 8) != 0, sub.q + 2 * l);
    }
    sub.scale = float2(d * float(2 * (packed & 7) + 1) * 0.125f,
                       d * float(2 * ((packed >> 3) & 7) + 1) * 0.125f);
  }
}
