// GGML block layouts and decoding helpers shared by the GGML GEMV,
// dequantization and embedding kernels, composed before `ggml/decode.h`
// (and, for IQ1-IQ3 kernels, `ggml/grids.h` and `ggml/iq.h`). Values follow llama.cpp v1.0.0 `ggml-quants.c` (`dequantize_row_*`)
// value for value, as the CUDA decoders in `cuda/kernels/ggml` do.
//
// Every kernel works on 32-value sub-blocks: sub-block s of a row covers
// columns 32 s .. 32 s + 31. A decoded sub-block packs its values as signed
// bytes, value i in byte i % 4 of q[i / 4], with a per-16 scale and minimum:
// value i = scale[i / 16] * q[i] - minimum[i / 16]. Block types are their
// GGUF ids, so one template parameter selects a decoder. Blocks are only
// 2-byte aligned (MXFP4: 1), so bytes are shifted out of aligned words.
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

constant constexpr uint GGML_Q4_0 = 2;
constant constexpr uint GGML_Q4_1 = 3;
constant constexpr uint GGML_Q5_0 = 6;
constant constexpr uint GGML_Q5_1 = 7;
constant constexpr uint GGML_Q8_0 = 8;
constant constexpr uint GGML_Q2_K = 10;
constant constexpr uint GGML_Q3_K = 11;
constant constexpr uint GGML_Q4_K = 12;
constant constexpr uint GGML_Q5_K = 13;
constant constexpr uint GGML_Q6_K = 14;
constant constexpr uint GGML_IQ2_XXS = 16;
constant constexpr uint GGML_IQ2_XS = 17;
constant constexpr uint GGML_IQ3_XXS = 18;
constant constexpr uint GGML_IQ1_S = 19;
constant constexpr uint GGML_IQ4_NL = 20;
constant constexpr uint GGML_IQ3_S = 21;
constant constexpr uint GGML_IQ2_S = 22;
constant constexpr uint GGML_IQ1_M = 29;
constant constexpr uint GGML_IQ4_XS = 23;
constant constexpr uint GGML_MXFP4 = 39;

constexpr uint ggml_block_elements(uint type) {
  return type == GGML_Q4_0 || type == GGML_Q4_1 || type == GGML_Q5_0 || type == GGML_Q5_1 ||
                 type == GGML_Q8_0 || type == GGML_IQ4_NL || type == GGML_MXFP4
             ? 32
             : 256;
}

constexpr uint ggml_block_bytes(uint type) {
  switch (type) {
    case GGML_Q4_0: return 18;
    case GGML_Q4_1: return 20;
    case GGML_Q5_0: return 22;
    case GGML_Q5_1: return 24;
    case GGML_Q8_0: return 34;
    case GGML_Q2_K: return 84;
    case GGML_Q3_K: return 110;
    case GGML_Q4_K: return 144;
    case GGML_Q5_K: return 176;
    case GGML_Q6_K: return 210;
    case GGML_IQ2_XXS: return 66;
    case GGML_IQ2_XS: return 74;
    case GGML_IQ3_XXS: return 98;
    case GGML_IQ1_S: return 50;
    case GGML_IQ4_NL: return 18;
    case GGML_IQ3_S: return 110;
    case GGML_IQ2_S: return 82;
    case GGML_IQ1_M: return 56;
    case GGML_IQ4_XS: return 136;
    case GGML_MXFP4: return 17;
    default: return 0;
  }
}

constexpr ulong ggml_row_bytes(uint type, uint columns) {
  return ulong(columns / ggml_block_elements(type)) * ggml_block_bytes(type);
}

struct GgmlSub {
  uint q[8];
  float2 scale;
  float2 minimum;
};

// N little-endian words from byte `at` of a buffer whose base is aligned.
template <uint N>
inline void ggml_load(const device uchar* base, ulong at, thread uint* words) {
  const device uint* aligned = (const device uint*)base + (at >> 2);
  const uint shift = uint(at & 3) * 8;
  if (shift == 0) {
    for (uint i = 0; i < N; ++i) words[i] = aligned[i];
  } else {
    uint previous = aligned[0];
    for (uint i = 0; i < N; ++i) {
      const uint next = aligned[i + 1];
      words[i] = previous >> shift | next << (32 - shift);
      previous = next;
    }
  }
}

inline float ggml_half(const device uchar* base, ulong at) {
  return float(as_type<half>(ushort(ushort(base[at]) | ushort(base[at + 1]) << 8)));
}

inline uint ggml_u32(const device uchar* base, ulong at) {
  return uint(base[at]) | uint(base[at + 1]) << 8 | uint(base[at + 2]) << 16 |
         uint(base[at + 3]) << 24;
}

inline uint ggml_byte(thread const uint* words, uint index) {
  return (words[index >> 2] >> ((index & 3) * 8)) & 0xFF;
}

// Bits 0..3 of `bits` as bit 4 of bytes 0..3: GGML's fifth value bit.
inline uint ggml_spread4(uint bits) {
  return ((bits & 0xF) * 0x00204081u & 0x01010101u) << 4;
}

// Bytes of `value` (each below 0x80) minus `offset` bytes, without borrows
// between bytes; results are signed bytes.
inline uint ggml_sub_bytes(uint value, uint offset) {
  return ((value | 0x80808080u) - offset) ^ 0x80808080u;
}

constant uchar kGgmlIq4[16] = {0x81, 0x98, 0xAD, 0xBF, 0xCF, 0xDD, 0xEA, 0xF6,
                               1,    13,   25,   38,   53,   69,   89,   113};
constant uchar kGgmlMxFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, 0xFF, 0xFE, 0xFD, 0xFC, 0xFA, 0xF8, 0xF4};

inline void ggml_uniform(thread GgmlSub& sub, float scale, float minimum) {
  sub.scale = float2(scale);
  sub.minimum = float2(minimum);
}

// 16 bytes of nibbles: low nibbles are values 0..15, high nibbles 16..31.
inline void ggml_nibbles(thread const uint* words, thread GgmlSub& sub) {
  for (uint i = 0; i < 4; ++i) {
    sub.q[i] = words[i] & 0x0F0F0F0Fu;
    sub.q[i + 4] = (words[i] >> 4) & 0x0F0F0F0Fu;
  }
}

inline uint ggml_lookup(constant uchar* table, uint word, uint shift) {
  uint result = 0;
  for (uint b = 0; b < 4; ++b) result |= uint(table[(word >> (8 * b + shift)) & 15]) << (8 * b);
  return result;
}

inline void ggml_table(const device uchar* base, ulong at, constant uchar* table,
                       thread GgmlSub& sub) {
  uint words[4];
  ggml_load<4>(base, at, words);
  for (uint i = 0; i < 4; ++i) {
    sub.q[i] = ggml_lookup(table, words[i], 0);
    sub.q[i + 4] = ggml_lookup(table, words[i], 4);
  }
}

// Q4_K and Q5_K 6-bit scale and minimum of sub-block s from scales[12].
inline uint2 ggml_scale_min_k4(const device uchar* base, ulong at, uint s) {
  return s < 4 ? uint2(base[at + s] & 63, base[at + s + 4] & 63)
               : uint2((base[at + s + 4] & 15) | (base[at + s - 4] >> 6) << 4,
                       (base[at + s + 4] >> 4) | (base[at + s] >> 6) << 4);
}

