// GGML K-quant superblocks of 256 values, composed after `ggml/blocks.cuh`.
namespace {

struct Q4_K {
  static constexpr unsigned int elements = 256u;
  static constexpr unsigned int bytes = 144u;
  // Two units in flight per lane: measured faster for Q4_K only.
  static constexpr unsigned int staged_tokens = 1u;
  static constexpr bool whole_scales = true;
  static constexpr unsigned int unroll = 2u;
  static constexpr unsigned int units = 4u;
  static constexpr unsigned int width = 2u;
  // Unit u: sub-blocks 2u (low nibbles) and 2u + 1 (high nibbles) of
  // qs[32u..][0..32]. Blocks are 16-byte aligned: three vector loads.
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {
    const uint4* vectors = reinterpret_cast<const uint4*>(block);
    const uint4 header = __ldg(vectors);
    const uint4 low = __ldg(vectors + 1u + 2u * u);
    const uint4 high = __ldg(vectors + 2u + 2u * u);
    const unsigned int scales[3] = {header.y, header.z, header.w};
    const unsigned int q[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(header.x)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(header.x >> 16u)));
#pragma unroll
    for (unsigned int part = 0; part < 2u; ++part) {
      unsigned int scale;
      unsigned int minimum;
      scale_min_k4(scales, 2u * u + part, scale, minimum);
      sub[part].scale[0] = sub[part].scale[1] = d * static_cast<float>(scale);
      sub[part].minimum[0] = sub[part].minimum[1] = dmin * static_cast<float>(minimum);
      index[part] = 2u * u + part;
#pragma unroll
      for (unsigned int i = 0; i < 8u; ++i) sub[part].q[i] = (q[i] >> (4u * part)) & 0x0F0F0F0Fu;
    }
  }
  // Q4_K: half d, half dmin, scales[12], qs[128]; sub-block s holds the low
  // (even s) or high (odd s) nibbles of qs[32 * (s / 2) ..][0..32].
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned char* scales = block + 4u;
    unsigned int scale;
    unsigned int minimum;
    if (s < 4u) {
      scale = scales[s] & 63u;
      minimum = scales[s + 4u] & 63u;
    } else {
      scale = (scales[s + 4u] & 0xFu) | ((scales[s - 4u] >> 6u) << 4u);
      minimum = (scales[s + 4u] >> 4u) | ((scales[s] >> 6u) << 4u);
    }
    sub.scale[0] = sub.scale[1] = ggml_half(block) * static_cast<float>(scale);
    sub.minimum[0] = sub.minimum[1] = ggml_half(block + 2u) * static_cast<float>(minimum);
    load32(block + 16u + (s >> 1u) * 32u, sub.q);
    const unsigned int shift = (s & 1u) * 4u;
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = (sub.q[i] >> shift) & 0x0F0F0F0Fu;
  }
};

struct Q5_K {
  static constexpr unsigned int elements = 256u;
  static constexpr unsigned int bytes = 176u;
  static constexpr unsigned int staged_tokens = 16u;
  static constexpr bool whole_scales = true;
  static constexpr unsigned int unroll = 2u;
  static constexpr unsigned int units = 4u;
  static constexpr unsigned int width = 2u;
  // Q5_K: half d, half dmin, scales[12], qh[32], qs[128]. Sub-block s takes
  // the low (even s) or high (odd s) nibbles of qs[32 * (s / 2) ..][0..32]
  // and bit s of qh[l] as the fifth bit; value = d * sc * q - dmin * m.
  // Blocks are 16-byte aligned: unit u (sub-blocks 2u, 2u + 1) is five
  // vector loads. Four sub-blocks per unit measured slower (too few units).
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {
    const uint4* vectors = reinterpret_cast<const uint4*>(block);
    const uint4 header = __ldg(vectors);
    const uint4 high_low = __ldg(vectors + 1u);
    const uint4 high_high = __ldg(vectors + 2u);
    const uint4 low = __ldg(vectors + 3u + 2u * u);
    const uint4 high = __ldg(vectors + 4u + 2u * u);
    const unsigned int scales[3] = {header.y, header.z, header.w};
    const unsigned int qh[8] = {high_low.x, high_low.y, high_low.z, high_low.w,
                                high_high.x, high_high.y, high_high.z, high_high.w};
    const unsigned int q[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(header.x)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(header.x >> 16u)));
#pragma unroll
    for (unsigned int part = 0; part < 2u; ++part) {
      const unsigned int s = 2u * u + part;
      unsigned int scale;
      unsigned int minimum;
      scale_min_k4(scales, s, scale, minimum);
      sub[part].scale[0] = sub[part].scale[1] = d * static_cast<float>(scale);
      sub[part].minimum[0] = sub[part].minimum[1] = dmin * static_cast<float>(minimum);
      index[part] = s;
#pragma unroll
      for (unsigned int i = 0; i < 8u; ++i) {
        sub[part].q[i] = ((q[i] >> (4u * part)) & 0x0F0F0F0Fu) | (((qh[i] >> s) & 0x01010101u) << 4u);
      }
    }
  }
  // From bytes at any alignment, for the staged GEMV.
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    unsigned int words[4];
    load16(block, words);
    const unsigned int scales[3] = {words[1], words[2], words[3]};
    unsigned int scale;
    unsigned int minimum;
    scale_min_k4(scales, s, scale, minimum);
    sub.scale[0] = sub.scale[1] = ggml_half(block) * static_cast<float>(scale);
    sub.minimum[0] = sub.minimum[1] = ggml_half(block + 2u) * static_cast<float>(minimum);
    unsigned int qh[8];
    load32(block + 16u, qh);
    load32(block + 48u + (s >> 1u) * 32u, sub.q);
    const unsigned int shift = (s & 1u) * 4u;
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      sub.q[i] = ((sub.q[i] >> shift) & 0x0F0F0F0Fu) | (((qh[i] >> s) & 0x01010101u) << 4u);
    }
  }
};

struct Q6_K {
  static constexpr unsigned int elements = 256u;
  static constexpr unsigned int bytes = 210u;
  // One sub-block per unit: pairing quarters that share ql bytes measured
  // slower (more registers, two misaligned loads per lane).
  static constexpr unsigned int staged_tokens = 16u;
  static constexpr unsigned int unroll = 1u;
  static constexpr unsigned int units = 8u;
  static constexpr unsigned int width = 1u;
  __device__ static void decode_unit(
      const unsigned char* block, unsigned int u, IntSub* sub, unsigned int* index) {
    decode(block, u, sub[0]);
    index[0] = u;
  }
  // Q6_K: ql[128], qh[64], int8 scales[16], half d. Sub-block s is quarter
  // s % 4 of half s / 4; each 16 values share one scale.
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int part = s >> 2u;
    const unsigned int quarter = s & 3u;
    unsigned int high[8];
    load32(block + part * 64u + (quarter & 1u) * 32u, sub.q);
    load32(block + 128u + part * 32u, high);
    const signed char* scales = reinterpret_cast<const signed char*>(block + 192u + part * 8u);
    const float d = ggml_half(block + 208u);
    sub.scale[0] = d * static_cast<float>(scales[2u * quarter]);
    sub.scale[1] = d * static_cast<float>(scales[2u * quarter + 1u]);
    sub.minimum[0] = sub.minimum[1] = 0.0f;
    const unsigned int low_shift = quarter < 2u ? 0u : 4u;
    const unsigned int high_shift = 2u * quarter;
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      const unsigned int value = ((sub.q[i] >> low_shift) & 0x0F0F0F0Fu) |
                                 (((high[i] >> high_shift) & 0x03030303u) << 4u);
      sub.q[i] = __vsub4(value, 0x20202020u);
    }
  }
};

// Shared traits of the K-quants read one sub-block per unit from shared
// memory (`staged`): their blocks are not 16-byte aligned.
#define LIBMIR_GGML_STAGED_K(BYTES, STAGED)                                             \
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

// Q3_K: hmask[32], qs[64], scales[12] (sixteen 6-bit values, offset by 32),
// half d. Sub-block s is shift 2 * (s % 4) of qs[32 * (s / 4) ..][0..32]
// with bit s of hmask[l] clear meaning minus 4; value = d * (sc - 32) * q.
struct Q3_K {
  LIBMIR_GGML_STAGED_K(110u, 16u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int part = s >> 2u;
    const unsigned int shift = 2u * (s & 3u);
    unsigned int mask[8];
    unsigned int q[8];
    unsigned int raw[4];
    load32(block, mask);
    load32(block + 32u + part * 32u, q);
    load16(block + 96u, raw);
    const unsigned int scales[4] = {
        (raw[0] & 0x0F0F0F0Fu) | ((raw[2] & 0x03030303u) << 4u),
        (raw[1] & 0x0F0F0F0Fu) | (((raw[2] >> 2u) & 0x03030303u) << 4u),
        ((raw[0] >> 4u) & 0x0F0F0F0Fu) | (((raw[2] >> 4u) & 0x03030303u) << 4u),
        ((raw[1] >> 4u) & 0x0F0F0F0Fu) | (((raw[2] >> 6u) & 0x03030303u) << 4u),
    };
    const float d = ggml_half(block + 108u);
#pragma unroll
    for (unsigned int side = 0; side < 2u; ++side) {
      const int scale = static_cast<int>(byte_at(scales, part * 8u + 2u * (s & 3u) + side)) - 32;
      sub.scale[side] = d * static_cast<float>(scale);
      sub.minimum[side] = 0.0f;
    }
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
      const unsigned int value =
          ((q[i] >> shift) & 0x03030303u) | (((mask[i] >> s) & 0x01010101u) << 2u);
      sub.q[i] = __vsub4(value, 0x04040404u);
    }
  }
};

// Q2_K: scales[16] (4-bit scale, 4-bit minimum), qs[64], half d, half dmin.
// Sub-block s is shift 2 * (s % 4) of qs[32 * (s / 4) ..][0..32]; each 16
// values have their own scale and minimum: value = d * sc * q - dmin * m.
struct Q2_K {
  LIBMIR_GGML_STAGED_K(84u, 4u)
  __device__ static void decode(const unsigned char* block, unsigned int s, IntSub& sub) {
    const unsigned int part = s >> 2u;
    const unsigned int shift = 2u * (s & 3u);
    unsigned int q[8];
    load32(block + 16u + part * 32u, q);
    const float d = ggml_half(block + 80u);
    const float dmin = ggml_half(block + 82u);
#pragma unroll
    for (unsigned int side = 0; side < 2u; ++side) {
      const unsigned int packed = block[part * 8u + 2u * (s & 3u) + side];
      sub.scale[side] = d * static_cast<float>(packed & 0xFu);
      sub.minimum[side] = dmin * static_cast<float>(packed >> 4u);
    }
#pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) sub.q[i] = (q[i] >> shift) & 0x03030303u;
  }
};

#undef LIBMIR_GGML_STAGED_K

}  // namespace
