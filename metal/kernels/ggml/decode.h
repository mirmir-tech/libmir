// The GGML sub-block decoder, composed after `ggml/blocks.h` (and, for IQ1-IQ3
// kernels, `ggml/grids.h` and `ggml/iq.h`, which define GGML_IQ_DECODERS).

// Sub-block s (0 .. elements / 32 - 1) of the block at byte `at`. Template
// parameter names avoid the kernels' own template names, which validation
// defines as macros.
template <uint KIND>
inline void ggml_decode(const device uchar* base, ulong at, uint s, thread GgmlSub& sub) {
  if (KIND == GGML_Q4_0 || KIND == GGML_Q4_1 || KIND == GGML_IQ4_NL) {
    // Q4_0: half d, qs[16]; d (q - 8). Q4_1: half d, half m, qs[16]; d q + m.
    if (KIND == GGML_IQ4_NL) {
      ggml_table(base, at + 2, kGgmlIq4, sub);
      ggml_uniform(sub, ggml_half(base, at), 0.0f);
      return;
    }
    uint words[4];
    ggml_load<4>(base, at + (KIND == GGML_Q4_1 ? 4 : 2), words);
    ggml_nibbles(words, sub);
    if (KIND == GGML_Q4_0) {
      for (uint i = 0; i < 8; ++i) sub.q[i] = ggml_sub_bytes(sub.q[i], 0x08080808u);
    }
    ggml_uniform(sub, ggml_half(base, at), KIND == GGML_Q4_1 ? -ggml_half(base, at + 2) : 0.0f);
  } else if (KIND == GGML_Q5_0 || KIND == GGML_Q5_1) {
    // Q5_0: half d, qh[4], qs[16]; d (q - 16). Q5_1: half d, half m, qh[4],
    // qs[16]; d q + m. Bit j of qh is the fifth bit of value j.
    constexpr uint qh = KIND == GGML_Q5_1 ? 4 : 2;
    uint words[4];
    ggml_load<4>(base, at + qh + 4, words);
    const uint high = ggml_u32(base, at + qh);
    ggml_nibbles(words, sub);
    for (uint i = 0; i < 4; ++i) {
      sub.q[i] |= ggml_spread4(high >> (4 * i));
      sub.q[i + 4] |= ggml_spread4(high >> (16 + 4 * i));
    }
    if (KIND == GGML_Q5_0) {
      for (uint i = 0; i < 8; ++i) sub.q[i] = ggml_sub_bytes(sub.q[i], 0x10101010u);
    }
    ggml_uniform(sub, ggml_half(base, at), KIND == GGML_Q5_1 ? -ggml_half(base, at + 2) : 0.0f);
  } else if (KIND == GGML_Q8_0) {
    // half d, int8 qs[32].
    ggml_load<8>(base, at + 2, sub.q);
    ggml_uniform(sub, ggml_half(base, at), 0.0f);
  } else if (KIND == GGML_Q2_K) {
    // scales[16] (4-bit scale, 4-bit minimum), qs[64], half d, half dmin.
    // Sub-block s is shift 2 (s % 4) of qs[32 (s / 4) ..][0..32].
    const uint part = s >> 2;
    const uint shift = 2 * (s & 3);
    const float d = ggml_half(base, at + 80);
    const float dmin = ggml_half(base, at + 82);
    const uint low = base[at + part * 8 + 2 * (s & 3)];
    const uint high = base[at + part * 8 + 2 * (s & 3) + 1];
    sub.scale = float2(d * float(low & 15), d * float(high & 15));
    sub.minimum = float2(dmin * float(low >> 4), dmin * float(high >> 4));
    ggml_load<8>(base, at + 16 + part * 32, sub.q);
    for (uint i = 0; i < 8; ++i) sub.q[i] = (sub.q[i] >> shift) & 0x03030303u;
  } else if (KIND == GGML_Q3_K) {
    // hmask[32], qs[64], scales[12] (sixteen 6-bit values offset by 32),
    // half d. Sub-block s is shift 2 (s % 4) of qs[32 (s / 4) ..][0..32]; a
    // clear bit s of hmask[i] subtracts 4.
    const uint part = s >> 2;
    const uint shift = 2 * (s & 3);
    uint mask[8];
    uint raw[3];
    ggml_load<8>(base, at, mask);
    ggml_load<8>(base, at + 32 + part * 32, sub.q);
    ggml_load<3>(base, at + 96, raw);
    const uint scales[4] = {
        (raw[0] & 0x0F0F0F0Fu) | (raw[2] & 0x03030303u) << 4,
        (raw[1] & 0x0F0F0F0Fu) | ((raw[2] >> 2) & 0x03030303u) << 4,
        ((raw[0] >> 4) & 0x0F0F0F0Fu) | ((raw[2] >> 4) & 0x03030303u) << 4,
        ((raw[1] >> 4) & 0x0F0F0F0Fu) | ((raw[2] >> 6) & 0x03030303u) << 4,
    };
    const float d = ggml_half(base, at + 108);
    const uint scale = part * 8 + 2 * (s & 3);
    sub.scale = float2(d * float(int(ggml_byte(scales, scale)) - 32),
                       d * float(int(ggml_byte(scales, scale + 1)) - 32));
    sub.minimum = float2(0.0f);
    for (uint i = 0; i < 8; ++i) {
      const uint value = ((sub.q[i] >> shift) & 0x03030303u) | ((mask[i] >> s) & 0x01010101u) << 2;
      sub.q[i] = ggml_sub_bytes(value, 0x04040404u);
    }
  } else if (KIND == GGML_Q4_K || KIND == GGML_Q5_K) {
    // half d, half dmin, scales[12], [Q5_K: qh[32]], qs[128]. Sub-block s
    // takes the low (even s) or high (odd s) nibbles of qs[32 (s / 2) ..]
    // and, for Q5_K, bit s of qh[i]; value = d * sc * q - dmin * m.
    const uint2 scale = ggml_scale_min_k4(base, at + 4, s);
    ggml_uniform(sub, ggml_half(base, at) * float(scale.x),
                 ggml_half(base, at + 2) * float(scale.y));
    const uint shift = (s & 1) * 4;
    ggml_load<8>(base, at + (KIND == GGML_Q5_K ? 48 : 16) + (s >> 1) * 32, sub.q);
    for (uint i = 0; i < 8; ++i) sub.q[i] = (sub.q[i] >> shift) & 0x0F0F0F0Fu;
    if (KIND == GGML_Q5_K) {
      uint high[8];
      ggml_load<8>(base, at + 16, high);
      for (uint i = 0; i < 8; ++i) sub.q[i] |= ((high[i] >> s) & 0x01010101u) << 4;
    }
  } else if (KIND == GGML_Q6_K) {
    // ql[128], qh[64], int8 scales[16], half d. Sub-block s is quarter s % 4
    // of half s / 4; each 16 values share one scale.
    const uint part = s >> 2;
    const uint quarter = s & 3;
    uint high[8];
    ggml_load<8>(base, at + part * 64 + (quarter & 1) * 32, sub.q);
    ggml_load<8>(base, at + 128 + part * 32, high);
    const float d = ggml_half(base, at + 208);
    const ulong scales = at + 192 + part * 8 + 2 * quarter;
    sub.scale = float2(d * float(as_type<char>(base[scales])),
                       d * float(as_type<char>(base[scales + 1])));
    sub.minimum = float2(0.0f);
    const uint low_shift = quarter < 2 ? 0 : 4;
    const uint high_shift = 2 * quarter;
    for (uint i = 0; i < 8; ++i) {
      const uint value = ((sub.q[i] >> low_shift) & 0x0F0F0F0Fu) |
                         ((high[i] >> high_shift) & 0x03030303u) << 4;
      sub.q[i] = ggml_sub_bytes(value, 0x20202020u);
    }
  } else if (KIND == GGML_IQ4_XS) {
    // half d, ushort scales_h, scales_l[4], qs[128]; a 6-bit scale per
    // sub-block, offset by 32.
    const uint high = ((uint(base[at + 2]) | uint(base[at + 3]) << 8) >> (2 * s)) & 3;
    const uint low = (base[at + 4 + s / 2] >> (4 * (s & 1))) & 15;
    ggml_table(base, at + 8 + 16 * s, kGgmlIq4, sub);
    ggml_uniform(sub, ggml_half(base, at) * float(int(low | high << 4) - 32), 0.0f);
  } else if (KIND == GGML_MXFP4) {
    // E8M0 e, qs[16] of e2m1 indices; the table is doubled, so the scale is
    // 2^(e - 128).
    ggml_table(base, at + 1, kGgmlMxFp4, sub);
    ggml_uniform(sub, ldexp(1.0f, int(base[at]) - 128), 0.0f);
  }
#ifdef GGML_IQ_DECODERS
  else {
    ggml_decode_iq<KIND>(base, at, s, sub);
  }
#endif
}

// Sub-block s of row `row` of a `[rows, row_bytes]` matrix of KIND blocks.
template <uint KIND, uint WIDTH>
inline void ggml_decode_row(const device uchar* weight, ulong row, uint s, thread GgmlSub& sub) {
  constexpr uint per_block = ggml_block_elements(KIND) / 32;
  const ulong at =
      row * ggml_row_bytes(KIND, WIDTH) + ulong(s / per_block) * ggml_block_bytes(KIND);
  ggml_decode<KIND>(weight, at, s % per_block, sub);
}

inline float ggml_value(thread const GgmlSub& sub, uint i) {
  return sub.scale[i >> 4] * float(as_type<char>(uchar(ggml_byte(sub.q, i)))) -
         sub.minimum[i >> 4];
}

// A sub-block's values as floats, converted once for every token they meet.
struct GgmlFloats {
  float4 q[8];
  float2 scale;
  float2 minimum;
};

inline GgmlFloats ggml_floats(thread const GgmlSub& sub) {
  GgmlFloats floats;
  for (uint k = 0; k < 8; ++k) floats.q[k] = float4(as_type<char4>(sub.q[k]));
  floats.scale = sub.scale;
  floats.minimum = sub.minimum;
  return floats;
}

// sum_i sub[i] * x[i] given the activations' two half sums.
inline float ggml_dot(thread const GgmlFloats& sub, thread const float4* x, float2 sums) {
  float2 dot = 0.0f;
  for (uint k = 0; k < 8; ++k) dot[k >> 2] += metal::dot(sub.q[k], x[k]);
  return sub.scale.x * dot.x + sub.scale.y * dot.y - sub.minimum.x * sums.x -
         sub.minimum.y * sums.y;
}
