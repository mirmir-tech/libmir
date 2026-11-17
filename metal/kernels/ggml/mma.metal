// GGML product for 2 to 64 tokens on simdgroup matrices: each simdgroup owns
// TILES 8-row output tiles for all tokens, in tiles of 8 (padded). Per 128 columns,
// lane l decodes sub-block l % 4 of row l / 4 of every tile into
// threadgroup memory (as half: 11 significant bits, more than the BF16 the
// prefill path dequantizes to); then sixteen 8x8x8 products per tile
// accumulate rows x tokens, all tiles sharing one activation fragment read
// straight from the cache. Grid: [ceil(OUTPUT / (8 TILES * 2)) * 64, 1, 1].
constexpr uint TILES = 2;
constexpr uint TOKEN_TILES = (TOKENS + 7) / 8;
constexpr uint SUBS = INPUT / 32;
constexpr uint STRIDE = 136;  // values per staged row; padding spreads banks
const uint simdgroups = threads_per_threadgroup.x / 32;
const uint simd = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint row0 = (threadgroup_position_in_grid.x * simdgroups + simd) * 8 * TILES;
if (row0 >= OUTPUT) return;

threadgroup half stage[2][8 * TILES * STRIDE];
threadgroup half* weights = stage[simd];
const ushort fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const ushort fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const uint tile_sub = lane % 4;
// Token tile u: lane reads tokens 8u + fn and 8u + fn + 1 (clamped; padding
// tokens are scaled by zero).
float2 present[TOKEN_TILES];
const device T* x0[TOKEN_TILES];
const device T* x1[TOKEN_TILES];
for (uint u = 0; u < TOKEN_TILES; ++u) {
  const uint first = 8 * u + fn;
  present[u] = float2(first < TOKENS ? 1.0f : 0.0f, first + 1 < TOKENS ? 1.0f : 0.0f);
  x0[u] = input + ulong(min(first, uint(TOKENS - 1))) * INPUT + fm;
  x1[u] = input + ulong(min(first + 1, uint(TOKENS - 1))) * INPUT + fm;
}

simdgroup_float8x8 acc[TILES][TOKEN_TILES];
for (uint t = 0; t < TILES; ++t)
  for (uint u = 0; u < TOKEN_TILES; ++u) acc[t][u] = make_filled_simdgroup_matrix<float, 8>(0.0f);
for (uint s0 = 0; s0 < SUBS; s0 += 4) {
  const uint s = s0 + tile_sub;
  for (uint t = 0; t < TILES; ++t) {
    const uint tile_row = 8 * t + lane / 4;
    threadgroup half* w = weights + tile_row * STRIDE + tile_sub * 32;
    if (s < SUBS) {
      GgmlSub sub;
      ggml_decode_row<TYPE, INPUT>(weight, min(row0 + tile_row, uint(OUTPUT - 1)), s, sub);
      for (uint i = 0; i < 32; ++i) w[i] = half(ggml_value(sub, i));
    } else {
      for (uint i = 0; i < 32; ++i) w[i] = 0.0h;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 16; ++j) {
    const uint column = s0 * 32 + 8 * j;
    const bool inside = column < INPUT;
    simdgroup_float8x8 b[TOKEN_TILES];
    for (uint u = 0; u < TOKEN_TILES; ++u) {
      b[u].thread_elements()[0] = inside ? present[u].x * float(x0[u][column]) : 0.0f;
      b[u].thread_elements()[1] = inside ? present[u].y * float(x1[u][column]) : 0.0f;
    }
    for (uint t = 0; t < TILES; ++t) {
      const half2 pair =
          *(threadgroup const half2*)(weights + (8 * t + fm) * STRIDE + 8 * j + fn);
      simdgroup_float8x8 a;
      a.thread_elements()[0] = float(pair.x);
      a.thread_elements()[1] = float(pair.y);
      for (uint u = 0; u < TOKEN_TILES; ++u) {
        simdgroup_multiply_accumulate(acc[t][u], a, b[u], acc[t][u]);
      }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// acc[t][u]: lane holds row fm of row tile t, tokens 8u + fn and + 1.
for (uint t = 0; t < TILES; ++t) {
  for (uint u = 0; u < TOKEN_TILES; ++u) {
    for (uint e = 0; e < 2; ++e) {
      const uint token = 8 * u + fn + e;
      const uint out_row = row0 + 8 * t + fm;
      if (token < TOKENS && out_row < OUTPUT) {
        output[ulong(token) * OUTPUT + out_row] = T(acc[t][u].thread_elements()[e]);
      }
    }
  }
}
