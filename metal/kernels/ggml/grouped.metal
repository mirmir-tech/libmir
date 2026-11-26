// GGML products over expert-sorted routed assignments (prefill): output[a, r]
// = sum_c input[a, c] * W[indices[a]][r, c], with `indices` ascending. A
// threadgroup takes 16 consecutive assignments (two 8-token tiles) and, per
// simdgroup, two 8-row tiles; it walks the experts present in its
// assignments (usually one), masking the others' activations to zero, so
// every expert's blocks are decoded once per tile as in `ggml/mma.metal`.
// Larger token tiles spill registers. Grid: [ceil(OUTPUT / 32) * 64,
// ceil(ASSIGNMENTS / 16), 1], 64 threads.
constexpr uint TILES = 2;
constexpr uint TOKEN_TILES = 2;
constexpr uint SUBS = INPUT / 32;
constexpr uint STRIDE = 136;
const uint simdgroups = threads_per_threadgroup.x / 32;
const uint simd = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint row0 = (threadgroup_position_in_grid.x * simdgroups + simd) * 8 * TILES;
const uint first = threadgroup_position_in_grid.y * 8 * TOKEN_TILES;
if (row0 >= OUTPUT) return;

threadgroup half stage[2][8 * TILES * STRIDE];
threadgroup half* weights = stage[simd];
const ushort fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const ushort fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const uint tile_sub = lane % 4;
const uint last = min(first + 8 * TOKEN_TILES - 1, uint(ASSIGNMENTS - 1));

uint assignment[TOKEN_TILES][2];
uint expert_of[TOKEN_TILES][2];
for (uint u = 0; u < TOKEN_TILES; ++u) {
  for (uint e = 0; e < 2; ++e) {
    assignment[u][e] = min(first + 8 * u + fn + e, last);
    const bool inside = first + 8 * u + fn + e <= last;
    expert_of[u][e] = inside ? indices[assignment[u][e]] : EXPERTS;
  }
}

simdgroup_float8x8 acc[TILES][TOKEN_TILES];
for (uint t = 0; t < TILES; ++t)
  for (uint u = 0; u < TOKEN_TILES; ++u) acc[t][u] = make_filled_simdgroup_matrix<float, 8>(0.0f);

for (uint expert = indices[first]; expert <= min(indices[last], uint(EXPERTS - 1)); ++expert) {
  // Skip experts no assignment of this tile selects (sorted gaps).
  bool used = false;
  for (uint u = 0; u < TOKEN_TILES; ++u) used |= expert_of[u][0] == expert || expert_of[u][1] == expert;
  if (!simd_any(used)) continue;
  float2 present[TOKEN_TILES];
  for (uint u = 0; u < TOKEN_TILES; ++u) {
    present[u] = float2(expert_of[u][0] == expert ? 1.0f : 0.0f, expert_of[u][1] == expert ? 1.0f : 0.0f);
  }
  const ulong base = ulong(expert) * OUTPUT;
  for (uint s0 = 0; s0 < SUBS; s0 += 4) {
    const uint s = s0 + tile_sub;
    for (uint t = 0; t < TILES; ++t) {
      const uint tile_row = 8 * t + lane / 4;
      threadgroup half* w = weights + tile_row * STRIDE + tile_sub * 32;
      if (s < SUBS) {
        GgmlSub sub;
        ggml_decode_row<TYPE, INPUT>(weight, base + min(row0 + tile_row, uint(OUTPUT - 1)), s, sub);
        for (uint i = 0; i < 32; ++i) w[i] = half(ggml_value(sub, i));
      } else {
        for (uint i = 0; i < 32; ++i) w[i] = 0.0h;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 16; ++j) {
      const uint column = s0 * 32 + 8 * j + fm;
      const bool inside = s0 * 32 + 8 * j < INPUT;
      simdgroup_float8x8 b[TOKEN_TILES];
      // Past INPUT (not a multiple of 128) the last token's row ends: read
      // only inside it, as `mma.metal` does.
      for (uint u = 0; u < TOKEN_TILES; ++u) {
        b[u].thread_elements()[0] =
            inside ? present[u].x * float(input[ulong(assignment[u][0]) * INPUT + column]) : 0.0f;
        b[u].thread_elements()[1] =
            inside ? present[u].y * float(input[ulong(assignment[u][1]) * INPUT + column]) : 0.0f;
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
}

for (uint t = 0; t < TILES; ++t) {
  for (uint u = 0; u < TOKEN_TILES; ++u) {
    for (uint e = 0; e < 2; ++e) {
      const uint index = first + 8 * u + fn + e;
      const uint out_row = row0 + 8 * t + fm;
      if (index <= last && out_row < OUTPUT) {
        output[ulong(index) * OUTPUT + out_row] = T(acc[t][u].thread_elements()[e]);
      }
    }
  }
}
