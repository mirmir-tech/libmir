// GGML GEMV for a few tokens: output[t, r] = sum_c input[t, c] * W[r, c].
// Each simdgroup owns ROWS consecutive rows; lane l walks sub-blocks
// l, l + 32, ... of those rows, decodes each once and applies it to all
// TOKENS activation rows, so every weight byte is read once per launch.
// Grid: [ceil(OUTPUT / (ROWS * simdgroups)) * threads, 1, 1].
constexpr uint SUBS = INPUT / 32;
const uint simdgroups = threads_per_threadgroup.x / 32;
const uint row0 =
    (threadgroup_position_in_grid.x * simdgroups + simdgroup_index_in_threadgroup) * ROWS;
const uint lane = thread_index_in_simdgroup;
if (row0 >= OUTPUT) return;

float acc[ROWS][TOKENS];
for (uint r = 0; r < ROWS; ++r)
  for (uint t = 0; t < TOKENS; ++t) acc[r][t] = 0.0f;

for (uint s = lane; s < SUBS; s += 32) {
  GgmlFloats sub[ROWS];
  for (uint r = 0; r < ROWS; ++r) {
    GgmlSub decoded;
    ggml_decode_row<TYPE, INPUT>(weight, min(row0 + r, uint(OUTPUT - 1)), s, decoded);
    sub[r] = ggml_floats(decoded);
  }
  for (uint t = 0; t < TOKENS; ++t) {
    const device vec<T, 4>* x = (const device vec<T, 4>*)(input + ulong(t) * INPUT + s * 32);
    float4 xv[8];
    float2 sums = 0.0f;
    for (uint k = 0; k < 8; ++k) {
      xv[k] = float4(x[k]);
      sums[k >> 2] += xv[k].x + xv[k].y + xv[k].z + xv[k].w;
    }
    for (uint r = 0; r < ROWS; ++r) acc[r][t] += ggml_dot(sub[r], xv, sums);
  }
}

for (uint r = 0; r < ROWS; ++r) {
  const uint row = row0 + r;
  for (uint t = 0; t < TOKENS; ++t) {
    const float value = simd_sum(acc[r][t]);
    if (lane == 0 && row < OUTPUT) output[ulong(t) * OUTPUT + row] = T(value);
  }
}
