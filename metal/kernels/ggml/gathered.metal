// GGML GEMV over routed experts: output[a, r] = sum_c input[a / GROUP, c] *
// W[indices[a]][r, c]. The bank is `[experts * OUTPUT, row_bytes]`; each
// simdgroup owns one row of the expert assignment a selects, lane l walks
// sub-blocks l, l + 32, ... as the single-token GEMV does.
// Grid: [ceil(OUTPUT / simdgroups) * threads, assignments, 1].
constexpr uint SUBS = INPUT / 32;
const uint simdgroups = threads_per_threadgroup.x / 32;
const uint row = threadgroup_position_in_grid.x * simdgroups + simdgroup_index_in_threadgroup;
const uint assignment = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
if (row >= OUTPUT) return;
const uint expert = indices[assignment];
const bool valid = expert < EXPERTS;
const ulong matrix_row = ulong(valid ? expert : 0) * OUTPUT + row;
const device T* x_row = input + ulong(assignment / GROUP) * INPUT;

float acc = 0.0f;
for (uint s = lane; s < SUBS; s += 32) {
  GgmlSub decoded;
  ggml_decode_row<TYPE, INPUT>(weight, matrix_row, s, decoded);
  const GgmlFloats sub = ggml_floats(decoded);
  const device vec<T, 4>* x = (const device vec<T, 4>*)(x_row + s * 32);
  float4 xv[8];
  float2 sums = 0.0f;
  for (uint k = 0; k < 8; ++k) {
    xv[k] = float4(x[k]);
    sums[k >> 2] += xv[k].x + xv[k].y + xv[k].z + xv[k].w;
  }
  acc += ggml_dot(sub, xv, sums);
}
const float value = simd_sum(acc);
if (lane == 0) output[ulong(assignment) * OUTPUT + row] = T(valid ? value : 0.0f);
