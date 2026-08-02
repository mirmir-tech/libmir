uint packed = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
constexpr uint words = INPUT / 8;
constexpr uint groups = INPUT / GROUP;

if (packed < words) {
  weight[row * words + packed] = qweight[packed * OUTPUT + row];
}

if (packed < groups) {
  constexpr uint packed_output = OUTPUT / 8;
  uint shift = (row & 7u) * 4u;
  uint encoded = (qzeros[packed * packed_output + row / 8u] >> shift) & 15u;
  uint zero = (encoded + uint(LEGACY)) & 15u;
  half scale = scales[packed * OUTPUT + row];
  uint target = row * groups + packed;
  native_scales[target] = scale;
  biases[target] = half(-float(zero) * float(scale));
}
