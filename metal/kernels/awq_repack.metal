uint packed = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
constexpr uint words = INPUT / 8;
constexpr uint groups = INPUT / GROUP;
uint lane = row & 7u;
uint shift = (((lane & 1u) << 2u) | (lane >> 1u)) * 4u;

if (packed < words) {
  uint word = 0u;
  uint feature = packed * 8u;
  uint source_column = row / 8u;
  constexpr uint packed_output = OUTPUT / 8;
  for (uint index = 0; index < 8; ++index) {
    uint source = qweight[(feature + index) * packed_output + source_column];
    word |= ((source >> shift) & 15u) << (index * 4u);
  }
  weight[row * words + packed] = word;
}

if (packed < groups) {
  constexpr uint packed_output = OUTPUT / 8;
  uint source = packed * packed_output + row / 8u;
  uint zero = (qzeros[source] >> shift) & 15u;
  half scale = scales[packed * OUTPUT + row];
  uint target = row * groups + packed;
  native_scales[target] = scale;
  biases[target] = half(-float(zero) * float(scale));
}
