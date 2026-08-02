device const uchar* packed_weight = reinterpret_cast<device const uchar*>(weight);
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint token = thread_position_in_grid.y;
float total = 0.0f;
for (uint column = lane; column < INPUT; column += 32u) {
  uint element = row * INPUT + column;
  uchar packed = packed_weight[element >> 1u];
  uint code = (element & 1u) == 0u ? packed >> 4u : packed & 15u;
  uint block = element / BLOCK;
  float scale;
  if constexpr (NESTED == 0) {
    scale = float(absmax[block]);
  } else {
    float offset = as_type<float>(uint(OFFSET_BITS));
    scale = nested_quant_map[uchar(absmax[block])] * nested_absmax[block / uint(NESTED)] + offset;
  }
  total = fma(float(input[token * INPUT + column]), quant_map[code] * scale, total);
}
total = simd_sum(total);
if (lane == 0) {
  output[token * OUTPUT + row] = T(total);
}
