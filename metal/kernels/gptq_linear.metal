uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint token = thread_position_in_grid.y;
constexpr uint packed_output = OUTPUT / 8;
constexpr uint groups = INPUT / GROUP;
float sum = 0.0f;

for (uint feature = lane; feature < INPUT; feature += 32) {
  int mapped_group = group_indices[feature];
  if (mapped_group < 0 || uint(mapped_group) >= groups) {
    continue;
  }
  uint group = uint(mapped_group);
  uint word = qweight[(feature / 8) * OUTPUT + row];
  uint value = (word >> ((feature & 7u) * 4u)) & 15u;
  uint zero_word = qzeros[group * packed_output + row / 8u];
  uint encoded_zero = (zero_word >> ((row & 7u) * 4u)) & 15u;
  uint zero = (encoded_zero + uint(LEGACY)) & 15u;
  float weight = (float(value) - float(zero)) *
      float(scales[group * OUTPUT + row]);
  sum += float(input[token * INPUT + feature]) * weight;
}

sum = simd_sum(sum);
if (lane == 0) {
  output[token * OUTPUT + row] = T(sum);
}
