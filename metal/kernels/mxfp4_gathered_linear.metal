constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
device const uchar* packed_weight = reinterpret_cast<device const uchar*>(weight);
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint assignment = thread_position_in_grid.y;
uint matrix = indices[assignment];
if (matrix >= MATRICES) {
  if (lane == 0) output[assignment * OUTPUT + row] = T(0.0f);
  return;
}
uint groups = INPUT / 32;
uint matrix_row = matrix * OUTPUT + row;
uint input_row = assignment / SELECTIONS;
float total = 0.0f;
for (uint group = 0; group < groups; ++group) {
  uint column = group * 32 + lane;
  uchar packed = packed_weight[(matrix_row * groups + group) * 16 + lane / 2];
  uint code = lane % 2 == 0 ? packed & 15 : packed >> 4;
  float scale = ldexp(1.0f, int(scales[matrix_row * groups + group]) - 127);
  total = fma(float(input[input_row * INPUT + column]), values[code] * scale, total);
}
total = simd_sum(total);
if (lane == 0) output[assignment * OUTPUT + row] = T(total + float(bias[matrix_row]));
