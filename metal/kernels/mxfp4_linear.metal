constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
device const uchar* packed_weight = reinterpret_cast<device const uchar*>(weight);
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint token = thread_position_in_grid.y;
uint groups = INPUT / 32;
float total = 0.0f;
for (uint group = 0; group < groups; ++group) {
  uint column = group * 32 + lane;
  uchar packed = packed_weight[(row * groups + group) * 16 + lane / 2];
  uint code = lane % 2 == 0 ? packed & 15 : packed >> 4;
  float scale = ldexp(1.0f, int(scales[row * groups + group]) - 127);
  total = fma(float(input[token * INPUT + column]), values[code] * scale, total);
}
total = simd_sum(total);
if (lane == 0) {
  output[token * OUTPUT + row] = T(total + float(bias[row]));
}
