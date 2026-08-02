uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint token = thread_position_in_grid.y;
float total = 0.0f;
uint scale_row = SCALE_GRID != 0 ? row / OUTPUT_BLOCK : 0;
for (uint column = lane; column < INPUT_FEATURES; column += 32) {
  float activation = ACTIVATION_FP8 != 0
      ? mirmir_e4m3_to_float(uchar(input[token * INPUT_FEATURES + column]))
      : float(input[token * INPUT_FEATURES + column]);
  uchar encoded = weight[row * INPUT_FEATURES + column];
  float value = WEIGHT_E5M2 != 0
      ? mirmir_e5m2_to_float(encoded)
      : mirmir_e4m3_to_float(encoded);
  if (SCALE_GRID != 0) {
    uint scale_column = column / INPUT_BLOCK;
    value *= scales[scale_row * INPUT_GROUPS + scale_column];
  }
  total = fma(activation, value, total);
}
total = simd_sum(total);
if (lane == 0) {
  float scale = SCALE_GRID != 0 ? 1.0f : scales[row * SCALE_STRIDE];
  if (ACTIVATION_FP8 != 0) {
    scale *= input_scales[token * ACTIVATION_STRIDE];
  }
  output[token * OUTPUT_FEATURES + row] = T(fma(total, scale, float(bias[row])));
}
