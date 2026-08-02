uint column = thread_position_in_grid.x;
uint token = thread_position_in_grid.y;
uint row = indices[token];
uint output_index = token * HIDDEN + column;
if (row >= VOCAB) {
  output[output_index] = T(0.0f);
  return;
}
uchar encoded = weight[row * HIDDEN + column];
float value = WEIGHT_E5M2 != 0
    ? mirmir_e5m2_to_float(encoded)
    : mirmir_e4m3_to_float(encoded);
if (SCALE_GRID != 0) {
  uint scale_row = row / OUTPUT_BLOCK;
  uint scale_column = column / INPUT_BLOCK;
  value *= scales[scale_row * INPUT_GROUPS + scale_column];
} else {
  value *= scales[row * SCALE_STRIDE];
}
output[output_index] = T(value);
