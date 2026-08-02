constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint assignment = thread_position_in_grid.y;
uint matrix = indices[assignment];
if (matrix >= MATRICES) {
  if (lane == 0) output[assignment * OUTPUT + row] = T(0.0f);
  return;
}
uint matrix_row = matrix * OUTPUT + row;
uint input_row = assignment / SELECTIONS;
float total = 0.0f;
for (uint column = lane; column < INPUT; column += 32) {
  uint source_index = matrix_row * INPUT + column;
  uchar pair = weight[source_index / 2];
  uint code = (source_index & 1u) == 0u ? pair & 15u : pair >> 4;
  uint global_index = PER_MATRIX_GLOBAL ? matrix : 0;
  float scale =
      mirmir_e4m3_to_float(scales[source_index / 16]) * global_scale[global_index];
  total = fma(float(input[input_row * INPUT + column]), values[code] * scale, total);
}
total = simd_sum(total);
if (lane == 0) output[assignment * OUTPUT + row] = T(total);
