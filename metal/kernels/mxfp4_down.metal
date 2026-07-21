constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint token = thread_position_in_grid.y;
constexpr uint top_k = TOP_K;
uint groups = INTERMEDIATE / 32;
float total = 0.0f;
for (uint selected = 0; selected < top_k; ++selected) {
  uint expert = indices[token * top_k + selected];
  float sum = 0.0f;
  for (uint group = lane; group < groups; group += 32) {
    uint scale_index = (expert * HIDDEN + row) * groups + group;
    uint block_base = scale_index * 16;
    float scale = ldexp(1.0f, int(scales[scale_index]) - 127);
    uint input_base = (token * top_k + selected) * INTERMEDIATE + group * 32;
    for (uint packed = 0; packed < 16; ++packed) {
      uchar byte = blocks[block_base + packed];
      float first = float(input[input_base + packed * 2]);
      float second = float(input[input_base + packed * 2 + 1]);
      sum += (values[byte & 15] * first + values[byte >> 4] * second) * scale;
    }
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    total += (sum + float(bias[expert * HIDDEN + row])) *
        float(routing[token * top_k + selected]);
  }
}
if (lane == 0) {
  output[token * HIDDEN + row] = T(total);
}
