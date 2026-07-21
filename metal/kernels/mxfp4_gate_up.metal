constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint selected = thread_position_in_grid.y;
uint token = thread_position_in_grid.z;
constexpr uint top_k = TOP_K;
uint expert = indices[token * top_k + selected];
uint groups = HIDDEN / 32;
float gate = 0.0f;
float linear = 0.0f;
for (uint group = lane; group < groups; group += 32) {
  uint scale_base = (expert * INTERMEDIATE * 2 + row * 2) * groups + group;
  uint block_base = scale_base * 16;
  float gate_scale = ldexp(1.0f, int(scales[scale_base]) - 127);
  float linear_scale = ldexp(1.0f, int(scales[scale_base + groups]) - 127);
  uint input_base = token * HIDDEN + group * 32;
  uint linear_base = block_base + groups * 16;
  for (uint packed = 0; packed < 16; ++packed) {
    uchar gate_byte = blocks[block_base + packed];
    uchar linear_byte = blocks[linear_base + packed];
    float first = float(input[input_base + packed * 2]);
    float second = float(input[input_base + packed * 2 + 1]);
    gate += (values[gate_byte & 15] * first + values[gate_byte >> 4] * second) * gate_scale;
    linear +=
        (values[linear_byte & 15] * first + values[linear_byte >> 4] * second) * linear_scale;
  }
}
gate = simd_sum(gate);
linear = simd_sum(linear);
if (lane == 0) {
  uint bias_base = (expert * INTERMEDIATE + row) * 2;
  gate = min(gate + float(bias[bias_base]), limit);
  linear = clamp(linear + float(bias[bias_base + 1]), -limit, limit);
  float activated = gate / (1.0f + exp(-1.702f * gate));
  output[(token * top_k + selected) * INTERMEDIATE + row] = T(activated * (linear + 1.0f));
}
