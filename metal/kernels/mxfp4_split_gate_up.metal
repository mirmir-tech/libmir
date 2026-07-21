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
  uint scale_index = (expert * INTERMEDIATE + row) * groups + group;
  uint word_base = scale_index * 4;
  float gate_scale = ldexp(1.0f, int(gate_scales[scale_index]) - 127);
  float up_scale = ldexp(1.0f, int(up_scales[scale_index]) - 127);
  uint input_base = token * HIDDEN + group * 32;
  for (uint packed = 0; packed < 16; ++packed) {
    uint shift = (packed & 3) * 8;
    uchar gate_byte = uchar(gate_blocks[word_base + packed / 4] >> shift);
    uchar up_byte = uchar(up_blocks[word_base + packed / 4] >> shift);
    float first = float(input[input_base + packed * 2]);
    float second = float(input[input_base + packed * 2 + 1]);
    gate += (values[gate_byte & 15] * first + values[gate_byte >> 4] * second) * gate_scale;
    linear += (values[up_byte & 15] * first + values[up_byte >> 4] * second) * up_scale;
  }
}
gate = simd_sum(gate);
linear = simd_sum(linear);
if (lane == 0) {
  uint bias_index = expert * INTERMEDIATE + row;
  gate = min(gate + float(gate_bias[bias_index]), limit);
  linear = clamp(linear + float(up_bias[bias_index]), -limit, limit);
  float activated = gate / (1.0f + exp(-1.702f * gate));
  output[(token * top_k + selected) * INTERMEDIATE + row] = T(activated * (linear + 1.0f));
}
