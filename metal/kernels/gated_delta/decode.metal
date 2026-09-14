// Four-lane recurrence tree adapted from MLX-LM PR #1559.
// Copyright © 2026 Apple Inc.; see LICENSE-MLX-LM.txt.
// Keep our fused normalization and gate arithmetic unchanged.
uint state_index = thread_position_in_grid.z;
uint batch = state_index / HV;
uint value_head = state_index % HV;
uint key_head = value_head / (HV / HK);
constexpr int VALUES_PER_THREAD = DK / 32;

auto query_at = query + (batch * HK + key_head) * DK;
auto key_at = key + (batch * HK + key_head) * DK;
auto value_at = value + (batch * HV + value_head) * DV;
output += (batch * HV + value_head) * DV;
auto key_index = thread_index_in_simdgroup;

float inverse = 1.0f / sqrt(float(DK));
float query_norm = 1.0f;
float key_norm = 1.0f;
if constexpr (NORMALIZE) {
  float query_squares = 0.0f;
  float key_squares = 0.0f;
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    auto dimension = VALUES_PER_THREAD * key_index + item;
    float query_value = float(query_at[dimension]);
    float key_value = float(key_at[dimension]);
    query_squares += query_value * query_value;
    key_squares += key_value * key_value;
  }
  query_squares = simd_sum(query_squares);
  key_squares = simd_sum(key_squares);
  query_norm = metal::precise::rsqrt(query_squares / float(DK) + 1.0e-6f);
  key_norm = metal::precise::rsqrt(key_squares / float(DK) + 1.0e-6f);
}
float query_values[32];
float key_values[32];
for (int item = 0; item < 32; ++item) {
  auto dimension = 32 * (key_index % 4) + item;
  query_values[item] = float(query_at[dimension]);
  key_values[item] = float(key_at[dimension]);
  if constexpr (NORMALIZE) {
    query_values[item] =
        float(InT(InT(inverse * inverse) * InT(query_values[item] * query_norm)));
    key_values[item] = float(InT(InT(inverse) * InT(key_values[item] * key_norm)));
  }
}
threadgroup float shared_decay;
threadgroup float shared_update;
if (thread_position_in_threadgroup.x == 0 && thread_position_in_threadgroup.y == 0) {
  uint gate_index = batch * HV + value_head;
  float parameter = float(alpha[gate_index]) + float(dt_bias[value_head]);
  float softplus = max(parameter, 0.0f) + log(1.0f + exp(-abs(parameter)));
  shared_decay = exp(-exp(float(a_log[value_head])) * softplus);
  shared_update = 1.0f / (1.0f + exp(-float(beta[gate_index])));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float decay_value = shared_decay;
float update_value = shared_update;
uint value_index = thread_position_in_grid.y * 8 + key_index / 4;
uint offset = (state_index * DV + value_index) * DK + (key_index % 4) * 32;
float memory[32];
float partials[8];
for (int part = 0; part < 8; ++part) {
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    int e = part * 4 + i;
    memory[e] = float(state[offset + e]) * decay_value;
    sum += memory[e] * key_values[e];
  }
  partials[part] = sum;
}
float projection = ((partials[0] + partials[1]) + (partials[2] + partials[3])) +
                   ((partials[4] + partials[5]) + (partials[6] + partials[7]));
projection += simd_shuffle_xor(projection, 1);
projection += simd_shuffle_xor(projection, 2);
float delta = (float(value_at[value_index]) - projection) * update_value;
for (int part = 0; part < 8; ++part) {
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    int e = part * 4 + i;
    memory[e] += key_values[e] * delta;
    sum += memory[e] * query_values[e];
    next_state[offset + e] = StT(memory[e]);
  }
  partials[part] = sum;
}
float result = ((partials[0] + partials[1]) + (partials[2] + partials[3])) +
               ((partials[4] + partials[5]) + (partials[6] + partials[7]));
result += simd_shuffle_xor(result, 1);
result += simd_shuffle_xor(result, 2);
if (key_index % 4 == 0) output[value_index] = InT(result);
