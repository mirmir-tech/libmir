uint state_index = thread_position_in_grid.z;
uint batch = state_index / HV;
uint value_head = state_index % HV;
uint key_head = value_head / (HV / HK);
constexpr int VALUES_PER_THREAD = DK / 32;
auto query_at = query + (batch * STEPS * HK * DK + key_head * DK);
auto key_at = key + (batch * STEPS * HK * DK + key_head * DK);
auto value_at = value + (batch * STEPS * HV * DV + value_head * DV);
output += batch * STEPS * HV * DV + value_head * DV;
auto key_index = thread_position_in_threadgroup.x;
auto value_index = thread_position_in_grid.y;
auto input_state = state + (state_index * DV + value_index) * DK;
auto output_state = next_state + (state_index * DV + value_index) * DK;
float memory[VALUES_PER_THREAD];
for (int item = 0; item < VALUES_PER_THREAD; ++item) {
  memory[item] = float(input_state[VALUES_PER_THREAD * key_index + item]);
}
auto decay_at = decay + batch * STEPS * HV + value_head;
auto update_at = update + batch * STEPS * HV + value_head;
for (uint time = 0; time < STEPS; ++time) {
  float decay_value = decay_at[0];
  float update_value = update_at[0];
  float projection = 0.0f;
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    auto dimension = VALUES_PER_THREAD * key_index + item;
    memory[item] *= decay_value;
    projection += memory[item] * float(key_at[dimension]);
  }
  projection = simd_sum(projection);
  auto delta = (float(value_at[value_index]) - projection) * update_value;
  float result = 0.0f;
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    auto dimension = VALUES_PER_THREAD * key_index + item;
    memory[item] += float(key_at[dimension]) * delta;
    result += memory[item] * float(query_at[dimension]);
  }
  result = simd_sum(result);
  if (thread_index_in_simdgroup == 0) {
    output[value_index] = InT(result);
  }
  query_at += HK * DK;
  key_at += HK * DK;
  value_at += HV * DV;
  output += HV * DV;
  decay_at += HV;
  update_at += HV;
}
for (int item = 0; item < VALUES_PER_THREAD; ++item) {
  output_state[VALUES_PER_THREAD * key_index + item] = StT(memory[item]);
}
