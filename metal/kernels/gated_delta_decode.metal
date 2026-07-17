uint state_index = thread_position_in_grid.z;
uint batch = state_index / HV;
uint value_head = state_index % HV;
uint key_head = value_head / (HV / HK);
constexpr int VALUES_PER_THREAD = DK / 32;
constexpr int SIMD_GROUPS = 8;
auto query_at = query + (batch * HK + key_head) * DK;
auto key_at = key + (batch * HK + key_head) * DK;
auto value_at = value + (batch * HV + value_head) * DV;
output += (batch * HV + value_head) * DV;
auto key_index = thread_index_in_simdgroup;
auto value_group = simdgroup_index_in_threadgroup;
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
float query_values[VALUES_PER_THREAD];
float key_values[VALUES_PER_THREAD];
for (int item = 0; item < VALUES_PER_THREAD; ++item) {
  auto dimension = VALUES_PER_THREAD * key_index + item;
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
if (thread_position_in_threadgroup.x == 0) {
  uint gate_index = batch * HV + value_head;
  float parameter = float(alpha[gate_index]) + float(dt_bias[value_head]);
  float softplus = max(parameter, 0.0f) + log(1.0f + exp(-abs(parameter)));
  shared_decay = exp(-exp(float(a_log[value_head])) * softplus);
  shared_update = 1.0f / (1.0f + exp(-float(beta[gate_index])));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float decay_value = shared_decay;
float update_value = shared_update;
for (uint value_index = value_group; value_index < DV; value_index += SIMD_GROUPS) {
  auto input_state = state + (state_index * DV + value_index) * DK;
  auto output_state = next_state + (state_index * DV + value_index) * DK;
  float memory[VALUES_PER_THREAD];
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    memory[item] = float(input_state[VALUES_PER_THREAD * key_index + item]);
  }
  float projection = 0.0f;
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    memory[item] *= decay_value;
    projection += memory[item] * key_values[item];
  }
  projection = simd_sum(projection);
  auto delta = (float(value_at[value_index]) - projection) * update_value;
  float result = 0.0f;
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    memory[item] += key_values[item] * delta;
    result += memory[item] * query_values[item];
  }
  result = simd_sum(result);
  if (key_index == 0) output[value_index] = InT(result);
  for (int item = 0; item < VALUES_PER_THREAD; ++item) {
    output_state[VALUES_PER_THREAD * key_index + item] = StT(memory[item]);
  }
}
