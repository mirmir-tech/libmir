// Copyright © 2026 Apple Inc.
// Adapted from MLX-LM e9308d7 (PR #1559); see LICENSE-MLX-LM.txt.
constexpr int lanes_per_row = 4;
constexpr int rows_per_simdgroup = 32 / lanes_per_row;
constexpr int values_per_lane = DK / lanes_per_row;
constexpr int partials_per_lane = values_per_lane / 4;

auto n = thread_position_in_grid.z;
auto b_idx = n / HV;
auto hv_idx = n % HV;
auto hk_idx = hv_idx / (HV / HK);

auto lane = thread_index_in_simdgroup;
auto row_in_simdgroup = lane / lanes_per_row;
auto lane_in_row = lane & (lanes_per_row - 1);
auto row_group = thread_position_in_grid.y;
auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

// query, key: [B, STEPS, HK, DK]
auto q_ = query + (b_idx * STEPS * HK + hk_idx) * DK + lane_in_row * values_per_lane;
auto k_ = key + (b_idx * STEPS * HK + hk_idx) * DK + lane_in_row * values_per_lane;

// value, output: [B, STEPS, HV, DV]
auto v_ = value + (b_idx * STEPS * HV + hv_idx) * DV;
output += (b_idx * STEPS * HV + hv_idx) * DV;

// state, next_state: [B, HV, DV, DK]
auto i_state = state + (n * DV + dv_idx) * DK + lane_in_row * values_per_lane;
auto o_state = next_state + (n * DV + dv_idx) * DK + lane_in_row * values_per_lane;

float memory[values_per_lane];
for (int i = 0; i < values_per_lane; ++i) {
  memory[i] = static_cast<float>(i_state[i]);
}

// decay, update: [B, STEPS, HV]
auto g_ = decay + b_idx * STEPS * HV;
auto beta_ = update + b_idx * STEPS * HV;

for (int t = 0; t < STEPS; ++t) {
  float gt = static_cast<float>(g_[hv_idx]);

  // Partials mirror the generic kernel: each 4-element chain is one
  // original lane's sequential accumulation.
  float part[partials_per_lane];
  for (int pb = 0; pb < partials_per_lane; ++pb) {
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      int e = pb * 4 + i;
      memory[e] = memory[e] * gt;
      acc += memory[e] * static_cast<float>(k_[e]);
    }
    part[pb] = acc;
  }
  // Butterfly levels xor 1,2,4 stay inside this lane (commutative
  // pairwise tree); levels xor 8,16 become the row-group shuffles.
  float kv_mem =
      ((part[0] + part[1]) + (part[2] + part[3])) +
      ((part[4] + part[5]) + (part[6] + part[7]));
  kv_mem += simd_shuffle_xor(kv_mem, 1);
  kv_mem += simd_shuffle_xor(kv_mem, 2);

  auto delta =
      (static_cast<float>(v_[dv_idx]) - kv_mem) *
      static_cast<float>(beta_[hv_idx]);

  for (int pb = 0; pb < partials_per_lane; ++pb) {
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
      int e = pb * 4 + i;
      memory[e] = memory[e] + static_cast<float>(k_[e]) * delta;
      acc += memory[e] * static_cast<float>(q_[e]);
    }
    part[pb] = acc;
  }
  float out =
      ((part[0] + part[1]) + (part[2] + part[3])) +
      ((part[4] + part[5]) + (part[6] + part[7]));
  out += simd_shuffle_xor(out, 1);
  out += simd_shuffle_xor(out, 2);
  if (lane_in_row == 0) {
    output[dv_idx] = static_cast<InT>(out);
  }

  q_ += HK * DK;
  k_ += HK * DK;
  v_ += HV * DV;
  output += HV * DV;
  g_ += HV;
  beta_ += HV;
}

for (int i = 0; i < values_per_lane; ++i) {
  o_state[i] = static_cast<StT>(memory[i]);
}
