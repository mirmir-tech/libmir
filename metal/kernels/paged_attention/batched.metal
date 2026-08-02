constexpr uint BN = 32;
constexpr uint BD = 32;
uint query_head = thread_position_in_grid.y;
uint batch = thread_position_in_grid.z;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint kv_head = query_head / (QUERY_HEADS / KV_HEADS);
const device T* key_pages = key_pages_0;
const device T* value_pages = value_pages_0;
switch (batch) {
  case 1: key_pages = key_pages_1; value_pages = value_pages_1; break;
  case 2: key_pages = key_pages_2; value_pages = value_pages_2; break;
  case 3: key_pages = key_pages_3; value_pages = value_pages_3; break;
  case 4: key_pages = key_pages_4; value_pages = value_pages_4; break;
  case 5: key_pages = key_pages_5; value_pages = value_pages_5; break;
  case 6: key_pages = key_pages_6; value_pages = value_pages_6; break;
  case 7: key_pages = key_pages_7; value_pages = value_pages_7; break;
}
thread float query[QK_PER_THREAD];
thread float accumulator[V_PER_THREAD];
threadgroup float outputs[BN * BD];
threadgroup float maximums[BN];
threadgroup float normalizers[BN];
for (uint index = 0; index < QK_PER_THREAD; ++index) {
  uint dimension = lane * QK_PER_THREAD + index;
  query[index] = dimension < HEAD_DIM
      ? float(queries[(batch * QUERY_HEADS + query_head) * HEAD_DIM + dimension])
      : 0.0f;
}
for (uint index = 0; index < V_PER_THREAD; ++index) accumulator[index] = 0.0f;
float maximum = -INFINITY;
float normalizer = 0.0f;
uint context_tokens = page_dependencies[batch];
uint page_capacity = page_capacities[batch];
for (uint token = simd_group; token < context_tokens; token += BN) {
  uint page = page_tables[batch * PAGE_TABLE_CAPACITY + token / PAGE_SIZE];
  uint in_page = token % PAGE_SIZE;
  uint base = ((kv_head * page_capacity + page) * PAGE_SIZE + in_page) * HEAD_DIM;
  float score = 0.0f;
  for (uint index = 0; index < QK_PER_THREAD; ++index) {
    uint dimension = lane * QK_PER_THREAD + index;
    score += dimension < HEAD_DIM
        ? query[index] * float(key_pages[base + dimension])
        : 0.0f;
  }
  score = simd_sum(score) * float(attention_scale);
  float next_maximum = metal::max(maximum, score);
  float factor = metal::fast::exp(maximum - next_maximum);
  float exp_score = metal::fast::exp(score - next_maximum);
  maximum = next_maximum;
  normalizer = normalizer * factor + exp_score;
  for (uint index = 0; index < V_PER_THREAD; ++index) {
    uint dimension = lane * V_PER_THREAD + index;
    accumulator[index] = dimension < HEAD_DIM
        ? accumulator[index] * factor + exp_score * float(value_pages[base + dimension])
        : 0.0f;
  }
}
if (lane == 0) {
  maximums[simd_group] = maximum;
  normalizers[simd_group] = normalizer;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
maximum = maximums[lane];
float next_maximum = simd_max(maximum);
float factor = metal::fast::exp(maximum - next_maximum);
normalizer = simd_sum(normalizers[lane] * factor);
for (uint index = 0; index < V_PER_THREAD; ++index) {
  outputs[lane * BD + simd_group] = accumulator[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  accumulator[index] = simd_sum(outputs[simd_group * BD + lane] * factor);
  accumulator[index] =
      normalizer == 0.0f ? accumulator[index] : accumulator[index] / normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) {
  for (uint index = 0; index < V_PER_THREAD; ++index) {
    uint dimension = simd_group * V_PER_THREAD + index;
    if (dimension < HEAD_DIM) {
      output[(batch * QUERY_HEADS + query_head) * HEAD_DIM + dimension] =
          T(accumulator[index]);
    }
  }
}
