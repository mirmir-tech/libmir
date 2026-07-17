constexpr uint BN = 32;
constexpr uint BD = 32;
uint query_head = thread_position_in_grid.y;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint kv_head = query_head / (QUERY_HEADS / KV_HEADS);
thread float query[QK_PER_THREAD];
thread float accumulator[V_PER_THREAD];
threadgroup float outputs[BN * BD];
threadgroup float maximums[BN];
threadgroup float normalizers[BN];
uint context_tokens = page_dependency[0];
for (uint index = 0; index < QK_PER_THREAD; ++index) {
  uint dimension = lane * QK_PER_THREAD + index;
  query[index] =
      dimension < HEAD_DIM ? float(queries[query_head * HEAD_DIM + dimension]) : 0.0f;
}
for (uint index = 0; index < V_PER_THREAD; ++index) accumulator[index] = 0.0f;
float maximum = -INFINITY;
float normalizer = 0.0f;
for (uint token = simd_group; token < context_tokens; token += BN) {
  uint page = page_table[token / PAGE_SIZE];
  uint in_page = token % PAGE_SIZE;
  uint base =
      ((kv_head * PAGE_CAPACITY + page) * PAGE_SIZE + in_page) * HEAD_DIM;
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
      output[query_head * HEAD_DIM + dimension] = T(accumulator[index]);
    }
  }
}
