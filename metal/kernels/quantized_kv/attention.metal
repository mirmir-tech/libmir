constexpr uint BN = 32;
constexpr uint BD = 32;
uint query_head = thread_position_in_grid.y;
uint query_token = thread_position_in_grid.z;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint kv_head = query_head / (QUERY_HEADS / KV_HEADS);
thread float query[QK_PER_THREAD];
thread float accumulator[V_PER_THREAD];
threadgroup float outputs[BN * BD];
threadgroup float maximums[BN];
threadgroup float normalizers[BN];
uint context_tokens = page_dependency[0];
uint visible_tokens = context_tokens - QUERY_TOKENS + query_token + 1;
uint query_base = (query_head * QUERY_TOKENS + query_token) * HEAD_DIM;
for (uint index = 0; index < QK_PER_THREAD; ++index) {
  uint dimension = lane * QK_PER_THREAD + index;
  query[index] = dimension < HEAD_DIM ? float(queries[query_base + dimension]) : 0.0f;
}
for (uint index = 0; index < V_PER_THREAD; ++index) accumulator[index] = 0.0f;
float maximum = -INFINITY;
float normalizer = 0.0f;
for (uint token = simd_group; token < visible_tokens; token += BN) {
  uint page = page_table[token / PAGE_SIZE];
  uint in_page = token % PAGE_SIZE;
  uint scale_index = (kv_head * PAGE_CAPACITY + page) * PAGE_SIZE + in_page;
  uint packed_base = scale_index * PACKED_DIM;
  float key_scale = key_scales[scale_index];
  float value_scale = value_scales[scale_index];
  float score = 0.0f;
  for (uint index = 0; index < QK_PER_THREAD; ++index) {
    uint dimension = lane * QK_PER_THREAD + index;
    if (dimension < HEAD_DIM) {
      uint word = dimension / 4;
      uint shift = (dimension % 4) * 8;
      int encoded = int((key_pages[packed_base + word] >> shift) & 0xffu);
      int quantized = encoded >= 128 ? encoded - 256 : encoded;
      score += query[index] * float(quantized) * key_scale;
    }
  }
  score = simd_sum(score) * float(attention_scale);
  float next_maximum = metal::max(maximum, score);
  float factor = metal::fast::exp(maximum - next_maximum);
  float exp_score = metal::fast::exp(score - next_maximum);
  maximum = next_maximum;
  normalizer = normalizer * factor + exp_score;
  for (uint index = 0; index < V_PER_THREAD; ++index) {
    uint dimension = lane * V_PER_THREAD + index;
    if (dimension < HEAD_DIM) {
      uint word = dimension / 4;
      uint shift = (dimension % 4) * 8;
      int encoded = int((value_pages[packed_base + word] >> shift) & 0xffu);
      int quantized = encoded >= 128 ? encoded - 256 : encoded;
      accumulator[index] =
          accumulator[index] * factor + exp_score * float(quantized) * value_scale;
    } else {
      accumulator[index] = 0.0f;
    }
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
  accumulator[index] = normalizer == 0.0f ? accumulator[index] : accumulator[index] / normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) {
  for (uint index = 0; index < V_PER_THREAD; ++index) {
    uint dimension = simd_group * V_PER_THREAD + index;
    if (dimension < HEAD_DIM) output[query_base + dimension] = T(accumulator[index]);
  }
}
