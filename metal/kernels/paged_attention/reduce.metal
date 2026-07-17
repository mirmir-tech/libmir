constexpr uint SIMD_GROUPS = 32;
uint lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint query_head = threadgroup_position_in_grid.y;
constexpr uint VALUES_PER_THREAD = HEAD_DIM / SIMD_GROUPS;
thread float accumulator[VALUES_PER_THREAD];
threadgroup float outputs[SIMD_GROUPS * SIMD_GROUPS];
uint statistic = query_head * BLOCKS;
float maximum = -INFINITY;
for (uint chunk = 0; chunk < BLOCKS / SIMD_GROUPS; ++chunk) {
  maximum = metal::max(maximum, maximums[statistic + lane + SIMD_GROUPS * chunk]);
}
maximum = simd_max(maximum);
float normalizer = 0.0f;
for (uint chunk = 0; chunk < BLOCKS / SIMD_GROUPS; ++chunk) {
  uint block = lane + SIMD_GROUPS * chunk;
  float factor = metal::fast::exp(maximums[statistic + block] - maximum);
  normalizer += factor * sums[statistic + block];
}
normalizer = simd_sum(normalizer);
for (uint index = 0; index < VALUES_PER_THREAD; ++index) accumulator[index] = 0.0f;
for (uint chunk = 0; chunk < BLOCKS / SIMD_GROUPS; ++chunk) {
  uint block = simd_group + SIMD_GROUPS * chunk;
  float factor = metal::fast::exp(maximums[statistic + block] - maximum);
  uint partial = (statistic + block) * HEAD_DIM + lane * VALUES_PER_THREAD;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
    accumulator[index] += factor * float(partials[partial + index]);
  }
}
for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
  outputs[lane * SIMD_GROUPS + simd_group] = accumulator[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  accumulator[index] = simd_sum(outputs[simd_group * SIMD_GROUPS + lane]);
  accumulator[index] = normalizer == 0.0f ? accumulator[index] : accumulator[index] / normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) {
  uint output_base = query_head * HEAD_DIM + simd_group * VALUES_PER_THREAD;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
    output[output_base + index] = T(accumulator[index]);
  }
}
