constexpr uint DIMENSION_GROUPS = 32;
constexpr uint SIMD_GROUPS = REDUCTION_GROUPS;
uint lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint query_head = threadgroup_position_in_grid.y;
constexpr uint VALUES_PER_THREAD = HEAD_DIM / DIMENSION_GROUPS;
thread float accumulator[VALUES_PER_THREAD];
threadgroup float outputs[DIMENSION_GROUPS * SIMD_GROUPS];
uint statistic = query_head * BLOCKS;
float maximum = -INFINITY;
for (uint chunk = 0; chunk < BLOCKS / DIMENSION_GROUPS; ++chunk) {
  maximum = metal::max(maximum, maximums[statistic + lane + DIMENSION_GROUPS * chunk]);
}
maximum = simd_max(maximum);
float normalizer = 0.0f;
for (uint chunk = 0; chunk < BLOCKS / DIMENSION_GROUPS; ++chunk) {
  uint block = lane + DIMENSION_GROUPS * chunk;
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
  for (uint dimension_group = simd_group;
       dimension_group < DIMENSION_GROUPS;
       dimension_group += SIMD_GROUPS) {
    float value =
        lane < SIMD_GROUPS ? outputs[dimension_group * SIMD_GROUPS + lane] : 0.0f;
    value = simd_sum(value);
    if (lane == 0) {
      uint output_base =
          query_head * HEAD_DIM + dimension_group * VALUES_PER_THREAD;
      output[output_base + index] =
          T(normalizer == 0.0f ? value : value / normalizer);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
