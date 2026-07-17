#include <metal_stdlib>
using namespace metal;

struct PagedAttentionParameters {
  uint query_heads;
  uint kv_heads;
  uint page_capacity;
  uint blocks;
  uint page_size;
  uint scale_bits;
};

template <typename T, uint HEAD_DIM>
inline void paged_attention_partial(
    const device T* queries,
    const device T* key_pages,
    const device T* value_pages,
    const device uint* page_table,
    const device uint* page_dependency,
    device T* partials,
    device float* sums,
    device float* maximums,
    const device T* barrier,
    constant PagedAttentionParameters& parameters,
    uint lane,
    uint3 local,
    uint3 group) {
  constexpr uint VALUES_PER_THREAD = HEAD_DIM / 32;
  uint kv_head = group.y;
  uint block = group.z;
  uint group_factor = parameters.query_heads / parameters.kv_heads;
  uint query_head = kv_head * group_factor + local.y;
  thread float query[VALUES_PER_THREAD];
  thread float accumulator[VALUES_PER_THREAD];
  float scale = as_type<float>(parameters.scale_bits);
  uint context_tokens = page_dependency[0];
  (void)barrier;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
    uint dimension = lane * VALUES_PER_THREAD + index;
    query[index] = scale * float(queries[query_head * HEAD_DIM + dimension]);
    accumulator[index] = 0.0f;
  }
  float maximum = -INFINITY;
  float normalizer = 0.0f;
  for (uint token = block; token < context_tokens; token += parameters.blocks) {
    uint page = page_table[token / parameters.page_size];
    uint in_page = token % parameters.page_size;
    uint base = ((kv_head * parameters.page_capacity + page) * parameters.page_size + in_page) *
        HEAD_DIM;
    float score = 0.0f;
    for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
      score += query[index] * float(key_pages[base + lane * VALUES_PER_THREAD + index]);
    }
    score = simd_sum(score);
    float next_maximum = metal::max(maximum, score);
    float factor = metal::fast::exp(maximum - next_maximum);
    float exp_score = metal::fast::exp(score - next_maximum);
    maximum = next_maximum;
    normalizer = normalizer * factor + exp_score;
    for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
      uint dimension = lane * VALUES_PER_THREAD + index;
      accumulator[index] = accumulator[index] * factor +
          exp_score * float(value_pages[base + dimension]);
    }
  }
  uint statistic = query_head * parameters.blocks + block;
  if (lane == 0) {
    sums[statistic] = normalizer;
    maximums[statistic] = maximum;
  }
  uint output = statistic * HEAD_DIM + lane * VALUES_PER_THREAD;
  for (uint index = 0; index < VALUES_PER_THREAD; ++index) {
    partials[output + index] = T(accumulator[index]);
  }
}

#define PAGED_PARTIAL_KERNEL(NAME, TYPE, HEAD_DIM) \
kernel void NAME( \
    const device TYPE* queries [[buffer(0)]], \
    const device TYPE* key_pages [[buffer(1)]], \
    const device TYPE* value_pages [[buffer(2)]], \
    const device uint* page_table [[buffer(3)]], \
    const device uint* page_dependency [[buffer(4)]], \
    device TYPE* partials [[buffer(5)]], \
    device float* sums [[buffer(6)]], \
    device float* maximums [[buffer(7)]], \
    const device TYPE* barrier [[buffer(8)]], \
    constant PagedAttentionParameters& parameters [[buffer(9)]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint3 local [[thread_position_in_threadgroup]], \
    uint3 group [[threadgroup_position_in_grid]]) { \
  paged_attention_partial<TYPE, HEAD_DIM>( \
      queries, key_pages, value_pages, page_table, page_dependency, partials, sums, maximums, \
      barrier, parameters, lane, local, group); \
}

#define INSTANTIATE_DIM(TYPE, SUFFIX, DIM) \
  PAGED_PARTIAL_KERNEL(mirmir_paged_sdpa_partial_hd ## DIM ## _ ## SUFFIX, TYPE, DIM)
#define INSTANTIATE_TYPE(TYPE, SUFFIX) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 32) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 64) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 96) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 128) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 160) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 192) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 224) \
  INSTANTIATE_DIM(TYPE, SUFFIX, 256)

INSTANTIATE_TYPE(float, f32)
INSTANTIATE_TYPE(half, f16)
INSTANTIATE_TYPE(bfloat, bf16)
