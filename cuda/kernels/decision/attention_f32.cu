// Bidirectional attention over a fused `[batch, length, 3 × heads × dim]`
// projection. One warp evaluates one listed query of one head with an online
// softmax; keys form the contiguous range `[query - radius, query + radius]`
// clipped to the valid tokens, or every valid token when `radius` is
// `0xffffffff`. Queries past their sequence length produce zeros.

constexpr unsigned int MAX_LANE_VALUES = 4;  // head_dim <= 128

extern "C" __global__ void libmir_decision_attention(
    const float* qkv, float* output, const unsigned int* queries,
    const unsigned int* lengths, unsigned int query_count, unsigned int length,
    unsigned int heads, unsigned int head_dim, unsigned int radius, float scale) {
  const unsigned int listed = blockIdx.x;
  const unsigned int head = blockIdx.y;
  const unsigned int lane = threadIdx.x;
  if (listed >= query_count) return;
  const unsigned int flat = queries[listed];
  const unsigned int sequence = flat / length;
  const unsigned int query = flat % length;
  const unsigned int valid = lengths[sequence];
  const unsigned int hidden = heads * head_dim;
  const unsigned int width = 3u * hidden;
  float* target = output + static_cast<size_t>(listed) * hidden + head * head_dim;
  if (query >= valid) {
    for (unsigned int d = lane; d < head_dim; d += 32u) target[d] = 0.0f;
    return;
  }
  const float* base = qkv + static_cast<size_t>(sequence) * length * width;
  float q[MAX_LANE_VALUES];
  float accumulator[MAX_LANE_VALUES];
  for (unsigned int slot = 0; slot < MAX_LANE_VALUES; ++slot) {
    const unsigned int d = lane + slot * 32u;
    q[slot] = d < head_dim ? base[static_cast<size_t>(query) * width + head * head_dim + d] : 0.0f;
    accumulator[slot] = 0.0f;
  }
  const bool full = radius == 0xffffffffu;
  const unsigned int first = full || query < radius ? 0u : query - radius;
  const unsigned int last = full ? valid : min(valid, query + radius + 1u);
  float maximum = -__int_as_float(0x7f800000);  // -inf without <cmath>
  float total = 0.0f;
  for (unsigned int key = first; key < last; ++key) {
    const float* key_row = base + static_cast<size_t>(key) * width + hidden + head * head_dim;
    float dot = 0.0f;
    for (unsigned int slot = 0; slot < MAX_LANE_VALUES; ++slot) {
      const unsigned int d = lane + slot * 32u;
      if (d < head_dim) dot = fmaf(q[slot], key_row[d], dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, offset);
    const float score = dot * scale;
    const float next = fmaxf(maximum, score);
    const float rescale = expf(maximum - next);
    const float weight = expf(score - next);
    total = total * rescale + weight;
    const float* value_row = key_row + hidden;
    for (unsigned int slot = 0; slot < MAX_LANE_VALUES; ++slot) {
      const unsigned int d = lane + slot * 32u;
      if (d < head_dim) accumulator[slot] = fmaf(accumulator[slot], rescale, weight * value_row[d]);
    }
    maximum = next;
  }
  for (unsigned int slot = 0; slot < MAX_LANE_VALUES; ++slot) {
    const unsigned int d = lane + slot * 32u;
    if (d < head_dim) target[d] = accumulator[slot] / total;
  }
}
