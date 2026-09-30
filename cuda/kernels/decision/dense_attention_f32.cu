// Tiled bidirectional attention for every query of a padded batch.
// A block owns TILE consecutive queries of one head of one sequence and walks
// TILE-key tiles staged in shared memory; each warp keeps an online softmax
// for QUERIES_PER_WARP queries, one key per lane. Queries past the sequence
// length produce zeros; `radius == 0xffffffff` means no band.

constexpr unsigned int TILE = 32;
constexpr unsigned int WARPS = 8;
constexpr unsigned int QUERIES_PER_WARP = TILE / WARPS;
constexpr unsigned int MAX_DIM = 64;
constexpr unsigned int VALUES_PER_LANE = MAX_DIM / 32;

extern "C" __global__ void libmir_decision_dense_attention(
    const float* qkv, float* output, const unsigned int* lengths, unsigned int length,
    unsigned int heads, unsigned int head_dim, unsigned int radius, float scale) {
  __shared__ float keys[TILE][MAX_DIM + 1];
  __shared__ float values[TILE][MAX_DIM];
  __shared__ float queries[TILE][MAX_DIM];
  const unsigned int sequence = blockIdx.z;
  const unsigned int head = blockIdx.y;
  const unsigned int query_first = blockIdx.x * TILE;
  const unsigned int valid = lengths[sequence];
  const unsigned int hidden = heads * head_dim;
  const unsigned int width = 3u * hidden;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  const float* base = qkv + static_cast<size_t>(sequence) * length * width;
  float* out = output + static_cast<size_t>(sequence) * length * hidden + head * head_dim;
  for (unsigned int index = threadIdx.x; index < TILE * head_dim; index += blockDim.x) {
    const unsigned int row = index / head_dim, d = index % head_dim;
    const unsigned int query = query_first + row;
    queries[row][d] = query < valid ? base[static_cast<size_t>(query) * width + head * head_dim + d] : 0.0f;
  }
  const bool full = radius == 0xffffffffu;
  const unsigned int block_last = min(query_first + TILE, valid);
  const unsigned int key_first = full || query_first < radius ? 0u : query_first - radius;
  const unsigned int key_last = full ? valid : min(valid, block_last + radius);
  float maximum[QUERIES_PER_WARP], total[QUERIES_PER_WARP];
  float accumulator[QUERIES_PER_WARP][VALUES_PER_LANE];
  for (unsigned int q = 0; q < QUERIES_PER_WARP; ++q) {
    maximum[q] = -__int_as_float(0x7f800000);
    total[q] = 0.0f;
    for (unsigned int v = 0; v < VALUES_PER_LANE; ++v) accumulator[q][v] = 0.0f;
  }
  for (unsigned int tile = key_first; tile < key_last; tile += TILE) {
    __syncthreads();
    for (unsigned int index = threadIdx.x; index < TILE * head_dim; index += blockDim.x) {
      const unsigned int row = index / head_dim, d = index % head_dim;
      const unsigned int key = tile + row;
      const bool inside = key < key_last;
      const float* key_row = base + static_cast<size_t>(key) * width + hidden + head * head_dim;
      keys[row][d] = inside ? key_row[d] : 0.0f;
      values[row][d] = inside ? key_row[hidden + d] : 0.0f;
    }
    __syncthreads();
    const unsigned int key = tile + lane;
    for (unsigned int q = 0; q < QUERIES_PER_WARP; ++q) {
      const unsigned int row = warp * QUERIES_PER_WARP + q;
      const unsigned int query = query_first + row;
      if (query >= valid) continue;
      const bool allowed = key < key_last && (full || (key + radius >= query && key <= query + radius));
      float score = -__int_as_float(0x7f800000);
      if (allowed) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; ++d) dot = fmaf(queries[row][d], keys[lane][d], dot);
        score = dot * scale;
      }
      float tile_max = score;
      for (int offset = 16; offset > 0; offset >>= 1)
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
      if (tile_max == -__int_as_float(0x7f800000)) continue;
      const float next = fmaxf(maximum[q], tile_max);
      const float rescale = expf(maximum[q] - next);
      const float weight = allowed ? expf(score - next) : 0.0f;
      float tile_total = weight;
      for (int offset = 16; offset > 0; offset >>= 1)
        tile_total += __shfl_xor_sync(0xffffffffu, tile_total, offset);
      total[q] = total[q] * rescale + tile_total;
      for (unsigned int v = 0; v < VALUES_PER_LANE; ++v) accumulator[q][v] *= rescale;
      for (unsigned int j = 0; j < TILE; ++j) {
        const float p = __shfl_sync(0xffffffffu, weight, j);
        for (unsigned int v = 0; v < VALUES_PER_LANE; ++v) {
          const unsigned int d = lane + v * 32u;
          if (d < head_dim) accumulator[q][v] = fmaf(p, values[j][d], accumulator[q][v]);
        }
      }
      maximum[q] = next;
    }
  }
  for (unsigned int q = 0; q < QUERIES_PER_WARP; ++q) {
    const unsigned int query = query_first + warp * QUERIES_PER_WARP + q;
    if (query >= length) continue;
    for (unsigned int v = 0; v < VALUES_PER_LANE; ++v) {
      const unsigned int d = lane + v * 32u;
      if (d < head_dim)
        out[static_cast<size_t>(query) * hidden + d] = query < valid ? accumulator[q][v] / total[q] : 0.0f;
    }
  }
}
