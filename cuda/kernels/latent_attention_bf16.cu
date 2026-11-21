#include <cuda_bf16.h>

// Absorbed multi-head latent attention over a paged latent cache. Every cached
// row holds one token's normalized latent followed by its rotated key:
// [LIBMIR_LATENT_RANK ; LIBMIR_LATENT_ROPE]. A query row has the same layout
// (the absorbed position-free query, then the rotated query), so attention is
// multi-query attention whose values are the first LIBMIR_LATENT_RANK columns
// of the key rows.

#ifndef LIBMIR_LATENT_RANK
#error "LIBMIR_LATENT_RANK must be defined"
#endif
#ifndef LIBMIR_LATENT_ROPE
#error "LIBMIR_LATENT_ROPE must be defined"
#endif

#define LATENT_ROW (LIBMIR_LATENT_RANK + LIBMIR_LATENT_ROPE)
#define ROW_WORDS (LATENT_ROW / 2)
#define TILE_WORDS (ROW_WORDS + 1)
#define VALUE_WORDS (LIBMIR_LATENT_RANK / 64)
#define HALF_ROPE (LIBMIR_LATENT_ROPE / 2)
#define HEADS_PER_BLOCK 8
#define TILE_TOKENS 32
#define NO_SLOT 0xffffffffu
#define BLOCK_THREADS (HEADS_PER_BLOCK * 32)
#define ROW_VECTORS (LATENT_ROW / 8)
#define TILE_LOADS ((TILE_TOKENS * ROW_VECTORS + BLOCK_THREADS - 1) / BLOCK_THREADS)
#define NEGATIVE_INFINITY __int_as_float(0xff800000)

static_assert(LIBMIR_LATENT_RANK % 64 == 0, "latent rank must be a multiple of 64");
static_assert(LIBMIR_LATENT_ROPE % 8 == 0, "rotary width must be a multiple of 8");

__device__ __forceinline__ unsigned int rotary_first(unsigned int pair) {
#ifdef LIBMIR_LATENT_INTERLEAVED
  return 2u * pair;
#else
  return pair;
#endif
}

__device__ __forceinline__ unsigned int rotary_second(unsigned int pair) {
#ifdef LIBMIR_LATENT_INTERLEAVED
  return 2u * pair + 1u;
#else
  return pair + HALF_ROPE;
#endif
}

__device__ __forceinline__ void rotate(
    const __nv_bfloat16* source, __nv_bfloat16* target, unsigned int pair,
    float cosine, float sine) {
  const unsigned int first = rotary_first(pair);
  const unsigned int second = rotary_second(pair);
  const float a = __bfloat162float(source[first]);
  const float b = __bfloat162float(source[second]);
  target[first] = __float2bfloat16_rn(fmaf(-b, sine, a * cosine));
  target[second] = __float2bfloat16_rn(fmaf(a, sine, b * cosine));
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  if (lane == 0u) scratch[warp] = value;
  __syncthreads();
  const unsigned int warps = (blockDim.x + 31u) >> 5u;
  value = threadIdx.x < warps ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0u) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0u) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// One block per token: normalizes the latent, rotates the shared key and every
// head's rotary query, stores the cache row at the token's slot (none for
// `NO_SLOT`: padding, or a shared page that already holds it) and fills the
// rotary half of the attention query. The absorbed half of the query comes
// from a batched GEMM.
extern "C" __global__ void libmir_cuda_latent_prepare_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* latent,
    const __nv_bfloat16* norm_weight, const unsigned int* positions,
    const float* inverse_frequencies, const unsigned int* slot_mapping,
    __nv_bfloat16* attention_query, unsigned char* pages, unsigned int tokens,
    unsigned int heads, unsigned int nope_dim, float epsilon, float rope_scale) {
  __shared__ float scratch[32];
  const unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  const __nv_bfloat16* source = latent + token * LATENT_ROW;
  float squares = 0.0f;
  for (unsigned int d = threadIdx.x; d < LIBMIR_LATENT_RANK; d += blockDim.x) {
    const float value = __bfloat162float(source[d]);
    squares = fmaf(value, value, squares);
  }
  const float inverse_rms =
      rsqrtf(block_sum(squares, scratch) / LIBMIR_LATENT_RANK + epsilon);
  const unsigned int slot = slot_mapping[token];
  __nv_bfloat16* row =
      reinterpret_cast<__nv_bfloat16*>(pages) + size_t(slot) * LATENT_ROW;
  const float position = float(positions[token]);
  if (slot != NO_SLOT) {
    for (unsigned int d = threadIdx.x; d < LIBMIR_LATENT_RANK; d += blockDim.x) {
      row[d] = __float2bfloat16_rn(__bfloat162float(source[d]) * inverse_rms *
                                   __bfloat162float(norm_weight[d]));
    }
    for (unsigned int pair = threadIdx.x; pair < HALF_ROPE; pair += blockDim.x) {
      float sine;
      float cosine;
      sincosf(position * inverse_frequencies[pair], &sine, &cosine);
      rotate(source + LIBMIR_LATENT_RANK, row + LIBMIR_LATENT_RANK, pair,
             cosine * rope_scale, sine * rope_scale);
    }
  }
  const unsigned int query_width = nope_dim + LIBMIR_LATENT_ROPE;
  for (unsigned int index = threadIdx.x; index < heads * HALF_ROPE;
       index += blockDim.x) {
    const unsigned int head = index / HALF_ROPE;
    const unsigned int pair = index % HALF_ROPE;
    float sine;
    float cosine;
    sincosf(position * inverse_frequencies[pair], &sine, &cosine);
    const size_t query_row = size_t(token) * heads + head;
    rotate(query + query_row * query_width + nope_dim,
           attention_query + query_row * LATENT_ROW + LIBMIR_LATENT_RANK, pair,
           cosine * rope_scale, sine * rope_scale);
  }
}

__device__ __forceinline__ float2 unpack(unsigned int word) {
  __nv_bfloat162 pair;
  *reinterpret_cast<unsigned int*>(&pair) = word;
  return __bfloat1622float2(pair);
}

// Grid (head groups, queries, splits); each warp attends one head of one query
// token over its split of the tokens up to its position, sharing every
// 32-token tile of cache rows with the other heads of the block.
extern "C" __global__ void __launch_bounds__(BLOCK_THREADS)
libmir_cuda_latent_attention_split_bf16(
    const __nv_bfloat16* query, const unsigned char* pages,
    const unsigned int* block_tables, const unsigned int* query_sequences,
    const unsigned int* positions, float* partial_values,
    float* partial_statistics, unsigned int queries, unsigned int heads,
    unsigned int table_stride, unsigned int block_size, float scale) {
  __shared__ unsigned int query_words[HEADS_PER_BLOCK][ROW_WORDS];
  __shared__ unsigned int tile[TILE_TOKENS][TILE_WORDS];
  const unsigned int query_token = blockIdx.y;
  const unsigned int split = blockIdx.z;
  const unsigned int splits = gridDim.z;
  if (query_token >= queries) return;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int head = blockIdx.x * HEADS_PER_BLOCK + warp;
  const bool active = head < heads;
  const unsigned int* source_query = reinterpret_cast<const unsigned int*>(query);
  const uint4* vectors = reinterpret_cast<const uint4*>(pages);
  for (unsigned int index = threadIdx.x; index < HEADS_PER_BLOCK * ROW_WORDS;
       index += blockDim.x) {
    const unsigned int local = index / ROW_WORDS;
    const unsigned int global = blockIdx.x * HEADS_PER_BLOCK + local;
    query_words[local][index % ROW_WORDS] = global < heads
        ? source_query[(size_t(query_token) * heads + global) * ROW_WORDS +
                       index % ROW_WORDS]
        : 0u;
  }
  const unsigned int visible = positions[query_token] + 1u;
  const unsigned int* table = block_tables + size_t(query_sequences[query_token]) * table_stride;
  const unsigned int chunk = (visible + splits - 1u) / splits;
  const unsigned int begin = min(visible, split * chunk);
  const unsigned int end = min(visible, begin + chunk);
  float maximum = NEGATIVE_INFINITY;
  float denominator = 0.0f;
  float values[2 * VALUE_WORDS];
#pragma unroll
  for (unsigned int i = 0; i < 2 * VALUE_WORDS; ++i) values[i] = 0.0f;

  for (unsigned int start = begin; start < end; start += TILE_TOKENS) {
    const unsigned int count = min(TILE_TOKENS, end - start);
    __syncthreads();
    // Issue every 16-byte load of the tile before storing any of them, so
    // the block waits for memory once per tile rather than once per load.
    uint4 staged[TILE_LOADS];
#pragma unroll
    for (unsigned int load = 0; load < TILE_LOADS; ++load) {
      const unsigned int index = threadIdx.x + load * BLOCK_THREADS;
      const unsigned int local = index / ROW_VECTORS;
      if (index < TILE_TOKENS * ROW_VECTORS && local < count) {
        const unsigned int token = start + local;
        const size_t page_row =
            size_t(table[token / block_size]) * block_size + token % block_size;
        staged[load] = vectors[page_row * ROW_VECTORS + index % ROW_VECTORS];
      }
    }
#pragma unroll
    for (unsigned int load = 0; load < TILE_LOADS; ++load) {
      const unsigned int index = threadIdx.x + load * BLOCK_THREADS;
      const unsigned int local = index / ROW_VECTORS;
      if (index < TILE_TOKENS * ROW_VECTORS && local < count) {
        unsigned int* target = &tile[local][4u * (index % ROW_VECTORS)];
        target[0] = staged[load].x;
        target[1] = staged[load].y;
        target[2] = staged[load].z;
        target[3] = staged[load].w;
      }
    }
    __syncthreads();
    if (!active) continue;
    float score = NEGATIVE_INFINITY;
    if (lane < count) {
      // Four independent sums keep the FMA pipeline busy over the row.
      float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      static_assert(ROW_WORDS % 4 == 0, "cache row must hold a multiple of 8 values");
#pragma unroll 4
      for (unsigned int word = 0; word < ROW_WORDS; word += 4) {
#pragma unroll
        for (unsigned int part = 0; part < 4; ++part) {
          const float2 q = unpack(query_words[warp][word + part]);
          const float2 k = unpack(tile[lane][word + part]);
          sums[part] = fmaf(q.x, k.x, fmaf(q.y, k.y, sums[part]));
        }
      }
      score = ((sums[0] + sums[1]) + (sums[2] + sums[3])) * scale;
    }
    float tile_maximum = score;
    for (int offset = 16; offset > 0; offset >>= 1) {
      tile_maximum = fmaxf(tile_maximum, __shfl_xor_sync(0xffffffffu, tile_maximum, offset));
    }
    const float next = fmaxf(maximum, tile_maximum);
    const float alpha = expf(maximum - next);
    const float weight = lane < count ? expf(score - next) : 0.0f;
    float weight_sum = weight;
    for (int offset = 16; offset > 0; offset >>= 1) {
      weight_sum += __shfl_xor_sync(0xffffffffu, weight_sum, offset);
    }
    denominator = fmaf(denominator, alpha, weight_sum);
    maximum = next;
#pragma unroll
    for (unsigned int i = 0; i < 2 * VALUE_WORDS; ++i) values[i] *= alpha;
    for (unsigned int local = 0; local < count; ++local) {
      const float token_weight = __shfl_sync(0xffffffffu, weight, local);
#pragma unroll
      for (unsigned int i = 0; i < VALUE_WORDS; ++i) {
        const float2 v = unpack(tile[local][32u * i + lane]);
        values[2u * i] = fmaf(token_weight, v.x, values[2u * i]);
        values[2u * i + 1u] = fmaf(token_weight, v.y, values[2u * i + 1u]);
      }
    }
  }
  if (!active) return;
  const size_t row = (size_t(split) * queries + query_token) * heads + head;
  float* target = partial_values + row * LIBMIR_LATENT_RANK;
#pragma unroll
  for (unsigned int i = 0; i < VALUE_WORDS; ++i) {
    target[2u * (32u * i + lane)] = values[2u * i];
    target[2u * (32u * i + lane) + 1u] = values[2u * i + 1u];
  }
  if (lane == 0u) {
    partial_statistics[2u * row] = maximum;
    partial_statistics[2u * row + 1u] = denominator;
  }
}

// One block per (query token, head) row combines the splits into the latent
// attention output that the value up-projection expands. The first thread
// weighs the splits once; every thread then sums one latent column.
#define MAX_SPLITS 64
extern "C" __global__ void libmir_cuda_latent_attention_merge_bf16(
    const float* partial_values, const float* partial_statistics,
    __nv_bfloat16* output, unsigned int rows, unsigned int splits) {
  __shared__ float weights[MAX_SPLITS];
  __shared__ float inverse_denominator;
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  if (threadIdx.x == 0u) {
    float maximum = NEGATIVE_INFINITY;
    for (unsigned int split = 0; split < splits; ++split) {
      const size_t index = size_t(split) * rows + row;
      if (partial_statistics[2u * index + 1u] > 0.0f) {
        maximum = fmaxf(maximum, partial_statistics[2u * index]);
      }
    }
    float denominator = 0.0f;
    for (unsigned int split = 0; split < splits; ++split) {
      const size_t index = size_t(split) * rows + row;
      const float sum = partial_statistics[2u * index + 1u];
      weights[split] = sum > 0.0f ? expf(partial_statistics[2u * index] - maximum) : 0.0f;
      denominator = fmaf(sum, weights[split], denominator);
    }
    inverse_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  }
  __syncthreads();
  for (unsigned int d = threadIdx.x; d < LIBMIR_LATENT_RANK; d += blockDim.x) {
    float value = 0.0f;
    for (unsigned int split = 0; split < splits; ++split) {
      value = fmaf(weights[split],
                   partial_values[(size_t(split) * rows + row) * LIBMIR_LATENT_RANK + d], value);
    }
    output[size_t(row) * LIBMIR_LATENT_RANK + d] = __float2bfloat16_rn(value * inverse_denominator);
  }
}
