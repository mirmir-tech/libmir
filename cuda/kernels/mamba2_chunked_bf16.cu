#include <cuda_bf16.h>
#include <mma.h>

// Chunked structured state-space duality (SSD) prefill for Mamba-2.
//
// Within a chunk of L tokens, with a_s = dt_s * A and the inclusive prefix
// sum Λ_t = Σ_{s≤t} a_s:
//   y_t = Σ_{s≤t} (C_t·B_s) exp(Λ_t − Λ_s) dt_s x_s      (intra-chunk)
//       + exp(Λ_t) C_t · S_prev                          (carried state)
//       + D x_t
//   S_next = exp(Λ_{L−1}) S_prev + Σ_s exp(Λ_{L−1} − Λ_s) dt_s x_s ⊗ B_s
// Matrix products run on BF16 tensor cores with FP32 accumulation. Operands
// are rounded to BF16 at the same points as the reference Triton kernels:
// the weighted B, the decayed C·Bᵀ scores, and the carried state.

using namespace nvcuda;

#define LIBMIR_MAMBA2_CHUNK 64u
#define LIBMIR_MAMBA2_TILE 16u
#define LIBMIR_MAMBA2_PAD 8u

typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> LibmirMamba2Accumulator;

__device__ __forceinline__ float libmir_mamba2_chunk_softplus(float value) {
  return value > 20.0f ? value : log1pf(__expf(value));
}

// Chunked scan: one block per head walks every chunk in order and keeps
// the FP32 state on chip, so chunk states never round-trip through global
// memory. The next chunk's operands are fetched into registers while the
// current chunk updates the state.
// Grid: (heads); block: 512 threads (16 warps).
// Limits: head_dim <= 64, state_size <= 128, both multiples of 16,
// 2 * head_dim <= state_size.
// Dynamic shared memory, in bytes (each region 32-byte aligned):
//   R1 bf16 [L][state + PAD]       C → C·exp(Λ_t) → f32 y [L][head_dim + 4]
//   R2 bf16 [max(L, head_dim)][state + PAD]   B → S bf16 → weighted B
//   X  bf16 [L][head_dim + PAD]
//   R3 f32  [L][L + 4]              C·Bᵀ → bf16 G [L][L + PAD]
//   S  f32  [head_dim][state + 4]
//   Λ, dt f32 [L] each
#define LIBMIR_MAMBA2_SCAN_THREADS 512u

struct LibmirMamba2Prefetch {
  uint4 c[2];
  uint4 b[2];
  uint4 x;
  float dt;
};

__device__ __forceinline__ void libmir_mamba2_prefetch(
    LibmirMamba2Prefetch& fetched, const __nv_bfloat16* __restrict__ convolved,
    const __nv_bfloat16* __restrict__ projection, unsigned int begin,
    unsigned int tokens, unsigned int convolved_width, unsigned int inner,
    unsigned int head, unsigned int head_dim, unsigned int state_size,
    unsigned int groups, unsigned int group, unsigned int projection_stride,
    unsigned int dt_offset) {
  const unsigned int vectors = state_size / 8u;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (unsigned int item = 0; item < 2u; ++item) {
    const unsigned int index = threadIdx.x + item * LIBMIR_MAMBA2_SCAN_THREADS;
    const unsigned int token = index / vectors;
    const unsigned int column = (index % vectors) * 8u;
    const bool live = token < LIBMIR_MAMBA2_CHUNK && begin + token < tokens;
    const __nv_bfloat16* row = convolved +
        static_cast<size_t>(begin + token) * convolved_width + inner;
    fetched.c[item] = live
        ? *reinterpret_cast<const uint4*>(row + (groups + group) * state_size + column)
        : zero;
    fetched.b[item] = live
        ? *reinterpret_cast<const uint4*>(row + group * state_size + column)
        : zero;
  }
  const unsigned int x_vectors = head_dim / 8u;
  const unsigned int token = threadIdx.x / x_vectors;
  const bool live = token < LIBMIR_MAMBA2_CHUNK && begin + token < tokens;
  fetched.x = live
      ? *reinterpret_cast<const uint4*>(convolved +
            static_cast<size_t>(begin + token) * convolved_width + head * head_dim +
            (threadIdx.x % x_vectors) * 8u)
      : zero;
  fetched.dt = threadIdx.x < LIBMIR_MAMBA2_CHUNK && begin + threadIdx.x < tokens
      ? __bfloat162float(projection[static_cast<size_t>(begin + threadIdx.x) * projection_stride +
                                    dt_offset + head])
      : 0.0f;
}

extern "C" __global__ void __launch_bounds__(LIBMIR_MAMBA2_SCAN_THREADS)
libmir_cuda_mamba2_chunk_scan_bf16(
    const __nv_bfloat16* __restrict__ convolved,
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ skip,
    const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ state,
    __nv_bfloat16* __restrict__ output, unsigned int tokens,
    unsigned int heads, unsigned int head_dim, unsigned int state_size,
    unsigned int groups, unsigned int projection_stride,
    unsigned int dt_offset, float dt_minimum, float dt_maximum) {
  constexpr unsigned int chunk = LIBMIR_MAMBA2_CHUNK;
  constexpr unsigned int tile = LIBMIR_MAMBA2_TILE;
  constexpr unsigned int threads = LIBMIR_MAMBA2_SCAN_THREADS;
  extern __shared__ __align__(32) unsigned char raw[];
  const unsigned int wide = state_size + LIBMIR_MAMBA2_PAD;
  const unsigned int x_stride = head_dim + LIBMIR_MAMBA2_PAD;
  const unsigned int score_stride = chunk + LIBMIR_MAMBA2_PAD;
  const unsigned int product_stride = chunk + 4u;
  const unsigned int y_stride = head_dim + 4u;
  const unsigned int state_stride = state_size + 4u;
  __nv_bfloat16* first = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* second = first + chunk * wide;
  __nv_bfloat16* x_values = second + max(chunk, head_dim) * wide;
  float* products = reinterpret_cast<float*>(x_values + chunk * x_stride);
  float* carried = products + chunk * product_stride;
  float* sums = carried + head_dim * state_stride;
  float* steps = sums + chunk;
  __nv_bfloat16* scores = reinterpret_cast<__nv_bfloat16*>(products);
  float* y_values = reinterpret_cast<float*>(first);

  const unsigned int head = blockIdx.x;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warps = threads / 32u;
  const unsigned int inner = heads * head_dim;
  const unsigned int convolved_width = inner + 2u * groups * state_size;
  const unsigned int group = head / (heads / groups);
  const unsigned int chunk_tiles = chunk / tile;
  const unsigned int row_tiles = head_dim / tile;
  const unsigned int state_tiles = state_size / tile;
  const unsigned int vectors = state_size / 8u;
  const unsigned int x_vectors = head_dim / 8u;
  const float rate = -__expf(__bfloat162float(a_log[head]));
  const float bias = __bfloat162float(dt_bias[head]);
  const float skip_scale = __bfloat162float(skip[head]);
  float* head_state = state + static_cast<size_t>(head) * head_dim * state_size;
  for (unsigned int index = threadIdx.x; index < head_dim * state_size; index += threads) {
    carried[(index / state_size) * state_stride + index % state_size] = head_state[index];
  }
  LibmirMamba2Prefetch fetched;
  libmir_mamba2_prefetch(fetched, convolved, projection, 0u, tokens, convolved_width, inner,
                         head, head_dim, state_size, groups, group, projection_stride,
                         dt_offset);

  for (unsigned int begin = 0; begin < tokens; begin += chunk) {
    const unsigned int length = min(chunk, tokens - begin);
    // Publish the prefetched chunk; keep raw B in registers for the update.
    uint4 raw_b[2];
#pragma unroll
    for (unsigned int item = 0; item < 2u; ++item) {
      const unsigned int index = threadIdx.x + item * threads;
      const unsigned int token = index / vectors;
      raw_b[item] = fetched.b[item];
      if (token < chunk) {
        const unsigned int column = (index % vectors) * 8u;
        *reinterpret_cast<uint4*>(first + token * wide + column) = fetched.c[item];
        *reinterpret_cast<uint4*>(second + token * wide + column) = fetched.b[item];
      }
    }
    if (threadIdx.x < chunk * x_vectors) {
      *reinterpret_cast<uint4*>(x_values + (threadIdx.x / x_vectors) * x_stride +
                                (threadIdx.x % x_vectors) * 8u) = fetched.x;
    }
    // Steps and their inclusive prefix sum within the first two warps.
    if (threadIdx.x < chunk) {
      const float dt = threadIdx.x < length
          ? fminf(fmaxf(libmir_mamba2_chunk_softplus(fetched.dt + bias), dt_minimum), dt_maximum)
          : 0.0f;
      steps[threadIdx.x] = dt;
      float running = dt * rate;
#pragma unroll
      for (unsigned int offset = 1u; offset < 32u; offset <<= 1u) {
        const float neighbor = __shfl_up_sync(0xffffffffu, running, offset);
        if (lane >= offset) running += neighbor;
      }
      sums[threadIdx.x] = running;
    }
    __syncthreads();
    if (threadIdx.x >= 32u && threadIdx.x < chunk) {
      sums[threadIdx.x] += sums[31];
    }
    // products = C · Bᵀ.
    for (unsigned int output_tile = warp; output_tile < chunk_tiles * chunk_tiles;
         output_tile += warps) {
      const unsigned int row_tile = output_tile / chunk_tiles;
      const unsigned int column_tile = output_tile % chunk_tiles;
      LibmirMamba2Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      for (unsigned int step = 0; step < state_tiles; ++step) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, first + row_tile * tile * wide + step * tile, wide);
        wmma::load_matrix_sync(b, second + column_tile * tile * wide + step * tile, wide);
        wmma::mma_sync(accumulator, a, b, accumulator);
      }
      wmma::store_matrix_sync(products + row_tile * tile * product_stride + column_tile * tile,
                              accumulator, product_stride, wmma::mem_row_major);
    }
    __syncthreads();
    // Decayed causal scores in registers, then written over the products as
    // BF16; C scaled by exp(Λ_t); S rounded to BF16 over B.
    constexpr unsigned int per_thread = (LIBMIR_MAMBA2_CHUNK * LIBMIR_MAMBA2_CHUNK) / threads;
    float pending[per_thread];
#pragma unroll
    for (unsigned int item = 0; item < per_thread; ++item) {
      const unsigned int index = threadIdx.x + item * threads;
      const unsigned int target = index / chunk;
      const unsigned int source = index % chunk;
      pending[item] = source <= target && target < length
          ? products[target * product_stride + source] *
                __expf(sums[target] - sums[source]) * steps[source]
          : 0.0f;
    }
    for (unsigned int index = threadIdx.x; index < chunk * state_size; index += threads) {
      const unsigned int token = index / state_size;
      const unsigned int column = index % state_size;
      first[token * wide + column] = __float2bfloat16_rn(
          __bfloat162float(first[token * wide + column]) * __expf(sums[token]));
    }
    __syncthreads();
#pragma unroll
    for (unsigned int item = 0; item < per_thread; ++item) {
      const unsigned int index = threadIdx.x + item * threads;
      scores[(index / chunk) * score_stride + index % chunk] = __float2bfloat16_rn(pending[item]);
    }
    for (unsigned int index = threadIdx.x; index < head_dim * state_size; index += threads) {
      const unsigned int row = index / state_size;
      const unsigned int column = index % state_size;
      second[row * wide + column] = __float2bfloat16_rn(carried[row * state_stride + column]);
    }
    __syncthreads();
    // y = G · X + C_scaled · Sᵀ, kept in registers across the barrier.
    LibmirMamba2Accumulator result;
    const bool owns_output = warp < chunk_tiles * row_tiles;
    if (owns_output) {
      const unsigned int token_tile = warp / row_tiles;
      const unsigned int row_tile = warp % row_tiles;
      wmma::fill_fragment(result, 0.0f);
      for (unsigned int step = 0; step <= token_tile; ++step) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, scores + token_tile * tile * score_stride + step * tile, score_stride);
        wmma::load_matrix_sync(b, x_values + step * tile * x_stride + row_tile * tile, x_stride);
        wmma::mma_sync(result, a, b, result);
      }
      for (unsigned int step = 0; step < state_tiles; ++step) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, first + token_tile * tile * wide + step * tile, wide);
        wmma::load_matrix_sync(b, second + row_tile * tile * wide + step * tile, wide);
        wmma::mma_sync(result, a, b, result);
      }
    }
    __syncthreads();
    if (owns_output) {
      wmma::store_matrix_sync(y_values + (warp / row_tiles) * tile * y_stride +
                                  (warp % row_tiles) * tile,
                              result, y_stride, wmma::mem_row_major);
    }
    // Weighted B for the state update replaces S in R2; decay the state.
    const float last = sums[length - 1u];
#pragma unroll
    for (unsigned int item = 0; item < 2u; ++item) {
      const unsigned int index = threadIdx.x + item * threads;
      const unsigned int token = index / vectors;
      if (token < chunk) {
        const float weight = __expf(last - sums[token]) * steps[token];
        const __nv_bfloat16* source = reinterpret_cast<const __nv_bfloat16*>(&raw_b[item]);
        __nv_bfloat16 weighted[8];
#pragma unroll
        for (unsigned int lane_index = 0; lane_index < 8u; ++lane_index) {
          weighted[lane_index] = __float2bfloat16_rn(__bfloat162float(source[lane_index]) * weight);
        }
        *reinterpret_cast<uint4*>(second + token * wide + (index % vectors) * 8u) =
            *reinterpret_cast<const uint4*>(weighted);
      }
    }
    const float decay = __expf(last);
    for (unsigned int index = threadIdx.x; index < head_dim * state_size; index += threads) {
      carried[(index / state_size) * state_stride + index % state_size] *= decay;
    }
    if (begin + chunk < tokens) {
      libmir_mamba2_prefetch(fetched, convolved, projection, begin + chunk, tokens,
                             convolved_width, inner, head, head_dim, state_size, groups,
                             group, projection_stride, dt_offset);
    }
    __syncthreads();
    for (unsigned int index = threadIdx.x; index < length * head_dim; index += threads) {
      const unsigned int token = index / head_dim;
      const unsigned int row = index % head_dim;
      const float value = y_values[token * y_stride + row] +
                          skip_scale * __bfloat162float(x_values[token * x_stride + row]);
      output[static_cast<size_t>(begin + token) * inner + head * head_dim + row] =
          __float2bfloat16_rn(value);
    }
    // S += Xᵀ · weighted B.
    for (unsigned int output_tile = warp; output_tile < row_tiles * state_tiles;
         output_tile += warps) {
      const unsigned int row_tile = output_tile / state_tiles;
      const unsigned int column_tile = output_tile % state_tiles;
      LibmirMamba2Accumulator accumulator;
      float* target = carried + row_tile * tile * state_stride + column_tile * tile;
      wmma::load_matrix_sync(accumulator, target, state_stride, wmma::mem_row_major);
      for (unsigned int step = 0; step < chunk_tiles; ++step) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, x_values + step * tile * x_stride + row_tile * tile, x_stride);
        wmma::load_matrix_sync(b, second + step * tile * wide + column_tile * tile, wide);
        wmma::mma_sync(accumulator, a, b, accumulator);
      }
      wmma::store_matrix_sync(target, accumulator, state_stride, wmma::mem_row_major);
    }
    __syncthreads();
  }
  for (unsigned int index = threadIdx.x; index < head_dim * state_size; index += threads) {
    head_state[index] = carried[(index / state_size) * state_stride + index % state_size];
  }
}
