#include <cuda_bf16.h>
#include <mma.h>

// Routing and activation kernels of latent routed-expert blocks.
//
// Router: logits = x · Wᵀ in FP32; each expert's score is sigmoid(logit);
// experts are selected by score + correction bias, and the selected scores
// are renormalized to sum to one and multiplied by the routed scale.

using namespace nvcuda;

#define LIBMIR_ROUTER_TILE 16u
#define LIBMIR_ROUTER_DEPTH 64u
#define LIBMIR_ROUTER_STRIDE (LIBMIR_ROUTER_DEPTH + 8u)
#define LIBMIR_ROUTER_NEGATIVE_INFINITY __int_as_float(0xff800000)

// Grid: (ceil(tokens / 16), experts / 64); block: 128 threads (4 warps).
// Warp w produces logits for 16 tokens and experts [blockIdx.y * 64 + 16w,
// + 16). Requires hidden % 64 == 0 and experts % 64 == 0.
extern "C" __global__ void __launch_bounds__(128) libmir_cuda_router_logits_bf16_f32(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, float* __restrict__ logits,
    unsigned int tokens, unsigned int hidden, unsigned int experts) {
  constexpr unsigned int tile = LIBMIR_ROUTER_TILE;
  constexpr unsigned int depth = LIBMIR_ROUTER_DEPTH;
  constexpr unsigned int stride = LIBMIR_ROUTER_STRIDE;
  __shared__ __align__(32) __nv_bfloat16 activations[tile * stride];
  __shared__ __align__(32) float staged[4][tile * tile];
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int first_token = blockIdx.x * tile;
  const unsigned int first_expert = blockIdx.y * 64u + warp * tile;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
  for (unsigned int begin = 0; begin < hidden; begin += depth) {
    for (unsigned int index = threadIdx.x; index < tile * depth; index += blockDim.x) {
      const unsigned int row = index / depth;
      const unsigned int column = index % depth;
      const unsigned int token = first_token + row;
      activations[row * stride + column] = token < tokens
          ? input[static_cast<size_t>(token) * hidden + begin + column]
          : zero;
    }
    __syncthreads();
#pragma unroll
    for (unsigned int step = 0; step < depth / tile; ++step) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
      wmma::load_matrix_sync(a, activations + step * tile, stride);
      wmma::load_matrix_sync(
          b, weight + static_cast<size_t>(first_expert) * hidden + begin + step * tile, hidden);
      wmma::mma_sync(accumulator, a, b, accumulator);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(staged[warp], accumulator, tile, wmma::mem_row_major);
  __syncwarp();
  const unsigned int lane = threadIdx.x % 32u;
  for (unsigned int index = lane; index < tile * tile; index += 32u) {
    const unsigned int token = first_token + index / tile;
    if (token < tokens) {
      logits[static_cast<size_t>(token) * experts + first_expert + index % tile] =
          staged[warp][index];
    }
  }
}

#define LIBMIR_ROUTER_ROWS 8u

// Router logits for at most 8 tokens. Grid: experts / 8; block: 256 threads.
// Each warp computes one expert's dot products with 16-byte weight loads.
// Requires hidden % 256 == 0.
extern "C" __global__ void __launch_bounds__(256) libmir_cuda_router_logits_rows_bf16_f32(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, float* __restrict__ logits,
    unsigned int tokens, unsigned int hidden, unsigned int experts) {
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int expert = blockIdx.x * 8u + threadIdx.x / 32u;
  if (expert >= experts) return;
  float sums[LIBMIR_ROUTER_ROWS];
#pragma unroll
  for (unsigned int token = 0; token < LIBMIR_ROUTER_ROWS; ++token) sums[token] = 0.0f;
  const __nv_bfloat16* row = weight + static_cast<size_t>(expert) * hidden;
  for (unsigned int column = lane * 8u; column < hidden; column += 256u) {
    const uint4 packed = *reinterpret_cast<const uint4*>(row + column);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
    float2 values[4];
#pragma unroll
    for (unsigned int pair = 0; pair < 4u; ++pair) values[pair] = __bfloat1622float2(pairs[pair]);
#pragma unroll
    for (unsigned int token = 0; token < LIBMIR_ROUTER_ROWS; ++token) {
      if (token < tokens) {
        const uint4 source = *reinterpret_cast<const uint4*>(
            input + static_cast<size_t>(token) * hidden + column);
        const __nv_bfloat162* activations = reinterpret_cast<const __nv_bfloat162*>(&source);
#pragma unroll
        for (unsigned int pair = 0; pair < 4u; ++pair) {
          const float2 activation = __bfloat1622float2(activations[pair]);
          sums[token] = fmaf(activation.x, values[pair].x, sums[token]);
          sums[token] = fmaf(activation.y, values[pair].y, sums[token]);
        }
      }
    }
  }
#pragma unroll
  for (unsigned int token = 0; token < LIBMIR_ROUTER_ROWS; ++token) {
    float sum = sums[token];
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
    if (lane == 0u && token < tokens) logits[static_cast<size_t>(token) * experts + expert] = sum;
  }
}

// Block-wide argmax over (value, index) with the lower index winning ties.
__device__ __forceinline__ void libmir_router_argmax(
    float& value, unsigned int& index, float* warp_values,
    unsigned int* warp_indices) {
  for (int delta = 16; delta > 0; delta >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, delta);
    const unsigned int other_index = __shfl_down_sync(0xffffffffu, index, delta);
    if (other_value > value || (other_value == value && other_index < index)) {
      value = other_value;
      index = other_index;
    }
  }
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  if (lane == 0u) {
    warp_values[warp] = value;
    warp_indices[warp] = index;
  }
  __syncthreads();
  if (warp == 0u) {
    const unsigned int warps = blockDim.x / 32u;
    value = lane < warps ? warp_values[lane] : LIBMIR_ROUTER_NEGATIVE_INFINITY;
    index = lane < warps ? warp_indices[lane] : 0xffffffffu;
    for (int delta = 16; delta > 0; delta >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffffu, value, delta);
      const unsigned int other_index = __shfl_down_sync(0xffffffffu, index, delta);
      if (other_value > value || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
      }
    }
    if (lane == 0u) warp_indices[0] = index;
  }
  __syncthreads();
  index = warp_indices[0];
  __syncthreads();
}

// Grid: tokens; block: experts rounded up to a multiple of 32 (<= 1024).
// Writes `top_k` expert indices in descending selection order and their
// renormalized, scaled routing weights.
extern "C" __global__ void libmir_cuda_router_sigmoid_topk_bf16(
    const float* __restrict__ logits, const float* __restrict__ bias,
    unsigned int* __restrict__ selected, __nv_bfloat16* __restrict__ weights,
    unsigned int experts, unsigned int top_k, float routed_scale) {
  __shared__ float warp_values[32];
  __shared__ unsigned int warp_indices[32];
  __shared__ float chosen_scores[64];
  const unsigned int token = blockIdx.x;
  const unsigned int expert = threadIdx.x;
  const bool live = expert < experts;
  const float logit = live ? logits[static_cast<size_t>(token) * experts + expert] : 0.0f;
  const float score = 1.0f / (1.0f + __expf(-logit));
  float choice = live ? score + bias[expert] : LIBMIR_ROUTER_NEGATIVE_INFINITY;
  for (unsigned int rank = 0; rank < top_k; ++rank) {
    float value = choice;
    unsigned int index = live ? expert : 0xffffffffu;
    libmir_router_argmax(value, index, warp_values, warp_indices);
    if (expert == index) {
      chosen_scores[rank] = score;
      selected[static_cast<size_t>(token) * top_k + rank] = expert;
      choice = LIBMIR_ROUTER_NEGATIVE_INFINITY;
    }
  }
  __syncthreads();
  if (expert < top_k) {
    float total = 0.0f;
    for (unsigned int rank = 0; rank < top_k; ++rank) total += chosen_scores[rank];
    weights[static_cast<size_t>(token) * top_k + expert] =
        __float2bfloat16_rn(chosen_scores[expert] / total * routed_scale);
  }
}

// relu(x)^2, computed in FP32 and rounded once.
extern "C" __global__ void libmir_cuda_relu_squared_bf16(
    const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
    unsigned int elements) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = fmaxf(__bfloat162float(input[index]), 0.0f);
  output[index] = __float2bfloat16_rn(value * value);
}
