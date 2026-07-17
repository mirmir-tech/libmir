#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>

namespace {
#ifndef LIBMIR_OUTPUT_REFINE_TOP_K
#define LIBMIR_OUTPUT_REFINE_TOP_K 64
#endif
constexpr unsigned int kThreads = 256;
constexpr unsigned int kItems = 8;
constexpr unsigned int kTopK = LIBMIR_OUTPUT_REFINE_TOP_K;
constexpr unsigned int kChunk = kThreads * kItems;
constexpr unsigned int kWarps = kThreads / 32;

__device__ bool finite(float value) {
  const float infinity = __int_as_float(0x7f800000);
  return value == value && value != infinity && value != -infinity;
}

__device__ unsigned long long key(float score, unsigned int token) {
  const unsigned int bits = __float_as_uint(score);
  const unsigned int ordered =
      (bits & 0x80000000u) == 0u ? bits ^ 0x80000000u : ~bits;
  return (static_cast<unsigned long long>(ordered) << 32) | ~token;
}

__device__ unsigned int token(unsigned long long value) {
  return ~static_cast<unsigned int>(value);
}
}  // namespace

extern "C" __global__ void libmir_cuda_output_refine_candidates(
    const __nv_bfloat16* logits, unsigned long long* output,
    unsigned int vocab) {
  using Sort = cub::BlockRadixSort<unsigned long long, kThreads, kItems>;
  __shared__ typename Sort::TempStorage storage;
  unsigned long long keys[kItems];
  const unsigned int base = blockIdx.x * kChunk;
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int index = base + threadIdx.x + item * kThreads;
    const float score = index < vocab ? __bfloat162float(logits[index]) : 0.0f;
    keys[item] = index < vocab && finite(score) ? key(score, index) : 0ull;
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int rank = threadIdx.x * kItems + item;
    if (rank < kTopK) output[blockIdx.x * kTopK + rank] = keys[item];
  }
}

extern "C" __global__ void libmir_cuda_output_refine_merge(
    const unsigned long long* input, unsigned long long* output,
    unsigned int count) {
  using Sort = cub::BlockRadixSort<unsigned long long, kThreads, kItems>;
  __shared__ typename Sort::TempStorage storage;
  unsigned long long keys[kItems];
  const unsigned int base = blockIdx.x * kChunk;
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int index = base + threadIdx.x + item * kThreads;
    keys[item] = index < count ? input[index] : 0ull;
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int rank = threadIdx.x * kItems + item;
    if (rank < kTopK) output[blockIdx.x * kTopK + rank] = keys[item];
  }
}

extern "C" __global__ void libmir_cuda_output_refine_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const unsigned long long* candidates, __nv_bfloat16* output,
    unsigned int columns) {
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int candidate = blockIdx.x * kWarps + warp;
  if (candidate >= kTopK) return;
  const unsigned int row = token(candidates[candidate]);
  float sum = 0.0f;
  for (unsigned int pair = lane; pair < columns / 2u; pair += 32u) {
    const unsigned int column = pair * 2u;
    sum = fmaf(__bfloat162float(input[column]),
               __bfloat162float(weight[row * columns + column]), sum);
    sum = fmaf(__bfloat162float(input[column + 1u]),
               __bfloat162float(weight[row * columns + column + 1u]), sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0u) output[row] = __float2bfloat16_rn(sum);
}
