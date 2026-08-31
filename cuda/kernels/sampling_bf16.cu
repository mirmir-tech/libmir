#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace {
constexpr unsigned int kThreads = 256;
constexpr unsigned int kItems = 8;
constexpr unsigned int kChunk = kThreads * kItems;

__device__ float infinity() { return __int_as_float(0x7f800000); }

__device__ bool finite(float value) {
  const float limit = infinity();
  return value == value && value != limit && value != -limit;
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

extern "C" __global__ void libmir_cuda_sampling_candidates_bf16(
    const __nv_bfloat16* logits, unsigned long long* output,
    float* denominator, unsigned int vocab, unsigned int logits_stride, unsigned int top_k,
    unsigned int row, unsigned int workspace_stride) {
  using Sort = cub::BlockRadixSort<unsigned long long, kThreads, kItems>;
  __shared__ typename Sort::TempStorage storage;
  unsigned long long keys[kItems];
  logits += static_cast<unsigned long long>(row) * logits_stride;
  output += static_cast<unsigned long long>(row) * workspace_stride;
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
    if (rank < top_k) output[blockIdx.x * top_k + rank] = keys[item];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) denominator[row] = 0.0f;
}

extern "C" __global__ void libmir_cuda_sampling_merge(
    const unsigned long long* input, unsigned long long* output,
    float* denominator, unsigned int count, unsigned int top_k,
    unsigned int row, unsigned int workspace_stride) {
  using Sort = cub::BlockRadixSort<unsigned long long, kThreads, kItems>;
  __shared__ typename Sort::TempStorage storage;
  unsigned long long keys[kItems];
  input += static_cast<unsigned long long>(row) * workspace_stride;
  output += static_cast<unsigned long long>(row) * workspace_stride;
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
    if (rank < top_k) output[blockIdx.x * top_k + rank] = keys[item];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) denominator[row] = 0.0f;
}

extern "C" __global__ void libmir_cuda_sampling_mass_bf16(
    const __nv_bfloat16* logits, const unsigned long long* candidates,
    float* denominator, unsigned int vocab, unsigned int logits_stride, unsigned int row,
    unsigned int workspace_stride) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  logits += static_cast<unsigned long long>(row) * logits_stride;
  candidates += static_cast<unsigned long long>(row) * workspace_stride;
  const float maximum = __bfloat162float(logits[token(candidates[0])]);
  float local = 0.0f;
  for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < vocab; index += blockDim.x * gridDim.x) {
    const float score = __bfloat162float(logits[index]);
    if (finite(score)) local += expf(score - maximum);
  }
  const float total = Reduce(storage).Sum(local);
  if (threadIdx.x == 0) atomicAdd(denominator + row, total);
}

extern "C" __global__ void libmir_cuda_sampling_finalize_bf16(
    const __nv_bfloat16* logits, const unsigned long long* candidates,
    const float* denominator, unsigned int* output, unsigned int top_k,
    float top_p, float temperature, float draw, unsigned int vocab,
    unsigned int logits_stride, unsigned int row, unsigned int workspace_stride) {
  if (threadIdx.x != 0) return;
  logits += static_cast<unsigned long long>(row) * logits_stride;
  candidates += static_cast<unsigned long long>(row) * workspace_stride;
  const unsigned int first = token(candidates[0]);
  if (top_k == 1) {
    output[row] = first;
    return;
  }
  const float maximum = __bfloat162float(logits[first]);
  float nucleus_prior = 0.0f;
  float weight_total = 0.0f;
  unsigned int kept = 0;
  for (; kept < top_k && nucleus_prior < top_p; ++kept) {
    const float score = __bfloat162float(logits[token(candidates[kept])]);
    nucleus_prior += expf(score - maximum) / denominator[row];
    weight_total += expf((score - maximum) / temperature);
  }
  const float threshold = weight_total * draw;
  float cumulative = 0.0f;
  unsigned int selected = first;
  for (unsigned int index = 0; index < kept; ++index) {
    const unsigned int candidate = token(candidates[index]);
    cumulative += expf(
        (__bfloat162float(logits[candidate]) - maximum) / temperature);
    if (cumulative >= threshold) {
      selected = candidate;
      break;
    }
  }
  output[row] = selected;
}

extern "C" __global__ void libmir_cuda_sampling_full_mass_bf16(
    const __nv_bfloat16* logits, const unsigned long long* candidates,
    float* block_mass, unsigned int vocab, float temperature,
    unsigned int logits_stride, unsigned int row, unsigned int workspace_stride) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  logits += static_cast<unsigned long long>(row) * logits_stride;
  candidates += static_cast<unsigned long long>(row) * workspace_stride;
  block_mass += static_cast<unsigned long long>(row) * gridDim.x;
  const float maximum = __bfloat162float(logits[token(candidates[0])]);
  const unsigned int base = blockIdx.x * kChunk;
  float local = 0.0f;
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int index = base + threadIdx.x + item * kThreads;
    if (index < vocab) {
      const float score = __bfloat162float(logits[index]);
      if (finite(score)) local += expf((score - maximum) / temperature);
    }
  }
  const float total = Reduce(storage).Sum(local);
  if (threadIdx.x == 0) block_mass[blockIdx.x] = total;
}

extern "C" __global__ void libmir_cuda_sampling_full_finalize_bf16(
    const __nv_bfloat16* logits, const unsigned long long* candidates,
    const float* block_mass, unsigned int* output, float temperature, float draw,
    unsigned int vocab, unsigned int logits_stride, unsigned int row,
    unsigned int workspace_stride, unsigned int block_count) {
  using Scan = cub::BlockScan<float, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ unsigned int selected_block;
  __shared__ float selected_threshold;
  logits += static_cast<unsigned long long>(row) * logits_stride;
  candidates += static_cast<unsigned long long>(row) * workspace_stride;
  block_mass += static_cast<unsigned long long>(row) * block_count;
  const float maximum = __bfloat162float(logits[token(candidates[0])]);
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (unsigned int block = 0; block < block_count; ++block) total += block_mass[block];
    selected_threshold = total * draw;
    selected_block = 0;
    while (selected_block + 1 < block_count &&
           (block_mass[selected_block] <= 0.0f ||
            selected_threshold > block_mass[selected_block])) {
      selected_threshold -= block_mass[selected_block];
      selected_block++;
    }
    output[row] = token(candidates[0]);
  }
  __syncthreads();
  const unsigned int begin = selected_block * kChunk + threadIdx.x * kItems;
  float weights[kItems];
  float local = 0.0f;
  #pragma unroll
  for (unsigned int item = 0; item < kItems; ++item) {
    const unsigned int index = begin + item;
    const float score = index < vocab ? __bfloat162float(logits[index]) : 0.0f;
    weights[item] = index < vocab && finite(score)
                           ? expf((score - maximum) / temperature)
                           : 0.0f;
    local += weights[item];
  }
  float prefix = 0.0f;
  Scan(scan_storage).ExclusiveSum(local, prefix);
  const bool starts_distribution =
      selected_threshold == 0.0f && prefix == 0.0f && local > 0.0f;
  if (selected_threshold <= prefix + local &&
      (selected_threshold > prefix || starts_distribution)) {
    float cumulative = prefix;
    #pragma unroll
    for (unsigned int item = 0; item < kItems; ++item) {
      cumulative += weights[item];
      if (weights[item] > 0.0f && selected_threshold <= cumulative) {
        output[row] = begin + item;
        break;
      }
    }
  }
}
