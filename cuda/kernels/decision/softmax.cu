// Masked softmax over score rows of a strided batched product. Batch member
// `i` covers queries `g · queries + [0, queries)` of the flattened padded
// batch, `g = i % groups`, and keys starting `shift` tokens before its first
// query. A key counts when it lies in the query's sequence, among its valid
// tokens and within `radius` (`0xffffffff`: no band); rows of padded queries
// become zeros. Each block reads its row once into `keys` floats of dynamic
// shared memory.
#include <cuda_bf16.h>

__device__ inline void put(float* target, float value) { *target = value; }
__device__ inline void put(__nv_bfloat16* target, float value) {
  *target = __float2bfloat16(value);
}

__device__ float block_reduce(float value, float* shared, bool maximum) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = maximum ? fmaxf(value, other) : value + other;
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0u) shared[warp] = value;
  __syncthreads();
  const unsigned int warps = (blockDim.x + 31u) >> 5;
  value = lane < warps ? shared[lane] : (maximum ? -__int_as_float(0x7f800000) : 0.0f);
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = maximum ? fmaxf(value, other) : value + other;
  }
  __syncthreads();
  return value;
}

template <typename Out>
__device__ void masked_softmax(
    const float* scores, Out* probabilities, const unsigned int* lengths, unsigned int queries,
    unsigned int keys, unsigned int groups, unsigned int shift, unsigned int length,
    unsigned int radius) {
  __shared__ float partial[32];
  const size_t row = blockIdx.x;
  const long long group = static_cast<long long>((row / queries) % groups);
  const long long token = group * queries + static_cast<long long>(row % queries);
  const long long start = token / length * length;
  const long long valid = lengths[token / length];
  const long long base = group * queries - shift;
  long long first = start - base;
  long long last = start + valid - base;
  if (radius != 0xffffffffu) {
    first = max(first, token - radius - base);
    last = min(last, token + radius + 1 - base);
  }
  first = max(first, 0ll);
  last = token - start >= valid ? first : min(last, static_cast<long long>(keys));
  const float* input = scores + row * keys;
  Out* output = probabilities + row * keys;
  extern __shared__ float cached[];
  float maximum = -__int_as_float(0x7f800000);
  for (long long key = first + threadIdx.x; key < last; key += blockDim.x) {
    cached[key - first] = input[key];
    maximum = fmaxf(maximum, cached[key - first]);
  }
  maximum = block_reduce(maximum, partial, true);
  float total = 0.0f;
  for (long long key = first + threadIdx.x; key < last; key += blockDim.x) {
    cached[key - first] = expf(cached[key - first] - maximum);
    total += cached[key - first];
  }
  const float inverse = last > first ? 1.0f / block_reduce(total, partial, false) : 0.0f;
  for (long long key = threadIdx.x; key < keys; key += blockDim.x) {
    const bool inside = key >= first && key < last;
    put(output + key, inside ? cached[key - first] * inverse : 0.0f);
  }
}

#define LIBMIR_DECISION_SOFTMAX(SUFFIX, OUT)                                                  \
  extern "C" __global__ void libmir_decision_softmax_##SUFFIX(                                \
      const float* scores, OUT* probabilities, const unsigned int* lengths,                   \
      unsigned int queries, unsigned int keys, unsigned int groups, unsigned int shift,       \
      unsigned int length, unsigned int radius) {                                             \
    masked_softmax(scores, probabilities, lengths, queries, keys, groups, shift, length,      \
                   radius);                                                                   \
  }

LIBMIR_DECISION_SOFTMAX(f32, float)
LIBMIR_DECISION_SOFTMAX(bf16, __nv_bfloat16)
