#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#define LIBMIR_NEGATIVE_FLOAT_MAX -3.402823466e+38F
#ifndef LIBMIR_QUERY_GROUP
#define LIBMIR_QUERY_GROUP 1
#endif
#ifndef LIBMIR_WARP_VALUE_ITEMS
#define LIBMIR_WARP_VALUE_ITEMS 4
#endif

__device__ float load_split_page(const unsigned char* pages,
                                 unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

__device__ __nv_bfloat16 load_split_shared(
    const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  return __float2bfloat16_rn(load_split_page(pages, index));
#else
  return reinterpret_cast<const __nv_bfloat16*>(pages)[index];
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_split_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_table,
    float* partial_values, float* partial_maxima, float* partial_denominators,
    unsigned int token_count, unsigned int block_count,
    unsigned int block_size, unsigned int query_heads,
    unsigned int kv_heads, unsigned int head_dim,
    unsigned int value_head_dim, unsigned int window, float scale,
    unsigned int partition_tokens, unsigned int active_partitions,
    unsigned int max_partitions, unsigned int minimum_tokens) {
#ifdef LIBMIR_SPLIT_GQA_WMMA
  using namespace nvcuda;
  const unsigned int visible_tokens =
      window > 0 ? min(token_count, window) : token_count;
  if (visible_tokens < minimum_tokens) return;
  const unsigned int partition = blockIdx.x / kv_heads;
  const unsigned int kv_head = blockIdx.x % kv_heads;
  if (kv_head >= kv_heads || token_count == 0) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = lane / 32u;
  const unsigned int first_query_head = kv_head * LIBMIR_QUERY_GROUP;
  const unsigned int first =
      window > 0 && token_count > window ? token_count - window : 0;
  const unsigned int begin = first + partition * partition_tokens;
  const unsigned int end = min(begin + partition_tokens, token_count);
  constexpr unsigned int token_tile = 16u;
  constexpr unsigned int tensor_rows = 16u;
  constexpr unsigned int tensor_width = 64u;
  __shared__ unsigned int page_tokens[token_tile];
  __shared__ __nv_bfloat16 query_tile[tensor_rows][tensor_width];
  __shared__ __align__(16)
      __nv_bfloat16 key_tile[token_tile][tensor_width];
  __shared__ __align__(16)
      __nv_bfloat16 value_tile[token_tile][tensor_width];
  __shared__ __nv_bfloat16 weights[tensor_rows][token_tile];
  __shared__ float scores[tensor_rows][token_tile];
  __shared__ float weighted_values[tensor_rows][tensor_width];
  __shared__ float maxima[LIBMIR_QUERY_GROUP];
  __shared__ float denominators[LIBMIR_QUERY_GROUP];
  __shared__ float alphas[LIBMIR_QUERY_GROUP];
  float accumulators[2] = {};
  for (unsigned int index = lane; index < tensor_rows * tensor_width;
       index += blockDim.x) {
    const unsigned int head = index / tensor_width;
    const unsigned int dimension = index % tensor_width;
    query_tile[head][dimension] = head < LIBMIR_QUERY_GROUP
        ? query[(first_query_head + head) * tensor_width + dimension]
        : __float2bfloat16_rn(0.0f);
  }
  for (unsigned int index = lane; index < tensor_rows * token_tile;
       index += blockDim.x) {
    weights[0][index] = __float2bfloat16_rn(0.0f);
  }
  if (lane < LIBMIR_QUERY_GROUP) {
    maxima[lane] = LIBMIR_NEGATIVE_FLOAT_MAX;
    denominators[lane] = 0.0f;
  }
  __syncthreads();

  for (unsigned int token_base = begin; token_base < end;
       token_base += token_tile) {
    if (lane < token_tile) {
      const unsigned int token = token_base + lane;
      const unsigned int logical_block = token / block_size;
      page_tokens[lane] = token < end && logical_block < block_count
          ? block_table[logical_block] * block_size + token % block_size
          : 0xffffffffu;
    }
    __syncthreads();
    constexpr unsigned int vector_width =
        sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr unsigned int vectors_per_token =
        tensor_width / vector_width;
    for (unsigned int vector = lane;
         vector < token_tile * vectors_per_token;
         vector += blockDim.x) {
      const unsigned int token = vector / vectors_per_token;
      const unsigned int item = vector % vectors_per_token;
      const unsigned int page_token = page_tokens[token];
      const unsigned int page_index =
          (page_token * kv_heads + kv_head) * tensor_width +
          item * vector_width;
      uint4 key = {};
      uint4 value = {};
      if (page_token != 0xffffffffu) {
        key = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const __nv_bfloat16*>(key_pages) + page_index);
        value = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const __nv_bfloat16*>(value_pages) + page_index);
      }
      reinterpret_cast<uint4*>(key_tile[token])[item] = key;
      reinterpret_cast<uint4*>(value_tile[token])[item] = value;
    }
    __syncthreads();
    if (warp == 0u) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          query_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          key_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          score_fragment;
      wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
      for (unsigned int offset = 0; offset < tensor_width; offset += 16u) {
        wmma::load_matrix_sync(
            query_fragment, &query_tile[0][offset], tensor_width);
        wmma::load_matrix_sync(
            key_fragment, &key_tile[0][offset], tensor_width);
        wmma::mma_sync(
            score_fragment, query_fragment, key_fragment, score_fragment);
      }
      wmma::store_matrix_sync(
          &scores[0][0], score_fragment, token_tile, wmma::mem_row_major);
    }
    __syncthreads();
    if (lane < LIBMIR_QUERY_GROUP) {
      const unsigned int head = lane;
      float tile_maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
      for (unsigned int token = 0; token < token_tile; ++token) {
        if (page_tokens[token] != 0xffffffffu) {
          tile_maximum =
              fmaxf(tile_maximum, scores[head][token] * scale);
        }
      }
      const float next_maximum = fmaxf(maxima[head], tile_maximum);
      const float alpha = __expf(maxima[head] - next_maximum);
      float denominator = denominators[head] * alpha;
      for (unsigned int token = 0; token < token_tile; ++token) {
        const float weight = page_tokens[token] == 0xffffffffu
            ? 0.0f
            : __expf(scores[head][token] * scale - next_maximum);
        weights[head][token] = __float2bfloat16_rn(weight);
        denominator += weight;
      }
      maxima[head] = next_maximum;
      denominators[head] = denominator;
      alphas[head] = alpha;
    }
    __syncthreads();
    if (warp < 4u) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          value_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          value_accumulator;
      wmma::fill_fragment(value_accumulator, 0.0f);
      wmma::load_matrix_sync(
          weight_fragment, &weights[0][0], token_tile);
      wmma::load_matrix_sync(
          value_fragment, &value_tile[0][warp * 16u], tensor_width);
      wmma::mma_sync(
          value_accumulator, weight_fragment, value_fragment,
          value_accumulator);
      wmma::store_matrix_sync(
          &weighted_values[0][warp * 16u], value_accumulator,
          tensor_width, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (unsigned int item = 0; item < 2u; ++item) {
      const unsigned int index = lane + item * blockDim.x;
      if (index < LIBMIR_QUERY_GROUP * tensor_width) {
        const unsigned int head = index / tensor_width;
        const unsigned int dimension = index % tensor_width;
        accumulators[item] = fmaf(
            accumulators[item], alphas[head],
            weighted_values[head][dimension]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned int item = 0; item < 2u; ++item) {
    const unsigned int index = lane + item * blockDim.x;
    if (index < LIBMIR_QUERY_GROUP * tensor_width) {
      const unsigned int head = index / tensor_width;
      const unsigned int dimension = index % tensor_width;
      const unsigned int partial =
          (first_query_head + head) * max_partitions + partition;
      partial_values[partial * tensor_width + dimension] =
          accumulators[item];
    }
  }
  if (lane < LIBMIR_QUERY_GROUP) {
    const unsigned int partial =
        (first_query_head + lane) * max_partitions + partition;
    partial_maxima[partial] = maxima[lane];
    partial_denominators[partial] = denominators[lane];
  }
#elif defined(LIBMIR_SPLIT_GQA)
  const unsigned int visible_tokens =
      window > 0 ? min(token_count, window) : token_count;
  if (visible_tokens < minimum_tokens) return;
  const unsigned int partition = blockIdx.x / kv_heads;
  const unsigned int kv_head = blockIdx.x % kv_heads;
  if (kv_head >= kv_heads || token_count == 0) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = lane / 32u;
  const unsigned int warp_lane = lane & 31u;
  const unsigned int first =
      window > 0 && token_count > window ? token_count - window : 0;
  const unsigned int begin = first + partition * partition_tokens;
  const unsigned int end = min(begin + partition_tokens, token_count);
  constexpr unsigned int token_tile = 16u;
  constexpr unsigned int maximum_head_dim = 128u;
  __shared__ unsigned int page_tokens[token_tile];
  __shared__ __nv_bfloat16 key_tile[token_tile][maximum_head_dim];
  __shared__ __nv_bfloat16 value_tile[token_tile][maximum_head_dim];
  float accumulators[LIBMIR_WARP_VALUE_ITEMS] = {};
  float maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
  float denominator = 0.0f;

  for (unsigned int token_base = begin; token_base < end;
       token_base += token_tile) {
    if (lane < token_tile) {
      const unsigned int token = token_base + lane;
      const unsigned int logical_block = token / block_size;
      page_tokens[lane] = token < end && logical_block < block_count
          ? block_table[logical_block] * block_size + token % block_size
          : 0xffffffffu;
    }
    __syncthreads();
    for (unsigned int index = lane; index < token_tile * head_dim;
         index += blockDim.x) {
      const unsigned int token = index / head_dim;
      const unsigned int dimension = index % head_dim;
      const unsigned int page_token = page_tokens[token];
      key_tile[token][dimension] = page_token == 0xffffffffu
          ? __float2bfloat16_rn(0.0f)
          : load_split_shared(
                key_pages,
                (page_token * kv_heads + kv_head) * head_dim + dimension);
    }
    for (unsigned int index = lane; index < token_tile * value_head_dim;
         index += blockDim.x) {
      const unsigned int token = index / value_head_dim;
      const unsigned int dimension = index % value_head_dim;
      const unsigned int page_token = page_tokens[token];
      value_tile[token][dimension] = page_token == 0xffffffffu
          ? __float2bfloat16_rn(0.0f)
          : load_split_shared(
                value_pages,
                (page_token * kv_heads + kv_head) * value_head_dim +
                    dimension);
    }
    __syncthreads();
    const unsigned int query_head =
        kv_head * LIBMIR_QUERY_GROUP + warp;
    for (unsigned int token = 0; token < token_tile; ++token) {
      if (page_tokens[token] == 0xffffffffu) continue;
      float dot = 0.0f;
      for (unsigned int dimension = warp_lane; dimension < head_dim;
           dimension += 32u) {
        dot = fmaf(
            __bfloat162float(query[query_head * head_dim + dimension]),
            __bfloat162float(key_tile[token][dimension]), dot);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset);
      }
      float alpha = 0.0f;
      float beta = 0.0f;
      if (warp_lane == 0u) {
        const float score = dot * scale;
        if (score > maximum) {
          alpha = __expf(maximum - score);
          beta = 1.0f;
          maximum = score;
        } else {
          alpha = 1.0f;
          beta = __expf(score - maximum);
        }
        denominator = denominator * alpha + beta;
      }
      alpha = __shfl_sync(0xffffffffu, alpha, 0);
      beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
      for (unsigned int item = 0; item < LIBMIR_WARP_VALUE_ITEMS; ++item) {
        const unsigned int dimension = warp_lane + item * 32u;
        if (dimension < value_head_dim) {
          accumulators[item] = fmaf(
              accumulators[item], alpha,
              __bfloat162float(value_tile[token][dimension]) * beta);
        }
      }
    }
    __syncthreads();
  }
  const unsigned int query_head =
      kv_head * LIBMIR_QUERY_GROUP + warp;
  const unsigned int partial =
      query_head * max_partitions + partition;
#pragma unroll
  for (unsigned int item = 0; item < LIBMIR_WARP_VALUE_ITEMS; ++item) {
    const unsigned int dimension = warp_lane + item * 32u;
    if (dimension < value_head_dim) {
      partial_values[partial * value_head_dim + dimension] =
          accumulators[item];
    }
  }
  if (warp_lane == 0u) {
    partial_maxima[partial] = maximum;
    partial_denominators[partial] = denominator;
  }
#else
  const unsigned int visible_tokens = window > 0
      ? min(token_count, window) : token_count;
  if (visible_tokens < minimum_tokens) return;
  const unsigned int partition = blockIdx.x / query_heads;
  const unsigned int query_head = blockIdx.x % query_heads;
  if (query_head >= query_heads || token_count == 0) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int first = window > 0 && token_count > window
      ? token_count - window : 0;
  const unsigned int begin = first + partition * partition_tokens;
  const unsigned int end = min(begin + partition_tokens, token_count);
  const unsigned int partial = query_head * max_partitions + partition;
  float accumulators[2] = {0.0f, 0.0f};
  __shared__ float warp_sums[8];
  __shared__ float alpha;
  __shared__ float beta;
  __shared__ float denominator;
  __shared__ float maximum;
  if (lane == 0) {
    denominator = 0.0f;
    maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
  }
  __syncthreads();

  for (unsigned int token = begin; token < end; ++token) {
    const unsigned int logical_block = token / block_size;
    if (logical_block >= block_count) return;
    const unsigned int physical_block = block_table[logical_block];
    const unsigned int page_token =
        physical_block * block_size + token % block_size;
    float dot = 0.0f;
    for (unsigned int dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
      const unsigned int q_index = query_head * head_dim + dimension;
      const unsigned int k_index =
          (page_token * kv_heads + kv_head) * head_dim + dimension;
      dot = fmaf(__bfloat162float(query[q_index]),
                 load_split_page(key_pages, k_index), dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = dot;
    __syncthreads();
    if (lane < 32u) {
      float sum = lane < blockDim.x / 32u ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      if (lane == 0u) {
        const float score = sum * scale;
        const float next_maximum = fmaxf(maximum, score);
        alpha = expf(maximum - next_maximum);
        beta = expf(score - next_maximum);
        denominator = denominator * alpha + beta;
        maximum = next_maximum;
      }
    }
    __syncthreads();
    unsigned int local = 0;
    for (unsigned int dimension = lane; dimension < value_head_dim;
         dimension += blockDim.x, ++local) {
      const unsigned int v_index =
          (page_token * kv_heads + kv_head) * value_head_dim + dimension;
      accumulators[local] = fmaf(accumulators[local], alpha,
          load_split_page(value_pages, v_index) * beta);
    }
    __syncthreads();
  }
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    partial_values[partial * value_head_dim + dimension] =
        accumulators[dimension / blockDim.x];
  }
  if (lane == 0) {
    partial_maxima[partial] = maximum;
    partial_denominators[partial] = denominator;
  }
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_merge_bf16(
    const float* partial_values, const float* partial_maxima,
    const float* partial_denominators, __nv_bfloat16* output,
    unsigned int query_heads, unsigned int value_head_dim,
    unsigned int active_partitions, unsigned int max_partitions,
    unsigned int visible_tokens, unsigned int minimum_tokens) {
  if (visible_tokens < minimum_tokens) return;
  const unsigned int query_head = blockIdx.x;
  if (query_head >= query_heads) return;
  const unsigned int lane = threadIdx.x;
  const unsigned int base = query_head * max_partitions;
  extern __shared__ float partition_weights[];
  __shared__ float maximum;
  __shared__ float denominator;
  if (lane == 0) {
    float merged_maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      merged_maximum = fmaxf(merged_maximum, partial_maxima[base + partition]);
    }
    float merged_denominator = 0.0f;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      const float weight =
          expf(partial_maxima[base + partition] - merged_maximum);
      partition_weights[partition] = weight;
      merged_denominator +=
          partial_denominators[base + partition] * weight;
    }
    maximum = merged_maximum;
    denominator = merged_denominator;
  }
  __syncthreads();
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    float numerator = 0.0f;
    for (unsigned int partition = 0; partition < active_partitions; ++partition) {
      numerator +=
          partial_values[(base + partition) * value_head_dim + dimension] *
          partition_weights[partition];
    }
    output[query_head * value_head_dim + dimension] =
        __float2bfloat16_rn(numerator / denominator);
  }
}
