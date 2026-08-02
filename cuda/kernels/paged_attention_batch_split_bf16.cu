#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#define LIBMIR_NEGATIVE_FLOAT_MAX -3.402823466e+38F
#ifndef LIBMIR_QUERY_GROUP
#define LIBMIR_QUERY_GROUP 1
#endif
#ifndef LIBMIR_VALUE_ITEMS
#define LIBMIR_VALUE_ITEMS 2
#endif

__device__ float load_batch_split_page(
    const unsigned char* pages, unsigned int index) {
#ifdef LIBMIR_KV_FP8
  __nv_fp8_e4m3 value;
  value.__x = pages[index];
  return float(value);
#else
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(pages)[index]);
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_batch_split_bf16(
    const __nv_bfloat16* query, const unsigned char* key_pages,
    const unsigned char* value_pages, const unsigned int* block_tables,
    const unsigned int* token_counts, const unsigned int* block_counts,
    float* partial_values, float* partial_maxima, float* partial_denominators,
    unsigned int batch_size, unsigned int max_blocks, unsigned int block_size,
    unsigned int query_heads, unsigned int kv_heads, unsigned int head_dim,
    unsigned int value_head_dim, unsigned int window, float scale,
    unsigned int partition_tokens, unsigned int launch_partitions,
    unsigned int max_partitions,
    unsigned int minimum_tokens) {
  const unsigned int sequence = blockIdx.y;
  const unsigned int kv_head = blockIdx.x / launch_partitions;
  const unsigned int partition = blockIdx.x % launch_partitions;
  if (sequence >= batch_size || kv_head >= kv_heads) return;
  const unsigned int token_count = token_counts[sequence];
  const unsigned int block_count = block_counts[sequence];
  const unsigned int visible = window > 0 ? min(token_count, window) : token_count;
  const unsigned int active = (visible + partition_tokens - 1) / partition_tokens;
  if (visible < minimum_tokens || partition >= active || block_count > max_blocks) return;
  const unsigned int first =
      window > 0 && token_count > window ? token_count - window : 0;
  const unsigned int begin = first + partition * partition_tokens;
  const unsigned int end = min(begin + partition_tokens, token_count);
  const unsigned int lane = threadIdx.x;
  const unsigned int query_group = query_heads / kv_heads;
  const unsigned int first_query_head = kv_head * query_group;
  const unsigned int query_offset = sequence * query_heads * head_dim;
  const unsigned int table_offset = sequence * max_blocks;
#ifdef LIBMIR_BATCH_SPLIT_GQA_WMMA
  using namespace nvcuda;
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
  float accumulators[4] = {};
  const unsigned int warp = lane / 32u;
  for (unsigned int index = lane; index < tensor_rows * tensor_width;
       index += blockDim.x) {
    const unsigned int head = index / tensor_width;
    const unsigned int dimension = index % tensor_width;
    query_tile[head][dimension] = head < LIBMIR_QUERY_GROUP
        ? query[query_offset +
              (first_query_head + head) * tensor_width + dimension]
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
          ? block_tables[table_offset + logical_block] * block_size +
                token % block_size
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
          tile_maximum = fmaxf(tile_maximum, scores[head][token] * scale);
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
    for (unsigned int item = 0; item < 4u; ++item) {
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
  for (unsigned int item = 0; item < 4u; ++item) {
    const unsigned int index = lane + item * blockDim.x;
    if (index < LIBMIR_QUERY_GROUP * tensor_width) {
      const unsigned int head = index / tensor_width;
      const unsigned int dimension = index % tensor_width;
      const unsigned int partial =
          (sequence * query_heads + first_query_head + head) *
              max_partitions +
          partition;
      partial_values[partial * tensor_width + dimension] =
          accumulators[item];
    }
  }
  if (lane < LIBMIR_QUERY_GROUP) {
    const unsigned int partial =
        (sequence * query_heads + first_query_head + lane) *
            max_partitions +
        partition;
    partial_maxima[partial] = maxima[lane];
    partial_denominators[partial] = denominators[lane];
  }
#else
  constexpr unsigned int token_tile = 8u;
  constexpr unsigned int warp_count = 4u;
  constexpr unsigned int tokens_per_warp = token_tile / warp_count;
  float accumulators[LIBMIR_QUERY_GROUP][LIBMIR_VALUE_ITEMS] = {};
  const unsigned int warp = lane / 32u;
  const unsigned int warp_lane = lane & 31u;
  __shared__ float scores[LIBMIR_QUERY_GROUP][token_tile];
  __shared__ float weights[LIBMIR_QUERY_GROUP][token_tile];
  __shared__ float alpha[LIBMIR_QUERY_GROUP];
  __shared__ float denominators[LIBMIR_QUERY_GROUP];
  __shared__ float maxima[LIBMIR_QUERY_GROUP];
  if (lane < query_group) {
    denominators[lane] = 0.0f;
    maxima[lane] = LIBMIR_NEGATIVE_FLOAT_MAX;
  }
  __syncthreads();

  for (unsigned int token_base = begin; token_base < end;
       token_base += token_tile) {
    unsigned int tokens[tokens_per_warp];
    unsigned int page_tokens[tokens_per_warp];
    bool valid[tokens_per_warp];
#pragma unroll
    for (unsigned int item = 0; item < tokens_per_warp; ++item) {
      tokens[item] = token_base + warp + item * warp_count;
      const unsigned int logical_block = tokens[item] / block_size;
      valid[item] = tokens[item] < end && logical_block < block_count;
      if (valid[item]) {
        const unsigned int physical =
            block_tables[table_offset + logical_block];
        page_tokens[item] =
            physical * block_size + tokens[item] % block_size;
      }
    }
    float dots[tokens_per_warp][LIBMIR_QUERY_GROUP] = {};
    for (unsigned int dimension = warp_lane; dimension < head_dim;
         dimension += 32u) {
#pragma unroll
      for (unsigned int item = 0; item < tokens_per_warp; ++item) {
        if (valid[item]) {
          const unsigned int k_index =
              (page_tokens[item] * kv_heads + kv_head) * head_dim + dimension;
          const float key = load_batch_split_page(key_pages, k_index);
#pragma unroll
          for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
            const unsigned int q_index =
                query_offset + (first_query_head + head) * head_dim + dimension;
            dots[item][head] =
                fmaf(__bfloat162float(query[q_index]), key, dots[item][head]);
          }
        }
      }
    }
#pragma unroll
    for (unsigned int item = 0; item < tokens_per_warp; ++item) {
#pragma unroll
      for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
        for (int offset = 16; offset > 0; offset >>= 1) {
          dots[item][head] +=
              __shfl_down_sync(0xffffffffu, dots[item][head], offset);
        }
      }
    }
    if (warp_lane == 0u) {
#pragma unroll
      for (unsigned int item = 0; item < tokens_per_warp; ++item) {
#pragma unroll
        for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
          scores[head][warp + item * warp_count] = valid[item]
              ? dots[item][head] * scale
              : LIBMIR_NEGATIVE_FLOAT_MAX;
        }
      }
    }
    __syncthreads();
    if (lane < query_group) {
      const unsigned int head = lane;
      float tile_maximum = scores[head][0];
#pragma unroll
      for (unsigned int index = 1; index < token_tile; ++index) {
        tile_maximum = fmaxf(tile_maximum, scores[head][index]);
      }
      const float next_maximum = fmaxf(maxima[head], tile_maximum);
      alpha[head] = expf(maxima[head] - next_maximum);
      denominators[head] *= alpha[head];
#pragma unroll
      for (unsigned int index = 0; index < token_tile; ++index) {
        const float weight = expf(scores[head][index] - next_maximum);
        weights[head][index] = weight;
        denominators[head] += weight;
      }
      maxima[head] = next_maximum;
    }
    __syncthreads();
    unsigned int local = 0;
    for (unsigned int dimension = lane; dimension < value_head_dim;
         dimension += blockDim.x, ++local) {
      float weighted_values[LIBMIR_QUERY_GROUP] = {};
#pragma unroll
      for (unsigned int index = 0; index < token_tile; ++index) {
        const unsigned int value_token = token_base + index;
        if (value_token < end) {
          const unsigned int logical_block = value_token / block_size;
          if (logical_block < block_count) {
            const unsigned int physical_block =
                block_tables[table_offset + logical_block];
            const unsigned int page_token =
                physical_block * block_size + value_token % block_size;
            const unsigned int v_index =
                (page_token * kv_heads + kv_head) * value_head_dim + dimension;
            const float value = load_batch_split_page(value_pages, v_index);
#pragma unroll
            for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
              weighted_values[head] =
                  fmaf(value, weights[head][index], weighted_values[head]);
            }
          }
        }
      }
#pragma unroll
      for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
        accumulators[head][local] = fmaf(
            accumulators[head][local], alpha[head], weighted_values[head]);
      }
    }
  }
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    const unsigned int local = dimension / blockDim.x;
#pragma unroll
    for (unsigned int head = 0; head < LIBMIR_QUERY_GROUP; ++head) {
      const unsigned int partial =
          (sequence * query_heads + first_query_head + head) *
              max_partitions +
          partition;
      partial_values[partial * value_head_dim + dimension] =
          accumulators[head][local];
    }
  }
  if (lane < query_group) {
    const unsigned int partial =
        (sequence * query_heads + first_query_head + lane) * max_partitions +
        partition;
    partial_maxima[partial] = maxima[lane];
    partial_denominators[partial] = denominators[lane];
  }
#endif
}

extern "C" __global__ void libmir_cuda_paged_attention_batch_merge_bf16(
    const float* partial_values, const float* partial_maxima,
    const float* partial_denominators, const unsigned int* token_counts,
    __nv_bfloat16* output, unsigned int batch_size, unsigned int query_heads,
    unsigned int value_head_dim, unsigned int window,
    unsigned int partition_tokens, unsigned int max_partitions,
    unsigned int minimum_tokens) {
  const unsigned int sequence = blockIdx.y;
  const unsigned int query_head = blockIdx.x;
  if (sequence >= batch_size || query_head >= query_heads) return;
  const unsigned int token_count = token_counts[sequence];
  const unsigned int visible = window > 0 ? min(token_count, window) : token_count;
  if (visible < minimum_tokens) return;
  const unsigned int active = (visible + partition_tokens - 1) / partition_tokens;
  const unsigned int base =
      (sequence * query_heads + query_head) * max_partitions;
  const unsigned int lane = threadIdx.x;
  __shared__ float maximum;
  __shared__ float denominator;
  if (lane == 0) {
    float merged_maximum = LIBMIR_NEGATIVE_FLOAT_MAX;
    for (unsigned int partition = 0; partition < active; ++partition) {
      merged_maximum = fmaxf(merged_maximum, partial_maxima[base + partition]);
    }
    float merged_denominator = 0.0f;
    for (unsigned int partition = 0; partition < active; ++partition) {
      merged_denominator += partial_denominators[base + partition] *
          expf(partial_maxima[base + partition] - merged_maximum);
    }
    maximum = merged_maximum;
    denominator = merged_denominator;
  }
  __syncthreads();
  for (unsigned int dimension = lane; dimension < value_head_dim;
       dimension += blockDim.x) {
    float numerator = 0.0f;
    for (unsigned int partition = 0; partition < active; ++partition) {
      const float weight = expf(partial_maxima[base + partition] - maximum);
      numerator += partial_values[(base + partition) * value_head_dim + dimension] * weight;
    }
    const unsigned int out =
        (sequence * query_heads + query_head) * value_head_dim + dimension;
    output[out] = __float2bfloat16_rn(numerator / denominator);
  }
}
