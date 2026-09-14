#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_attention_transform_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* query_weight, const __nv_bfloat16* key_weight,
    const unsigned int* positions, __nv_bfloat16* rotated_query,
    __nv_bfloat16* rotated_key, __nv_bfloat16* gate,
    unsigned int tokens, unsigned int query_heads, unsigned int key_heads,
    unsigned int head_dim, unsigned int rotary_dim,
    unsigned int section_t, unsigned int section_h, unsigned int section_w,
    unsigned int interleaved, float theta, float epsilon, float weight_shift) {
  const unsigned int token = blockIdx.x / (query_heads + key_heads);
  const unsigned int head = blockIdx.x % (query_heads + key_heads);
  const bool is_query = head < query_heads;
  const unsigned int row = is_query ? token * query_heads + head
      : token * key_heads + head - query_heads;
  const __nv_bfloat16* input = is_query ? query + row * head_dim * 2
      : key + row * head_dim;
  const __nv_bfloat16* weight = is_query ? query_weight : key_weight;
  __nv_bfloat16* output = (is_query ? rotated_query : rotated_key) + row * head_dim;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const float value = __bfloat162float(input[column]);
    sum = fmaf(value, value, sum);
    if (is_query) gate[row * head_dim + column] = input[head_dim + column];
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  __shared__ float warps[8];
  if ((threadIdx.x & 31u) == 0u) warps[threadIdx.x / 32u] = sum;
  __syncthreads();
  if (threadIdx.x < 32u) {
    sum = threadIdx.x < 8u ? warps[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (threadIdx.x == 0u) warps[0] = rsqrtf(sum / head_dim + epsilon);
  }
  __syncthreads();
  for (unsigned int dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    if (dimension >= rotary_dim) {
      output[dimension] = __float2bfloat16_rn(__bfloat162float(input[dimension]) *
          warps[0] * (__bfloat162float(weight[dimension]) + weight_shift));
      continue;
    }
    const unsigned int half = rotary_dim / 2;
    const unsigned int frequency = dimension % half;
    const unsigned int axis = interleaved != 0
        ? (frequency % 3 == 1 && frequency < 3 * section_h ? 1
           : frequency % 3 == 2 && frequency < 3 * section_w ? 2 : 0)
        : (frequency < section_t ? 0 : frequency < section_t + section_h ? 1 : 2);
    const float inverse = powf(theta, -2.0f * frequency / rotary_dim);
    float sine, cosine;
    sincosf(static_cast<float>(positions[axis * tokens + token]) * inverse, &sine, &cosine);
    // Preserve the materialized BF16 RMSNorm result before applying RoPE.
    const float first = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[frequency]) * warps[0] *
        (__bfloat162float(weight[frequency]) + weight_shift)));
    const float second = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[frequency + half]) * warps[0] *
        (__bfloat162float(weight[frequency + half]) + weight_shift)));
    output[dimension] = __float2bfloat16_rn(dimension < half
        ? fmaf(-second, sine, first * cosine) : fmaf(first, sine, second * cosine));
  }
}
