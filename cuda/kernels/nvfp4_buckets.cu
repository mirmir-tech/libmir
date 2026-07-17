#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int libmir_bucket_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile_columns = columns / 64u;
  const unsigned int tile = (row / 128u) * tile_columns + block / 4u;
  const unsigned int local_row = row % 128u;
  return tile * 512u + (local_row % 32u) * 16u +
         (local_row / 32u) * 4u + block % 4u;
}

extern "C" __global__ void libmir_cuda_nvfp4_prepare_buckets(
    const unsigned int* selected, unsigned int* counts,
    unsigned int* offsets, unsigned int* order, unsigned int* positions,
    unsigned int* indices, unsigned int assignments, unsigned int experts) {
  extern __shared__ unsigned int shared[];
  unsigned int* local_counts = shared;
  unsigned int* cursors = shared + experts;
  for (unsigned int expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    local_counts[expert] = 0u;
    cursors[expert] = 0u;
    indices[expert] = expert;
  }
  __syncthreads();
  for (unsigned int assignment = threadIdx.x; assignment < assignments;
       assignment += blockDim.x) {
    const unsigned int expert = selected[assignment];
    if (expert < experts) atomicAdd(local_counts + expert, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    unsigned int offset = 0u;
    for (unsigned int expert = 0u; expert < experts; ++expert) {
      const unsigned int count = local_counts[expert];
      counts[expert] = count;
      offsets[expert] = offset;
      offset += count;
    }
  }
  __syncthreads();
  for (unsigned int assignment = threadIdx.x; assignment < assignments;
       assignment += blockDim.x) {
    const unsigned int expert = selected[assignment];
    if (expert >= experts) continue;
    const unsigned int compact =
        offsets[expert] + atomicAdd(cursors + expert, 1u);
    order[compact] = assignment;
    positions[assignment] = compact;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_buckets_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const unsigned int* order, const unsigned int* offsets,
    const float* global_scales, unsigned char* packed,
    unsigned char* scales, unsigned int assignments,
    unsigned int selected_count, unsigned int input_rows,
    unsigned int columns, unsigned int scale_stride, unsigned int ranked) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int compact = blockIdx.x / blocks_per_row;
  const unsigned int block = blockIdx.x % blocks_per_row;
  const unsigned int lane = threadIdx.x;
  if (compact >= assignments) return;
  const unsigned int assignment = order[compact];
  const unsigned int expert = selected[assignment];
  const unsigned int row = ranked == 0u ? assignment / selected_count : compact;
  if (row >= input_rows) return;
  const unsigned int local_row = compact - offsets[expert];
  const unsigned int feature = block * 16u + lane;
  float value = lane < 16u ? __bfloat162float(input[row * columns + feature]) : 0.0f;
  float amax = fabsf(value);
  for (int delta = 16; delta > 0; delta >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  __shared__ float divisor;
  if (lane == 0u) {
    const float global = global_scales[expert];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    scales[expert * scale_stride +
           libmir_bucket_scale_offset(local_row, block, columns)] = scale.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const float2 pair = make_float2(
        __bfloat162float(input[first]) / divisor,
        __bfloat162float(input[first + 1u]) / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[compact * columns / 2u + block * 8u + lane] = converted.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_quantize_bucket_pair_bf16(
    const __nv_bfloat16* input, const unsigned int* selected,
    const unsigned int* order, const unsigned int* offsets,
    const float* left_globals, const float* right_globals,
    unsigned char* left_packed, unsigned char* right_packed,
    unsigned char* left_scales, unsigned char* right_scales,
    unsigned int assignments, unsigned int selected_count,
    unsigned int input_rows, unsigned int columns, unsigned int scale_stride) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int compact = blockIdx.x / blocks_per_row;
  const unsigned int block = blockIdx.x % blocks_per_row;
  const unsigned int lane = threadIdx.x;
  if (compact >= assignments) return;
  const unsigned int assignment = order[compact];
  const unsigned int expert = selected[assignment];
  const unsigned int row = assignment / selected_count;
  if (row >= input_rows) return;
  const unsigned int local_row = compact - offsets[expert];
  float2 pair = make_float2(0.0f, 0.0f);
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    pair = make_float2(
        __bfloat162float(input[first]), __bfloat162float(input[first + 1u]));
  }
  float amax = fmaxf(fabsf(pair.x), fabsf(pair.y));
  for (int delta = 16; delta > 0; delta >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta));
  }
  __shared__ float divisors[2];
  if (lane == 0u) {
    const float left_global = left_globals[expert];
    const float right_global = right_globals[expert];
    __nv_fp8_e4m3 left_scale(amax == 0.0f ? 1.0f : amax / (6.0f * left_global));
    __nv_fp8_e4m3 right_scale(amax == 0.0f ? 1.0f : amax / (6.0f * right_global));
    divisors[0] = static_cast<float>(left_scale) * left_global;
    divisors[1] = static_cast<float>(right_scale) * right_global;
    const unsigned int scale = expert * scale_stride +
        libmir_bucket_scale_offset(local_row, block, columns);
    left_scales[scale] = left_scale.__x;
    right_scales[scale] = right_scale.__x;
  }
  __syncthreads();
  if (lane < 8u) {
    __nv_fp4x2_e2m1 left(make_float2(pair.x / divisors[0], pair.y / divisors[0]));
    __nv_fp4x2_e2m1 right(make_float2(pair.x / divisors[1], pair.y / divisors[1]));
    const unsigned int packed = compact * columns / 2u + block * 8u + lane;
    left_packed[packed] = left.__x;
    right_packed[packed] = right.__x;
  }
}
