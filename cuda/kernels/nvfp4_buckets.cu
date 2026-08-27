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

__device__ __forceinline__ float libmir_bucket_activate(
    float value, unsigned int activation) {
  if (activation == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

extern "C" __global__ void libmir_cuda_nvfp4_prepare_buckets(
    const unsigned int* selected, unsigned int* counts,
    unsigned int* offsets, unsigned int* scale_offsets,
    unsigned int* order, unsigned int* positions, unsigned int* indices,
    unsigned int assignments, unsigned int experts) {
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
    unsigned int scale_row_offset = 0u;
    for (unsigned int expert = 0u; expert < experts; ++expert) {
      const unsigned int count = local_counts[expert];
      counts[expert] = count;
      offsets[expert] = offset;
      scale_offsets[expert] = scale_row_offset;
      offset += count;
      scale_row_offset += ((count + 127u) / 128u) * 128u;
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
    const unsigned int* scale_offsets,
    const float* global_scales, unsigned char* packed,
    unsigned char* scales, unsigned int assignments,
    unsigned int selected_count, unsigned int input_rows,
    unsigned int columns, unsigned int ranked) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int work = blockIdx.x * (blockDim.x / 32u) + warp;
  const unsigned int compact = work / blocks_per_row;
  const unsigned int block = work % blocks_per_row;
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
  float divisor = 0.0f;
  if (lane == 0u) {
    const float global = global_scales[expert];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    const unsigned int scale_base =
        (scale_offsets[expert] / 128u) * (columns / 64u) * 512u;
    scales[scale_base +
           libmir_bucket_scale_offset(local_row, block, columns)] = scale.__x;
  }
  divisor = __shfl_sync(0xffffffffu, divisor, 0);
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
    const unsigned int* scale_offsets,
    const float* left_globals, const float* right_globals,
    unsigned char* left_packed, unsigned char* right_packed,
    unsigned char* left_scales, unsigned char* right_scales,
    unsigned int assignments, unsigned int selected_count,
    unsigned int input_rows, unsigned int columns) {
  const unsigned int blocks_per_row = columns / 16u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int work = blockIdx.x * (blockDim.x / 32u) + warp;
  const unsigned int compact = work / blocks_per_row;
  const unsigned int block = work % blocks_per_row;
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
  float left_divisor = 0.0f;
  float right_divisor = 0.0f;
  if (lane == 0u) {
    const float left_global = left_globals[expert];
    const float right_global = right_globals[expert];
    __nv_fp8_e4m3 left_scale(amax == 0.0f ? 1.0f : amax / (6.0f * left_global));
    __nv_fp8_e4m3 right_scale(amax == 0.0f ? 1.0f : amax / (6.0f * right_global));
    left_divisor = static_cast<float>(left_scale) * left_global;
    right_divisor = static_cast<float>(right_scale) * right_global;
    const unsigned int scale_base =
        (scale_offsets[expert] / 128u) * (columns / 64u) * 512u;
    const unsigned int scale = scale_base +
        libmir_bucket_scale_offset(local_row, block, columns);
    left_scales[scale] = left_scale.__x;
    right_scales[scale] = right_scale.__x;
  }
  left_divisor = __shfl_sync(0xffffffffu, left_divisor, 0);
  right_divisor = __shfl_sync(0xffffffffu, right_divisor, 0);
  if (lane < 8u) {
    __nv_fp4x2_e2m1 left(make_float2(pair.x / left_divisor, pair.y / left_divisor));
    __nv_fp4x2_e2m1 right(make_float2(pair.x / right_divisor, pair.y / right_divisor));
    const unsigned int packed = compact * columns / 2u + block * 8u + lane;
    left_packed[packed] = left.__x;
    right_packed[packed] = right.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_gather_quantized_buckets(
    const unsigned int* selected, const unsigned int* order,
    const unsigned int* offsets, const unsigned int* scale_offsets,
    const unsigned char* source_packed, const unsigned char* source_scales,
    unsigned char* packed, unsigned char* scales,
    unsigned int assignments, unsigned int selected_count,
    unsigned int input_rows, unsigned int columns) {
  const unsigned int compact = blockIdx.x;
  if (compact >= assignments) return;
  const unsigned int assignment = order[compact];
  const unsigned int expert = selected[assignment];
  const unsigned int row = assignment / selected_count;
  if (row >= input_rows) return;
  const unsigned int packed_columns = columns / 2u;
  for (unsigned int byte = threadIdx.x; byte < packed_columns;
       byte += blockDim.x) {
    packed[compact * packed_columns + byte] =
        source_packed[row * packed_columns + byte];
  }
  const unsigned int local_row = compact - offsets[expert];
  const unsigned int scale_base =
      (scale_offsets[expert] / 128u) * (columns / 64u) * 512u;
  const unsigned int blocks = columns / 16u;
  for (unsigned int block = threadIdx.x; block < blocks;
       block += blockDim.x) {
    scales[scale_base +
           libmir_bucket_scale_offset(local_row, block, columns)] =
        source_scales[libmir_bucket_scale_offset(row, block, columns)];
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_gated_quantize_buckets_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const unsigned int* selected, const unsigned int* order,
    const unsigned int* offsets, const unsigned int* scale_offsets,
    const float* global_scales, unsigned char* packed,
    unsigned char* scales, unsigned int assignments,
    unsigned int columns, unsigned int activation) {
  const unsigned int block_pairs_per_row = columns / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int work = blockIdx.x * (blockDim.x / 32u) + warp;
  const unsigned int compact = work / block_pairs_per_row;
  const unsigned int block = (work % block_pairs_per_row) * 2u + lane / 16u;
  if (compact >= assignments) return;
  const unsigned int assignment = order[compact];
  const unsigned int expert = selected[assignment];
  const unsigned int local_row = compact - offsets[expert];
  const unsigned int index = compact * columns + block * 16u + lane % 16u;
  const float activated = __bfloat162float(__float2bfloat16_rn(
      libmir_bucket_activate(__bfloat162float(gate[index]), activation)));
  const float value = __bfloat162float(__float2bfloat16_rn(
      activated * __bfloat162float(up[index])));
  float amax = fabsf(value);
  for (int delta = 8; delta > 0; delta >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, delta, 16));
  }
  float divisor = 0.0f;
  const unsigned int half_lane = lane % 16u;
  if (half_lane == 0u) {
    const float global = global_scales[expert];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    const unsigned int scale_base =
        (scale_offsets[expert] / 128u) * (columns / 64u) * 512u;
    scales[scale_base +
           libmir_bucket_scale_offset(local_row, block, columns)] = scale.__x;
  }
  divisor = __shfl_sync(0xffffffffu, divisor, 0, 16);
  const unsigned int pair_lane = (half_lane & 7u) * 2u;
  const float first_value = __shfl_sync(0xffffffffu, value, pair_lane, 16);
  const float second_value = __shfl_sync(0xffffffffu, value, pair_lane + 1u, 16);
  if (half_lane < 8u) {
    const float2 pair = make_float2(first_value / divisor, second_value / divisor);
    __nv_fp4x2_e2m1 converted(pair);
    packed[compact * columns / 2u + block * 8u + half_lane] = converted.__x;
  }
}
