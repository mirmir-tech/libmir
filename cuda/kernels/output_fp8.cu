#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr unsigned int kBlock = 128;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ float block_max(float value, float* maxima) {
  value = warp_max(value);
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  if (lane == 0u) maxima[warp] = value;
  __syncthreads();
  const unsigned int warps = blockDim.x / 32u;
  value = threadIdx.x < warps ? maxima[lane] : 0.0f;
  return warp_max(value);
}
}  // namespace

extern "C" __global__ void libmir_cuda_output_quantize_fp8_weight(
    const __nv_bfloat16* source, unsigned char* weight, float* scales,
    float* row_scales, unsigned int rows, unsigned int columns) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  float maximum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(source[row * columns + column])));
  }
  __shared__ float maxima[kWarps];
  maximum = block_max(maximum, maxima);
  __shared__ float scale;
  if (threadIdx.x == 0u) {
    scale = maximum == 0.0f ? 1.0f : maximum / 448.0f;
    row_scales[row] = scale;
    if (row % kBlock == 0u) {
      const unsigned int row_blocks = rows / kBlock;
      const unsigned int row_block = row / kBlock;
      for (unsigned int column_block = 0; column_block < columns / kBlock;
           ++column_block) {
        scales[column_block * row_blocks + row_block] = 1.0f;
      }
    }
  }
  __syncthreads();
  for (unsigned int column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float value = __bfloat162float(source[row * columns + column]) / scale;
    weight[row * columns + column] = __nv_fp8_e4m3(value).__x;
  }
}

extern "C" __global__ void libmir_cuda_output_quantize_fp8_input(
    const __nv_bfloat16* source, unsigned char* input, float* scales,
    unsigned int columns) {
  const unsigned int column = blockIdx.x * kBlock + threadIdx.x;
  const float value = column < columns ? __bfloat162float(source[column]) : 0.0f;
  __shared__ float maxima[kWarps];
  const float maximum = block_max(fabsf(value), maxima);
  __shared__ float scale;
  if (threadIdx.x == 0u) {
    scale = maximum == 0.0f ? 1.0f : maximum / 448.0f;
    scales[blockIdx.x] = scale;
  }
  __syncthreads();
  if (column < columns) input[column] = __nv_fp8_e4m3(value / scale).__x;
}

extern "C" __global__ void libmir_cuda_output_rescale_fp8_bf16(
    __nv_bfloat16* output, const float* row_scales, unsigned int rows) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    output[row] = __float2bfloat16_rn(
        __bfloat162float(output[row]) * row_scales[row]);
  }
}

extern "C" __global__ void libmir_cuda_output_fp8x4_bf16(
    const __nv_bfloat16* input, const unsigned char* weight,
    const float* row_scales, __nv_bfloat16* output, unsigned int rows,
    unsigned int columns) {
  constexpr unsigned int kRowsPerWarp = 8;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int first_row =
      blockIdx.x * kWarps * kRowsPerWarp + warp * kRowsPerWarp;
  float sums[kRowsPerWarp] = {};
  for (unsigned int group = lane; group < columns / 4u; group += 32u) {
    const unsigned int column = group * 4u;
    const float4 values = make_float4(
        __bfloat162float(input[column]), __bfloat162float(input[column + 1u]),
        __bfloat162float(input[column + 2u]),
        __bfloat162float(input[column + 3u]));
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        __nv_fp8x4_e4m3 packed;
        packed.__x = reinterpret_cast<const unsigned int*>(
            weight + row * columns)[group];
        const float4 scales = static_cast<float4>(packed);
        sums[item] = fmaf(values.x, scales.x, sums[item]);
        sums[item] = fmaf(values.y, scales.y, sums[item]);
        sums[item] = fmaf(values.z, scales.z, sums[item]);
        sums[item] = fmaf(values.w, scales.w, sums[item]);
      }
    }
  }
  #pragma unroll
  for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  if (lane == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        output[row] = __float2bfloat16_rn(sums[item] * row_scales[row]);
      }
    }
  }
}

extern "C" __global__ void libmir_cuda_output_quantize_fp8_int4_weight(
    const __nv_bfloat16* source, const unsigned char* weight,
    const float* row_scales,
    unsigned char* residual, float* residual_scales, unsigned int rows,
    unsigned int columns) {
  const unsigned int row = blockIdx.x;
  if (row >= rows) return;
  const unsigned int column_block = blockIdx.y;
  const unsigned int first_column = column_block * kBlock;
  const unsigned int row_offset = row * columns;
  const unsigned int column = first_column + threadIdx.x;
  __nv_fp8_e4m3 quantized;
  quantized.__x = weight[row_offset + column];
  const float scale = row_scales[row];
  const float value = __bfloat162float(source[row_offset + column]);
  const float difference = value - static_cast<float>(quantized) * scale;
  __shared__ float maxima[kWarps];
  const float residual_maximum = block_max(fabsf(difference), maxima);
  __shared__ float residual_scale;
  if (threadIdx.x == 0u) {
    residual_scale = residual_maximum == 0.0f ? 1.0f : residual_maximum / 7.0f;
    residual_scales[row * (columns / kBlock) + column_block] = residual_scale;
  }
  __syncthreads();
  if (threadIdx.x < kBlock / 2u) {
    const unsigned int pair = first_column / 2u + threadIdx.x;
    unsigned char packed = 0u;
    #pragma unroll
    for (unsigned int item = 0; item < 2u; ++item) {
      const unsigned int item_column = pair * 2u + item;
      __nv_fp8_e4m3 item_quantized;
      item_quantized.__x = weight[row_offset + item_column];
      const float item_value = __bfloat162float(source[row_offset + item_column]);
      const float item_difference =
          item_value - static_cast<float>(item_quantized) * scale;
      const int correction =
          max(-7, min(7, __float2int_rn(item_difference / residual_scale)));
      packed |= static_cast<unsigned char>(correction & 0xf) << (item * 4u);
    }
    residual[row * (columns / 2u) + pair] = packed;
  }
}

extern "C" __global__ void libmir_cuda_output_fp8_int4_bf16(
    const __nv_bfloat16* input, const unsigned char* weight,
    const float* row_scales, const unsigned char* residual,
    const float* residual_scales, __nv_bfloat16* output, unsigned int rows,
    unsigned int columns) {
  constexpr unsigned int kRowsPerWarp = 8;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int first_row =
      blockIdx.x * kWarps * kRowsPerWarp + warp * kRowsPerWarp;
  float sums[kRowsPerWarp] = {};
  for (unsigned int group = lane; group < columns / 4u; group += 32u) {
    const unsigned int column = group * 4u;
    const float values[4] = {
        __bfloat162float(input[column]), __bfloat162float(input[column + 1u]),
        __bfloat162float(input[column + 2u]),
        __bfloat162float(input[column + 3u])};
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) {
        __nv_fp8x4_e4m3 packed_weight;
        packed_weight.__x = reinterpret_cast<const unsigned int*>(
            weight + row * columns)[group];
        const float4 fp8 = static_cast<float4>(packed_weight);
        const float bases[4] = {fp8.x, fp8.y, fp8.z, fp8.w};
        const unsigned short packed_residual =
            reinterpret_cast<const unsigned short*>(
                residual + row * (columns / 2u))[group];
        const float fp8_scale = row_scales[row];
        const float int4_scale =
            residual_scales[row * (columns / kBlock) + group / 32u];
        #pragma unroll
        for (unsigned int value = 0; value < 4u; ++value) {
          int correction = (packed_residual >> (value * 4u)) & 0xfu;
          correction -= (correction & 0x8) << 1;
          sums[item] = fmaf(
              values[value],
              fmaf(static_cast<float>(correction), int4_scale,
                   bases[value] * fp8_scale),
              sums[item]);
        }
      }
    }
  }
  #pragma unroll
  for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  if (lane == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < kRowsPerWarp; ++item) {
      const unsigned int row = first_row + item;
      if (row < rows) output[row] = __float2bfloat16_rn(sums[item]);
    }
  }
}
