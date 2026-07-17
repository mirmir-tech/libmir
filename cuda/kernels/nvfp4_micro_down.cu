extern "C" __global__ void libmir_cuda_nvfp4_micro_gated_quantize(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const unsigned int* selected, const float* input_globals,
    unsigned char* output, unsigned char* output_scales,
    unsigned int groups, unsigned int columns, unsigned int activation) {
  const unsigned int blocks = columns / 16u;
  const unsigned int group = blockIdx.x / blocks;
  const unsigned int block = blockIdx.x % blocks;
  const unsigned int lane = threadIdx.x;
  if (group >= groups) return;
  const unsigned int feature = block * 16u + lane;
  __shared__ float values[16];
  __shared__ float divisor;
  float value = 0.0f;
  if (lane < 16u) {
    const unsigned int index = group * columns + feature;
    value = __bfloat162float(__float2bfloat16_rn(
        libmir_micro_activation(__bfloat162float(gate[index]), activation) *
        __bfloat162float(up[index])));
    values[lane] = value;
  }
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  if (lane == 0u) {
    const float global = input_globals[selected[group]];
    __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
    divisor = static_cast<float>(scale) * global;
    output_scales[group * blocks + block] = scale.__x;
  }
  __syncwarp();
  if (lane < 8u) {
    __nv_fp4x2_e2m1 packed(make_float2(
        values[lane * 2u] / divisor, values[lane * 2u + 1u] / divisor));
    output[group * columns / 2u + block * 8u + lane] = packed.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_micro_fc2(
    const unsigned char* input, const unsigned char* input_scales,
    const unsigned int* selected, const __nv_bfloat16* routing,
    const unsigned char* weight, const unsigned char* weight_scales,
    const float* combined_scales, __nv_bfloat16* output,
    unsigned int input_features, unsigned int output_features,
    unsigned int selected_count, unsigned int tokens) {
  const unsigned int token = blockIdx.y;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int row = blockIdx.x * 8u + warp;
  if (token >= tokens || row >= output_features) return;
  float reduced = 0.0f;
  for (unsigned int rank = 0; rank < selected_count; ++rank) {
    const unsigned int group = token * selected_count + rank;
    const unsigned int expert = selected[group];
    float sum = libmir_micro_projection(
        input, input_scales, weight, weight_scales, combined_scales[expert],
        group, expert, row, input_features, output_features);
    if (lane == 0u) {
      sum = __bfloat162float(__float2bfloat16_rn(sum));
      reduced = fmaf(sum, __bfloat162float(routing[group]), reduced);
    }
  }
  if (lane == 0u) {
    output[token * output_features + row] = __float2bfloat16_rn(reduced);
  }
}
