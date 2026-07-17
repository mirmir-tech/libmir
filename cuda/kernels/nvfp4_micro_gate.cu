extern "C" __global__ void libmir_cuda_nvfp4_micro_quantize_pair(
    const __nv_bfloat16* input, const unsigned int* selected,
    const float* gate_globals, const float* up_globals,
    unsigned char* gate_packed, unsigned char* up_packed,
    unsigned char* gate_scales, unsigned char* up_scales,
    unsigned int groups, unsigned int selected_count,
    unsigned int columns) {
  const unsigned int blocks = columns / 16u;
  const unsigned int group = blockIdx.x / blocks;
  const unsigned int block = blockIdx.x % blocks;
  const unsigned int lane = threadIdx.x;
  if (group >= groups) return;
  const unsigned int row = group / selected_count;
  const unsigned int expert = selected[group];
  const unsigned int feature = block * 16u + lane;
  float value = lane < 16u ? __bfloat162float(input[row * columns + feature]) : 0.0f;
  float amax = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
  }
  __shared__ float divisors[2];
  if (lane == 0u) {
    const float globals[2] = {gate_globals[expert], up_globals[expert]};
    __nv_fp8_e4m3 scales[2] = {
        __nv_fp8_e4m3(amax == 0.0f ? 1.0f : amax / (6.0f * globals[0])),
        __nv_fp8_e4m3(amax == 0.0f ? 1.0f : amax / (6.0f * globals[1]))};
    divisors[0] = static_cast<float>(scales[0]) * globals[0];
    divisors[1] = static_cast<float>(scales[1]) * globals[1];
    gate_scales[group * blocks + block] = scales[0].__x;
    up_scales[group * blocks + block] = scales[1].__x;
  }
  __syncwarp();
  if (lane < 8u) {
    const unsigned int first = row * columns + block * 16u + lane * 2u;
    const float2 pair = make_float2(__bfloat162float(input[first]),
                                    __bfloat162float(input[first + 1u]));
    __nv_fp4x2_e2m1 gate(make_float2(pair.x / divisors[0], pair.y / divisors[0]));
    __nv_fp4x2_e2m1 up(make_float2(pair.x / divisors[1], pair.y / divisors[1]));
    const unsigned int output = group * columns / 2u + block * 8u + lane;
    gate_packed[output] = gate.__x;
    up_packed[output] = up.__x;
  }
}

extern "C" __global__ void libmir_cuda_nvfp4_micro_fc1(
    const unsigned char* gate_input, const unsigned char* gate_input_scales,
    const unsigned char* up_input, const unsigned char* up_input_scales,
    const unsigned int* selected, const unsigned char* gate_weight,
    const unsigned char* gate_weight_scales, const float* gate_combined,
    const unsigned char* up_weight, const unsigned char* up_weight_scales,
    const float* up_combined, const float* down_input_globals,
    unsigned char* output, unsigned char* output_scales,
    unsigned int groups, unsigned int input_features,
    unsigned int output_features, unsigned int output_scale_stride,
    unsigned int activation) {
  const unsigned int output_blocks = output_features / 16u;
  const unsigned int group = blockIdx.x / output_blocks;
  const unsigned int output_block = blockIdx.x % output_blocks;
  const unsigned int warp = threadIdx.x / 32u;
  const unsigned int lane = threadIdx.x % 32u;
  if (group >= groups) return;
  const unsigned int expert = selected[group];
  __shared__ float values[16];
  __shared__ float divisor;
  for (unsigned int local = warp; local < 16u; local += 8u) {
    const unsigned int row = output_block * 16u + local;
    float gate = libmir_micro_projection(
        gate_input, gate_input_scales, gate_weight, gate_weight_scales,
        gate_combined[expert], group, expert, row, input_features, output_features);
    float up = libmir_micro_projection(
        up_input, up_input_scales, up_weight, up_weight_scales,
        up_combined[expert], group, expert, row, input_features, output_features);
    if (lane == 0u) {
      gate = __bfloat162float(__float2bfloat16_rn(gate));
      up = __bfloat162float(__float2bfloat16_rn(up));
      values[local] = __bfloat162float(__float2bfloat16_rn(
          libmir_micro_activation(gate, activation) * up));
    }
  }
  __syncthreads();
  if (warp == 0u) {
    const float value = lane < 16u ? values[lane] : 0.0f;
    float amax = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
    }
    if (lane == 0u) {
      const float global = down_input_globals[expert];
      __nv_fp8_e4m3 scale(amax == 0.0f ? 1.0f : amax / (6.0f * global));
      divisor = static_cast<float>(scale) * global;
      const unsigned int scale_index = output_scale_stride == 0u
          ? group * output_blocks + output_block
          : group * output_scale_stride +
                libmir_micro_scale_offset(0u, output_block, output_features);
      output_scales[scale_index] = scale.__x;
    }
    __syncwarp();
    if (lane < 8u) {
      __nv_fp4x2_e2m1 packed(make_float2(
          values[lane * 2u] / divisor, values[lane * 2u + 1u] / divisor));
      output[group * output_features / 2u + output_block * 8u + lane] = packed.__x;
    }
  }
}
