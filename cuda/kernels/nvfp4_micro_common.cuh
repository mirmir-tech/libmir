#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float libmir_micro_dot(
    unsigned int weight0, unsigned int weight1,
    unsigned int input0, unsigned int input1) {
  float result;
  asm volatile(
      "{\n"
      ".reg .b8 w0, w1, w2, w3, w4, w5, w6, w7;\n"
      ".reg .b8 a0, a1, a2, a3, a4, a5, a6, a7;\n"
      ".reg .b32 wh0, wh1, wh2, wh3, wh4, wh5, wh6, wh7;\n"
      ".reg .b32 ah0, ah1, ah2, ah3, ah4, ah5, ah6, ah7;\n"
      ".reg .b16 wlo, whi, alo, ahi;\n"
      ".reg .f32 wflo, wfhi, aflo, afhi, acc0, acc1;\n"
      "mov.b32 {w0, w1, w2, w3}, %1;\n"
      "mov.b32 {w4, w5, w6, w7}, %2;\n"
      "mov.b32 {a0, a1, a2, a3}, %3;\n"
      "mov.b32 {a4, a5, a6, a7}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 wh0, w0;\n"
      "cvt.rn.f16x2.e2m1x2 wh1, w1;\n"
      "cvt.rn.f16x2.e2m1x2 wh2, w2;\n"
      "cvt.rn.f16x2.e2m1x2 wh3, w3;\n"
      "cvt.rn.f16x2.e2m1x2 wh4, w4;\n"
      "cvt.rn.f16x2.e2m1x2 wh5, w5;\n"
      "cvt.rn.f16x2.e2m1x2 wh6, w6;\n"
      "cvt.rn.f16x2.e2m1x2 wh7, w7;\n"
      "cvt.rn.f16x2.e2m1x2 ah0, a0;\n"
      "cvt.rn.f16x2.e2m1x2 ah1, a1;\n"
      "cvt.rn.f16x2.e2m1x2 ah2, a2;\n"
      "cvt.rn.f16x2.e2m1x2 ah3, a3;\n"
      "cvt.rn.f16x2.e2m1x2 ah4, a4;\n"
      "cvt.rn.f16x2.e2m1x2 ah5, a5;\n"
      "cvt.rn.f16x2.e2m1x2 ah6, a6;\n"
      "cvt.rn.f16x2.e2m1x2 ah7, a7;\n"
      "mov.f32 acc0, 0f00000000;\n"
      "mov.f32 acc1, 0f00000000;\n"
      "mov.b32 {wlo, whi}, wh0; mov.b32 {alo, ahi}, ah0;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh1; mov.b32 {alo, ahi}, ah1;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh2; mov.b32 {alo, ahi}, ah2;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh3; mov.b32 {alo, ahi}, ah3;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh4; mov.b32 {alo, ahi}, ah4;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh5; mov.b32 {alo, ahi}, ah5;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh6; mov.b32 {alo, ahi}, ah6;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "mov.b32 {wlo, whi}, wh7; mov.b32 {alo, ahi}, ah7;\n"
      "cvt.f32.f16 wflo, wlo; cvt.f32.f16 wfhi, whi;\n"
      "cvt.f32.f16 aflo, alo; cvt.f32.f16 afhi, ahi;\n"
      "fma.rn.f32 acc0, wflo, aflo, acc0; fma.rn.f32 acc1, wfhi, afhi, acc1;\n"
      "add.f32 %0, acc0, acc1;\n"
      "}\n"
      : "=f"(result)
      : "r"(weight0), "r"(weight1), "r"(input0), "r"(input1));
  return result;
}

__device__ __forceinline__ float libmir_micro_fp8(unsigned char value) {
  __nv_fp8_e4m3 converted;
  converted.__x = value;
  return static_cast<float>(converted);
}

__device__ __forceinline__ float libmir_micro_activation(
    float value, unsigned int activation) {
  if (activation == 1u) return value / (1.0f + expf(-value));
  const float cube = value * value * value;
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * (value + 0.044715f * cube)));
}

__device__ __forceinline__ unsigned int libmir_micro_scale_offset(
    unsigned int row, unsigned int block, unsigned int columns) {
  const unsigned int tile = (row / 128u) * (columns / 64u) + block / 4u;
  const unsigned int local = row % 128u;
  return tile * 512u + (local % 32u) * 16u + (local / 32u) * 4u + block % 4u;
}

__device__ __forceinline__ float libmir_micro_projection(
    const unsigned char* input, const unsigned char* input_scales,
    const unsigned char* weight, const unsigned char* weight_scales,
    float combined_scale, unsigned int group, unsigned int expert,
    unsigned int row, unsigned int input_features,
    unsigned int output_features) {
  const unsigned int lane = threadIdx.x % 32u;
  const unsigned int blocks = input_features / 16u;
  float sum = 0.0f;
  for (unsigned int block = lane; block < blocks; block += 32u) {
    const unsigned int input_byte = group * input_features / 2u + block * 8u;
    const unsigned int weight_byte =
        (expert * output_features + row) * input_features / 2u + block * 8u;
    const unsigned int* input_words =
        reinterpret_cast<const unsigned int*>(input + input_byte);
    const unsigned int* weight_words =
        reinterpret_cast<const unsigned int*>(weight + weight_byte);
    const float scale = libmir_micro_fp8(input_scales[group * blocks + block]) *
        libmir_micro_fp8(weight_scales[
            (expert * output_features + row) * blocks + block]);
    sum = fmaf(libmir_micro_dot(weight_words[0], weight_words[1],
                               input_words[0], input_words[1]), scale, sum);
  }
  for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  return sum * combined_scale;
}
