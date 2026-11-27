// The int8 dot products shared by the GGML GEMV kernels, composed after
// `ggml/blocks.cuh`.
namespace {

// One token's quantized activations for one sub-block.
struct Activation {
  uint4 low;
  uint4 high;
  // The activation scale, then the sums of values 0..15 and 16..31.
  float4 scale;
};

__device__ __forceinline__ Activation load_activation(
    const unsigned int* quantized, const float4* scales, unsigned long long at) {
  const uint4* x = reinterpret_cast<const uint4*>(quantized + at * 8u);
  return Activation{__ldg(x), __ldg(x + 1u), __ldg(scales + at)};
}

// The dot product of one decoded sub-block with one token's activations.
__device__ __forceinline__ float dot_sub(const IntSub& sub, const Activation& x) {
  int first = __dp4a(static_cast<int>(sub.q[0]), static_cast<int>(x.low.x), 0);
  first = __dp4a(static_cast<int>(sub.q[1]), static_cast<int>(x.low.y), first);
  first = __dp4a(static_cast<int>(sub.q[2]), static_cast<int>(x.low.z), first);
  first = __dp4a(static_cast<int>(sub.q[3]), static_cast<int>(x.low.w), first);
  int second = __dp4a(static_cast<int>(sub.q[4]), static_cast<int>(x.high.x), 0);
  second = __dp4a(static_cast<int>(sub.q[5]), static_cast<int>(x.high.y), second);
  second = __dp4a(static_cast<int>(sub.q[6]), static_cast<int>(x.high.z), second);
  second = __dp4a(static_cast<int>(sub.q[7]), static_cast<int>(x.high.w), second);
  const float dot =
      sub.scale[0] * static_cast<float>(first) + sub.scale[1] * static_cast<float>(second);
  return fmaf(x.scale.x, dot, -fmaf(sub.minimum[0], x.scale.y, sub.minimum[1] * x.scale.z));
}

// Output rows per warp: a pass of two or more tokens loads each activation
// once for two rows, which halves activation traffic per weight byte.
template <unsigned int kTokens>
__device__ constexpr unsigned int rows_per_warp() {
  return kTokens >= 2u ? 2u : 1u;
}

// Warp sums of each row and token, written by lane 0.
template <unsigned int kRows, unsigned int kTokens>
__device__ __forceinline__ void store_rows(
    float (&total)[kRows][kTokens], __nv_bfloat16* output, unsigned int lane, unsigned int row,
    unsigned int rows, unsigned int first, unsigned int count) {
#pragma unroll
  for (unsigned int r = 0; r < kRows; ++r) {
#pragma unroll
    for (unsigned int t = 0; t < kTokens; ++t) {
      float value = total[r][t];
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
      }
      if (lane == 0u && t < count && row + r < rows) {
        output[static_cast<unsigned long long>(first + t) * rows + row + r] =
            __float2bfloat16_rn(value);
      }
    }
  }
}

}  // namespace
