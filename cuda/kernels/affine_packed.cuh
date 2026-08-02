#ifndef LIBMIR_CUDA_AFFINE_PACKED_CUH
#define LIBMIR_CUDA_AFFINE_PACKED_CUH

template <unsigned int bits>
__device__ __forceinline__ unsigned int libmir_cuda_affine_unpack(
    const unsigned int* row,
    unsigned int feature) {
  static_assert(bits > 0u && bits < 32u, "affine bit width must fit one U32");
  constexpr unsigned int mask = (1u << bits) - 1u;
  const unsigned int bit = feature * bits;
  const unsigned int word = bit >> 5u;
  const unsigned int shift = bit & 31u;
  unsigned long long packed = static_cast<unsigned long long>(row[word]) >> shift;
  if (shift + bits > 32u) {
    packed |= static_cast<unsigned long long>(row[word + 1u]) << (32u - shift);
  }
  return static_cast<unsigned int>(packed) & mask;
}

template <unsigned int bits>
__host__ __device__ constexpr unsigned int libmir_cuda_affine_words(
    unsigned int values) {
  return values * bits / 32u;
}

#endif
