// The first key position a query sees, given the `context` tokens up to and
// including it and the layer's window argument: zero for full attention, a
// sliding window's token count, or a chunk size with the high bit set
// (Llama 4's chunked attention, which sees the keys since its chunk began).
#define LIBMIR_WINDOW_CHUNKED 0x80000000u

__device__ __forceinline__ unsigned int libmir_window_first(
    unsigned int context, unsigned int window) {
  if (window == 0u || context == 0u) return 0u;
  if (window & LIBMIR_WINDOW_CHUNKED) {
    const unsigned int chunk = window & ~LIBMIR_WINDOW_CHUNKED;
    return (context - 1u) / chunk * chunk;
  }
  return context > window ? context - window : 0u;
}
