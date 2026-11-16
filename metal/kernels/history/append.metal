#include <metal_stdlib>
using namespace metal;

struct Parameters {
  uint heads, dim, capacity;
  uint kb, kh, kd, vb, vh, vd;
};

// Exactly one token per row/head, at that row's own length. Old readers see
// only [0, length), so nothing visible to an earlier lazy attention changes.
#define APPEND(NAME, T) \
kernel void NAME( \
    device T* history_k [[buffer(0)]], \
    device T* history_v [[buffer(1)]], \
    const device T* keys [[buffer(2)]], \
    const device T* values [[buffer(3)]], \
    const device uint* offsets [[buffer(4)]], \
    constant Parameters& p [[buffer(5)]], \
    uint i [[thread_position_in_grid]]) { \
  uint d = i % p.dim; \
  uint h = (i / p.dim) % p.heads; \
  uint b = i / (p.dim * p.heads); \
  uint target = ((b * p.heads + h) * p.capacity + offsets[b]) * p.dim + d; \
  history_k[target] = keys[b * p.kb + h * p.kh + d * p.kd]; \
  history_v[target] = values[b * p.vb + h * p.vh + d * p.vd]; \
}

APPEND(history_append_f32, float)
APPEND(history_append_f16, half)
APPEND(history_append_bf16, bfloat)
