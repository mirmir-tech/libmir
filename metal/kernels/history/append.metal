#include <metal_stdlib>
using namespace metal;

struct Parameters {
  uint heads, dim, capacity, offset;
  uint kb, kh, kd, vb, vh, vd;
};

// Exactly one token per row/head. Old readers see only [0, offset), so no
// location visible to an earlier lazy attention is overwritten.
#define APPEND(NAME, T) \
kernel void NAME( \
    device T* history_k [[buffer(0)]], \
    device T* history_v [[buffer(1)]], \
    const device T* keys [[buffer(2)]], \
    const device T* values [[buffer(3)]], \
    constant Parameters& p [[buffer(4)]], \
    uint i [[thread_position_in_grid]]) { \
  uint d = i % p.dim; \
  uint h = (i / p.dim) % p.heads; \
  uint b = i / (p.dim * p.heads); \
  uint target = ((b * p.heads + h) * p.capacity + p.offset) * p.dim + d; \
  history_k[target] = keys[b * p.kb + h * p.kh + d * p.kd]; \
  history_v[target] = values[b * p.vb + h * p.vh + d * p.vd]; \
}

APPEND(history_append_f32, float)
APPEND(history_append_f16, half)
APPEND(history_append_bf16, bfloat)
