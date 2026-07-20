#include <metal_stdlib>
using namespace metal;

struct QuantizedPageWriteParameters {
  uint sequence;
  uint offset;
  uint kv_heads;
  uint page_capacity;
  uint page_size;
  uint head_dim;
  uint key_head_stride;
  uint key_sequence_stride;
  uint key_dimension_stride;
  uint value_head_stride;
  uint value_sequence_stride;
  uint value_dimension_stride;
};

inline uint encode_int8(float value, float inverse_scale) {
  int quantized = int(rint(clamp(value * inverse_scale, -127.0f, 127.0f)));
  return uint(quantized) & 0xffu;
}

template <typename T>
inline void quantized_page_write(
    const device T* keys,
    const device T* values,
    device uint* page_keys,
    device uint* page_values,
    device float* key_scales,
    device float* value_scales,
    const device uint* page_table,
    constant QuantizedPageWriteParameters& parameters,
    uint lane,
    uint head,
    uint token) {
  float key_maximum = 0.0f;
  float value_maximum = 0.0f;
  for (uint dimension = lane; dimension < parameters.head_dim; dimension += 32) {
    uint key_source = head * parameters.key_head_stride +
        token * parameters.key_sequence_stride + dimension * parameters.key_dimension_stride;
    uint value_source = head * parameters.value_head_stride +
        token * parameters.value_sequence_stride + dimension * parameters.value_dimension_stride;
    key_maximum = metal::max(key_maximum, abs(float(keys[key_source])));
    value_maximum = metal::max(value_maximum, abs(float(values[value_source])));
  }
  key_maximum = simd_max(key_maximum);
  value_maximum = simd_max(value_maximum);
  float key_scale = key_maximum == 0.0f ? 1.0f : key_maximum / 127.0f;
  float value_scale = value_maximum == 0.0f ? 1.0f : value_maximum / 127.0f;
  uint absolute = parameters.offset + token;
  uint page = page_table[absolute / parameters.page_size];
  uint in_page = absolute % parameters.page_size;
  uint scale_target = (head * parameters.page_capacity + page) * parameters.page_size + in_page;
  if (lane == 0) {
    key_scales[scale_target] = key_scale;
    value_scales[scale_target] = value_scale;
  }
  uint packed_dim = (parameters.head_dim + 3) / 4;
  uint packed_base = scale_target * packed_dim;
  for (uint word = lane; word < packed_dim; word += 32) {
    uint key_word = 0;
    uint value_word = 0;
    for (uint byte = 0; byte < 4; ++byte) {
      uint dimension = word * 4 + byte;
      if (dimension >= parameters.head_dim) break;
      uint key_source = head * parameters.key_head_stride +
          token * parameters.key_sequence_stride + dimension * parameters.key_dimension_stride;
      uint value_source = head * parameters.value_head_stride +
          token * parameters.value_sequence_stride + dimension * parameters.value_dimension_stride;
      key_word |= encode_int8(float(keys[key_source]), 1.0f / key_scale) << (byte * 8);
      value_word |= encode_int8(float(values[value_source]), 1.0f / value_scale) << (byte * 8);
    }
    page_keys[packed_base + word] = key_word;
    page_values[packed_base + word] = value_word;
  }
}

#define QUANTIZED_PAGE_WRITE_KERNEL(NAME, TYPE) \
kernel void NAME( \
    const device TYPE* keys [[buffer(0)]], \
    const device TYPE* values [[buffer(1)]], \
    device uint* page_keys [[buffer(2)]], \
    device uint* page_values [[buffer(3)]], \
    device float* key_scales [[buffer(4)]], \
    device float* value_scales [[buffer(5)]], \
    const device uint* page_table [[buffer(6)]], \
    constant QuantizedPageWriteParameters& parameters [[buffer(7)]], \
    uint3 position [[thread_position_in_grid]]) { \
  quantized_page_write(keys, values, page_keys, page_values, key_scales, value_scales, \
      page_table, parameters, position.x, position.y, position.z); \
}

QUANTIZED_PAGE_WRITE_KERNEL(mirmir_quantized_page_write_f32, float)
QUANTIZED_PAGE_WRITE_KERNEL(mirmir_quantized_page_write_f16, half)
QUANTIZED_PAGE_WRITE_KERNEL(mirmir_quantized_page_write_bf16, bfloat)
