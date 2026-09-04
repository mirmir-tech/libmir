#include <metal_stdlib>
using namespace metal;

struct PageWriteParameters {
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

template <typename T>
inline void write_page(
    const device T* keys,
    const device T* values,
    device T* page_keys,
    device T* page_values,
    const device uint* page_table,
    constant PageWriteParameters& parameters,
    uint index) {
  uint dimension = index % parameters.head_dim;
  uint token_head = index / parameters.head_dim;
  uint head = token_head % parameters.kv_heads;
  uint token = token_head / parameters.kv_heads;
  uint absolute = parameters.offset + token;
  uint page = page_table[absolute / parameters.page_size];
  uint in_page = absolute % parameters.page_size;
  uint key_source = head * parameters.key_head_stride +
      token * parameters.key_sequence_stride + dimension * parameters.key_dimension_stride;
  uint value_source = head * parameters.value_head_stride +
      token * parameters.value_sequence_stride + dimension * parameters.value_dimension_stride;
  uint target = ((head * parameters.page_capacity + page) * parameters.page_size + in_page) *
      parameters.head_dim + dimension;
  page_keys[target] = keys[key_source];
  page_values[target] = values[value_source];
}

#define PAGE_WRITE_KERNEL(NAME, TYPE) \
kernel void NAME( \
    const device TYPE* keys [[buffer(0)]], \
    const device TYPE* values [[buffer(1)]], \
    device TYPE* page_keys [[buffer(2)]], \
    device TYPE* page_values [[buffer(3)]], \
    const device uint* page_table [[buffer(4)]], \
    constant PageWriteParameters& parameters [[buffer(5)]], \
    uint index [[thread_position_in_grid]]) { \
  write_page(keys, values, page_keys, page_values, page_table, parameters, index); \
}

PAGE_WRITE_KERNEL(mirmir_page_write_f32, float)
PAGE_WRITE_KERNEL(mirmir_page_write_f16, half)
PAGE_WRITE_KERNEL(mirmir_page_write_bf16, bfloat)

struct PageCopyParameters {
  uint source;
  uint target;
  uint capacity;
  uint page_elements;
};

// Source pages are immutable/shared; the destination is exclusively reserved.
// Copy bits, so this also supports packed quantized words and FP32 scales.
#define PAGE_COPY_KERNEL(NAME, TYPE) \
kernel void NAME( \
    device TYPE* keys [[buffer(0)]], \
    device TYPE* values [[buffer(1)]], \
    constant PageCopyParameters& p [[buffer(2)]], \
    uint index [[thread_position_in_grid]]) { \
  uint head = index / p.page_elements; \
  uint element = index % p.page_elements; \
  uint source = (head * p.capacity + p.source) * p.page_elements + element; \
  uint target = (head * p.capacity + p.target) * p.page_elements + element; \
  keys[target] = keys[source]; \
  values[target] = values[source]; \
}

PAGE_COPY_KERNEL(mirmir_page_copy_16, ushort)
PAGE_COPY_KERNEL(mirmir_page_copy_32, uint)
