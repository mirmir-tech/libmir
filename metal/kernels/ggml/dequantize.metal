// GGML blocks to dense values: one thread per 32-value sub-block.
// Grid: [INPUT / 32, rows, 1].
const uint s = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (s >= INPUT / 32) return;
GgmlSub sub;
ggml_decode_row<TYPE, INPUT>(weight, row, s, sub);
device T* out = output + ulong(row) * INPUT + s * 32;
for (uint i = 0; i < 32; ++i) {
  out[i] = T(ggml_value(sub, i));
}
