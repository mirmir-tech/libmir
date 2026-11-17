// Rows of a GGML embedding table selected by token ids: one thread per
// 32-value sub-block of one selected row. Ids outside the table yield zeros.
// Grid: [HIDDEN / 32, tokens, 1].
const uint s = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
if (s >= HIDDEN / 32) return;
const uint id = indices[token];
device T* out = output + ulong(token) * HIDDEN + s * 32;
if (id >= VOCABULARY) {
  for (uint i = 0; i < 32; ++i) out[i] = T(0.0f);
  return;
}
GgmlSub sub;
ggml_decode_row<TYPE, HIDDEN>(weight, id, s, sub);
for (uint i = 0; i < 32; ++i) {
  out[i] = T(ggml_value(sub, i));
}
