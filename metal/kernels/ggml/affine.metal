// GGML rows whose 32-value sub-blocks share one scale and minimum, as MLX
// affine groups of 32: value scale * q - minimum, with q in
// [-OFFSET, 2^BITS - OFFSET), becomes scale * u + bias for u = q + OFFSET.
// Eight values pack into BITS bytes, value j at bit BITS * j, so a row is one
// little-endian bit stream: MLX's layout for 4-, 5- and 8-bit weights.
// Grid: [INPUT / 32, rows, 1].
constexpr uint SUBS = INPUT / 32;
const uint s = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (s >= SUBS) return;
GgmlSub sub;
ggml_decode_row<TYPE, INPUT>(weight, row, s, sub);
uint words[BITS];
for (uint w = 0; w < BITS; ++w) words[w] = 0;
for (uint i = 0; i < 32; ++i) {
  const uint u = uint(int(as_type<char>(uchar(ggml_byte(sub.q, i)))) + int(OFFSET));
  const uint bit = BITS * i;
  words[bit / 32] |= u << (bit % 32);
  if (bit % 32 + BITS > 32) words[bit / 32 + 1] |= u >> (32 - bit % 32);
}
device uint* out = packed + (ulong(row) * SUBS + s) * BITS;
for (uint w = 0; w < BITS; ++w) out[w] = words[w];
// The bias derives from the rounded scale: with a power-of-two offset and no
// minimum it is then exact, so a value is the rounded scale times its integer
// instead of a difference of two separately rounded terms.
const ulong group = ulong(row) * SUBS + s;
const T scale = T(sub.scale.x);
scales[group] = scale;
biases[group] = T(-float(scale) * float(OFFSET) - sub.minimum.x);
