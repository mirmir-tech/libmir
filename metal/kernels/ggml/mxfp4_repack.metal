// GGML MXFP4 blocks (e, qs[16]: value j in the low nibble of qs[j], value
// j + 16 in its high nibble) to the OCP layout the native MXFP4 expert
// kernels read: byte k holds values 2k (low) and 2k + 1 (high), and the E8M0
// scales are a separate array. Grid: [BLOCKS, rows, 1].
const uint block = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (block >= BLOCKS) return;
const device uchar* source = weight + (ulong(row) * BLOCKS + block) * 17;
device uchar* out = blocks + (ulong(row) * BLOCKS + block) * 16;
scales[ulong(row) * BLOCKS + block] = source[0];
const device uchar* qs = source + 1;
for (uint k = 0; k < 8; ++k) {
  out[k] = (qs[2 * k] & 15) | (qs[2 * k + 1] & 15) << 4;
  out[k + 8] = (qs[2 * k] >> 4) | (qs[2 * k + 1] & 0xF0);
}
