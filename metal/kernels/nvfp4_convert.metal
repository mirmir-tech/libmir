uint index = thread_position_in_grid.x;
uchar pair = weight[index >> 1];
uchar encoded = (index & 1u) == 0u ? pair & 0x0fu : pair >> 4;
uint magnitude = uint(encoded & 7u);
uint exponent = magnitude >> 1;
uint mantissa = magnitude & 1u;
float value = exponent == 0u
    ? float(mantissa) * 0.5f
    : ldexp(float(2u + mantissa), int(exponent) - 2);
if ((encoded & 8u) != 0u) {
  value = -value;
}
float scale = mirmir_e4m3_to_float(scales[index / 16u]);
output[index] = T(value * scale * global_scale[0]);
