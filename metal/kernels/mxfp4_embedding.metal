constexpr float values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
device const uchar* packed_weight = reinterpret_cast<device const uchar*>(weight);
uint feature = thread_position_in_grid.x;
uint token = thread_position_in_grid.y;
uint groups = HIDDEN / 32;
uint row = indices[token];
uint group = feature / 32;
uint offset = feature % 32;
uchar packed = packed_weight[(row * groups + group) * 16 + offset / 2];
uint code = offset % 2 == 0 ? packed & 15 : packed >> 4;
float scale = ldexp(1.0f, int(scales[row * groups + group]) - 127);
output[token * HIDDEN + feature] = bfloat(values[code] * scale);
