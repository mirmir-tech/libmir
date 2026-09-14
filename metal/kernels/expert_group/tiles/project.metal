// SIMD fragment layout and accumulation order adapted from MLX 0.32.2
// steel/gemm/mma.h, loader.h and fp4.h. Copyright © 2024-2026 Apple Inc.
// See LICENSE-MLX.txt.
// BM=16, BN=32, BK=32, WM=1, WN=2. Inputs stay compact; each descriptor
// owns disjoint real rows from one expert. Empty descriptors do no GEMM work.
const uint tile = threadgroup_position_in_grid.y;
const uint first = tiles[tile * 3];
const uint end = tiles[tile * 3 + 1];
if (first == end) return;
const uint expert = tiles[tile * 3 + 2];
const uint col = threadgroup_position_in_grid.x * 32;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint warp = tid / 32;
const uint fm = ((lane / 4) & 4) + (lane / 2) % 4;
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
threadgroup T xs[16 * 40];
threadgroup T ws[32 * 40];
simdgroup_matrix<float, 8, 8> accum[4];
for (uint i = 0; i < 4; ++i) {
    accum[i].thread_elements()[0] = 0;
    accum[i].thread_elements()[1] = 0;
}
struct InputVector { ushort values[8]; };
for (uint k = 0; k < INPUT; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint xr = tid / 4;
    const uint xc = (tid % 4) * 8;
    if (first + xr < end) {
        *reinterpret_cast<threadgroup InputVector*>(xs + xr * 40 + xc) =
            *reinterpret_cast<const device InputVector*>(input + (size_t(first) + xr) * INPUT + k + xc);
    } else {
        #pragma unroll
        for (uint j = 0; j < 8; ++j) xs[xr * 40 + xc + j] = T(0);
    }
    // Each thread dequantizes sixteen consecutive values, as in MLX's loader.
    const uint wr = tid / 2;
    const uint wc = (tid % 2) * 16;
    const size_t row = size_t(expert) * OUTPUT + col + wr;
    const uint exponent = scales[row * (INPUT / 32) + k / 32];
    const float scale = float(T(as_type<float>(exponent == 0 ? 0x00400000u
        : exponent << 23)));
    #pragma unroll
    for (uint p = 0; p < 2; ++p) {
        const uint packed = weight[row * (INPUT / 8) + (k + wc) / 8 + p];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint code = (packed >> (4 * j)) & 15;
            half decoded = as_type<half>(ushort((code & 7) << 9));
            decoded *= half(16384.0f);
            const float value = float((code & 8) ? -decoded : decoded);
            ws[wr * 40 + wc + p * 8 + j] = T(scale * value);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_matrix<float, 8, 8> a[2], b[2];
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            a[i].thread_elements()[0] = float(xs[(fm + i * 8) * 40 + kk + fn]);
            a[i].thread_elements()[1] = float(xs[(fm + i * 8) * 40 + kk + fn + 1]);
            b[i].thread_elements()[0] = float(ws[(warp * 8 + fn + i * 16) * 40 + kk + fm]);
            b[i].thread_elements()[1] = float(ws[(warp * 8 + fn + i * 16 + 1) * 40 + kk + fm]);
        }
        simdgroup_multiply_accumulate(accum[0], a[0], b[0], accum[0]);
        simdgroup_multiply_accumulate(accum[1], a[0], b[1], accum[1]);
        simdgroup_multiply_accumulate(accum[3], a[1], b[1], accum[3]);
        simdgroup_multiply_accumulate(accum[2], a[1], b[0], accum[2]);
    }
}
#pragma unroll
for (uint m = 0; m < 2; ++m) {
    const uint row = first + fm + m * 8;
    if (row < end) {
        #pragma unroll
        for (uint n = 0; n < 2; ++n) {
            const uint column = col + warp * 8 + fn + n * 16;
            output[size_t(row) * OUTPUT + column] = T(accum[m * 2 + n].thread_elements()[0]);
            output[size_t(row) * OUTPUT + column + 1] = T(accum[m * 2 + n].thread_elements()[1]);
        }
    }
}
