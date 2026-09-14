// SIMD fragment layout and accumulation order adapted from MLX 0.32.2
// steel/gemm/mma.h, loader.h and fp4.h. Copyright © 2024-2026 Apple Inc.
// See LICENSE-MLX.txt.
// BM=16, BN=64, BK=32, WM=1, WN=2. Inputs stay compact; each descriptor
// owns disjoint real rows from one expert. Empty descriptors do no GEMM work.
const uint tile = threadgroup_position_in_grid.y;
const uint first = tiles[tile * 3];
const uint end = tiles[tile * 3 + 1];
if (first == end) return;
const uint expert = tiles[tile * 3 + 2];
const uint col = threadgroup_position_in_grid.x * 64;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint warp = tid / 32;
const uint fm = ((lane / 4) & 4) + (lane / 2) % 4;
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
threadgroup T xs[16 * 40];
threadgroup T ws[64 * 40];
simdgroup_matrix<float, 8, 8> c0;
c0.thread_elements()[0] = 0;
c0.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c1;
c1.thread_elements()[0] = 0;
c1.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c2;
c2.thread_elements()[0] = 0;
c2.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c3;
c3.thread_elements()[0] = 0;
c3.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c4;
c4.thread_elements()[0] = 0;
c4.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c5;
c5.thread_elements()[0] = 0;
c5.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c6;
c6.thread_elements()[0] = 0;
c6.thread_elements()[1] = 0;
simdgroup_matrix<float, 8, 8> c7;
c7.thread_elements()[0] = 0;
c7.thread_elements()[1] = 0;
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
    #pragma unroll
    for (uint block = 0; block < 64; block += 32) {
    const size_t row = size_t(expert) * OUTPUT + col + block + wr;
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
            ws[(block + wr) * 40 + wc + p * 8 + j] = T(scale * value);
        }
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_matrix<float, 8, 8> a0;
        a0.thread_elements()[0] = float(xs[(fm + 0) * 40 + kk + fn]);
        a0.thread_elements()[1] = float(xs[(fm + 0) * 40 + kk + fn + 1]);
        simdgroup_matrix<float, 8, 8> a1;
        a1.thread_elements()[0] = float(xs[(fm + 8) * 40 + kk + fn]);
        a1.thread_elements()[1] = float(xs[(fm + 8) * 40 + kk + fn + 1]);
        simdgroup_matrix<float, 8, 8> b0;
        b0.thread_elements()[0] = float(ws[(warp * 8 + fn + 0) * 40 + kk + fm]);
        b0.thread_elements()[1] = float(ws[(warp * 8 + fn + 0 + 1) * 40 + kk + fm]);
        simdgroup_matrix<float, 8, 8> b1;
        b1.thread_elements()[0] = float(ws[(warp * 8 + fn + 16) * 40 + kk + fm]);
        b1.thread_elements()[1] = float(ws[(warp * 8 + fn + 16 + 1) * 40 + kk + fm]);
        simdgroup_matrix<float, 8, 8> b2;
        b2.thread_elements()[0] = float(ws[(warp * 8 + fn + 32) * 40 + kk + fm]);
        b2.thread_elements()[1] = float(ws[(warp * 8 + fn + 32 + 1) * 40 + kk + fm]);
        simdgroup_matrix<float, 8, 8> b3;
        b3.thread_elements()[0] = float(ws[(warp * 8 + fn + 48) * 40 + kk + fm]);
        b3.thread_elements()[1] = float(ws[(warp * 8 + fn + 48 + 1) * 40 + kk + fm]);
        simdgroup_multiply_accumulate(c0, a0, b0, c0);
        simdgroup_multiply_accumulate(c1, a0, b1, c1);
        simdgroup_multiply_accumulate(c2, a0, b2, c2);
        simdgroup_multiply_accumulate(c3, a0, b3, c3);
        simdgroup_multiply_accumulate(c7, a1, b3, c7);
        simdgroup_multiply_accumulate(c6, a1, b2, c6);
        simdgroup_multiply_accumulate(c5, a1, b1, c5);
        simdgroup_multiply_accumulate(c4, a1, b0, c4);
    }
}
if (first + fm + 0 < end) {
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 0] = T(c0.thread_elements()[0]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 1] = T(c0.thread_elements()[1]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 16] = T(c1.thread_elements()[0]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 17] = T(c1.thread_elements()[1]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 32] = T(c2.thread_elements()[0]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 33] = T(c2.thread_elements()[1]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 48] = T(c3.thread_elements()[0]);
    output[(size_t(first) + fm + 0) * OUTPUT + col + warp * 8 + fn + 49] = T(c3.thread_elements()[1]);
}
if (first + fm + 8 < end) {
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 0] = T(c4.thread_elements()[0]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 1] = T(c4.thread_elements()[1]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 16] = T(c5.thread_elements()[0]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 17] = T(c5.thread_elements()[1]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 32] = T(c6.thread_elements()[0]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 33] = T(c6.thread_elements()[1]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 48] = T(c7.thread_elements()[0]);
    output[(size_t(first) + fm + 8) * OUTPUT + col + warp * 8 + fn + 49] = T(c7.thread_elements()[1]);
}
