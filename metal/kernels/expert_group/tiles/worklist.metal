// Sorted route IDs enter this test-only kernel; descriptors never leave the GPU.
threadgroup uint starts[EXPERTS];
threadgroup uint counts[EXPERTS];
threadgroup uint offsets[EXPERTS + 1];
const uint lane = thread_position_in_threadgroup.x;
uint lo = 0, hi = ROUTES;
while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (indices[mid] < lane) lo = mid + 1;
    else hi = mid;
}
starts[lane] = lo;
hi = ROUTES;
uint begin = lo;
while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (indices[mid] <= lane) lo = mid + 1;
    else hi = mid;
}
counts[lane] = lo - begin;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    offsets[0] = 0;
    for (uint expert = 0; expert < EXPERTS; ++expert)
        offsets[expert + 1] = offsets[expert] + (counts[expert] + 15) / 16;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint tile = lane; tile < CAPACITY; tile += EXPERTS) {
    uint start = 0, end = 0, expert = 0;
    if (tile < offsets[EXPERTS]) {
        uint left = 0, right = EXPERTS;
        while (left < right) {
            uint mid = (left + right) / 2;
            if (offsets[mid + 1] <= tile) left = mid + 1;
            else right = mid;
        }
        expert = left;
        start = starts[expert] + (tile - offsets[expert]) * 16;
        end = min(start + 16, starts[expert] + counts[expert]);
    }
    tiles[tile * 3] = start;
    tiles[tile * 3 + 1] = end;
    tiles[tile * 3 + 2] = expert;
}
