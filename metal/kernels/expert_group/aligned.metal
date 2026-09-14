uint lane = thread_position_in_threadgroup.x;
threadgroup atomic_uint cursors[EXPERTS];
threadgroup uint starts[EXPERTS];
threadgroup uint owners[EXPERTS];
if (lane < EXPERTS) {
  atomic_store_explicit(&cursors[lane], 0, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint route = lane; route < ROUTES; route += threads_per_threadgroup.x) {
  atomic_fetch_add_explicit(&cursors[indices[route]], 1, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
  uint offset = 0, spent = 0, owner = 0;
  for (uint expert = 0; expert < EXPERTS; ++expert) {
    uint count = atomic_load_explicit(&cursors[expert], memory_order_relaxed);
    if (count != 0) {
      uint padding = (TILE - offset % TILE) % TILE;
      if (padding <= CAPACITY - ROUTES - spent) {
        offset += padding;
        spent += padding;
      }
      owner = expert;
    }
    starts[expert] = offset;
    owners[expert] = owner;
    atomic_store_explicit(&cursors[expert], offset, memory_order_relaxed);
    offset += count;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Padding uses a valid input and the preceding nonempty expert. Its output
// is never restored. Empty experts cannot introduce extra GEMM boundaries.
for (uint row = lane; row < CAPACITY; row += threads_per_threadgroup.x) {
  uint lo = 0, hi = EXPERTS;
  while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (starts[mid] <= row) lo = mid + 1;
    else hi = mid;
  }
  grouped_indices[row] = owners[lo - 1];
  order[row] = 0;
}
// All padding writes must precede the real route scatter in this threadgroup.
threadgroup_barrier(mem_flags::mem_device);
for (uint route = lane; route < ROUTES; route += threads_per_threadgroup.x) {
  uint destination =
      atomic_fetch_add_explicit(&cursors[indices[route]], 1, memory_order_relaxed);
  order[destination] = route;
  inverse[route] = destination;
}
