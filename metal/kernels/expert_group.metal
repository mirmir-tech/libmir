uint lane = thread_position_in_threadgroup.x;
threadgroup atomic_uint cursors[EXPERTS];

if (lane < EXPERTS) {
  atomic_store_explicit(&cursors[lane], 0, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint route = lane; route < ROUTES; route += threads_per_threadgroup.x) {
  atomic_fetch_add_explicit(&cursors[indices[route]], 1, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0) {
  uint offset = 0;
  for (uint expert = 0; expert < EXPERTS; ++expert) {
    uint count = atomic_load_explicit(&cursors[expert], memory_order_relaxed);
    atomic_store_explicit(&cursors[expert], offset, memory_order_relaxed);
    offset += count;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint route = lane; route < ROUTES; route += threads_per_threadgroup.x) {
  uint destination =
      atomic_fetch_add_explicit(&cursors[indices[route]], 1, memory_order_relaxed);
  order[destination] = route;
  inverse[route] = destination;
}
