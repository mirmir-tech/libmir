uint batch = threadgroup_position_in_grid.z / BLOCKS;
const device T* key_pages = key_pages_0;
const device T* value_pages = value_pages_0;
switch (batch) {
  case 1: key_pages = key_pages_1; value_pages = value_pages_1; break;
  case 2: key_pages = key_pages_2; value_pages = value_pages_2; break;
  case 3: key_pages = key_pages_3; value_pages = value_pages_3; break;
  case 4: key_pages = key_pages_4; value_pages = value_pages_4; break;
  case 5: key_pages = key_pages_5; value_pages = value_pages_5; break;
  case 6: key_pages = key_pages_6; value_pages = value_pages_6; break;
  case 7: key_pages = key_pages_7; value_pages = value_pages_7; break;
  case 8: key_pages = key_pages_8; value_pages = value_pages_8; break;
  case 9: key_pages = key_pages_9; value_pages = value_pages_9; break;
  case 10: key_pages = key_pages_10; value_pages = value_pages_10; break;
  case 11: key_pages = key_pages_11; value_pages = value_pages_11; break;
}
PagedAttentionParameters parameters = {
    QUERY_HEADS, KV_HEADS, metadata[BATCH + batch], BLOCKS, PAGE_SIZE, uint(SCALE_BITS)
};
uint statistic = batch * QUERY_HEADS * BLOCKS;
uint3 group = threadgroup_position_in_grid;
group.z %= BLOCKS;
paged_attention_partial<T, HEAD_DIM>(
    queries + batch * QUERY_HEADS * HEAD_DIM, key_pages, value_pages,
    page_tables + batch * metadata[2 * BATCH], metadata[batch],
    partials + statistic * HEAD_DIM, sums + statistic, maximums + statistic,
    queries, parameters, thread_index_in_simdgroup,
    thread_position_in_threadgroup, group);
