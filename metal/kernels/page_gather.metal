uint column = thread_position_in_grid.x * 4;
uint token = thread_position_in_grid.y;
uint head = thread_position_in_grid.z % KV_HEADS;
uint batch = thread_position_in_grid.z / KV_HEADS;
uint tokens = metadata[0], pages = metadata[4];
if (column >= HEAD_DIM || token >= tokens || batch >= metadata[5]) return;
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
uint physical = tables[batch * pages + token / PAGE_SIZE];
uint source = ((head * metadata[6 + batch] + physical) * PAGE_SIZE + token % PAGE_SIZE) * HEAD_DIM + column;
uint output = ((batch * KV_HEADS + head) * tokens + token) * HEAD_DIM + column;
#pragma unroll
for (uint lane = 0; lane < 4; ++lane) {
  if (column + lane < HEAD_DIM) {
    keys[output + lane] = key_pages[source + lane];
    values[output + lane] = value_pages[source + lane];
  }
}
