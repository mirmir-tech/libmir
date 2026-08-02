uint element = thread_position_in_grid.x;
uint token = element / HIDDEN;
uint dimension = element % HIDDEN;
float total = 0.0f;
for (uint selected = 0; selected < TOP_K; ++selected) {
  uint route = token * TOP_K + selected;
  uint sorted_route = inverse[route];
  total += float(weights[route]) * float(sorted[sorted_route * HIDDEN + dimension]);
}
output[element] = T(total);
