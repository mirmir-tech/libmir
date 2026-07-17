uint index = thread_position_in_grid.x;
uint value_head = index % HV;
float parameter = float(alpha[index]) + float(dt_bias[value_head]);
float softplus = max(parameter, 0.0f) + log(1.0f + exp(-abs(parameter)));
decay[index] = exp(-exp(float(a_log[value_head])) * softplus);
update[index] = 1.0f / (1.0f + exp(-float(beta[index])));
