__device__ __forceinline__ float dense_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ __forceinline__ unsigned short dense_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounding = 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>((bits + rounding) >> 16u);
}

__device__ __forceinline__ float dense_weight(
    const unsigned short* weight, unsigned int expert, unsigned int row,
    unsigned int column, unsigned int rows, unsigned int columns,
    unsigned int transposed) {
  const unsigned long long matrix =
      static_cast<unsigned long long>(expert) * rows * columns;
  const unsigned long long offset =
      transposed ? static_cast<unsigned long long>(column) * rows + row
                 : static_cast<unsigned long long>(row) * columns + column;
  return dense_bf16_to_float(weight[matrix + offset]);
}

__device__ __forceinline__ float dense_pair_dot(
    unsigned int values, unsigned int weights) {
  return dense_bf16_to_float(values & 0xffffu) *
             dense_bf16_to_float(weights & 0xffffu) +
         dense_bf16_to_float(values >> 16u) *
             dense_bf16_to_float(weights >> 16u);
}

__device__ __forceinline__ float dense_activate(
    float gate, float up, unsigned int activation, float alpha, float limit,
    float up_shift) {
  if (activation == 0u) {
    const float cube = gate * gate * gate;
    const float gelu =
        0.5f * gate *
        (1.0f + tanhf(0.7978845608f * (gate + 0.044715f * cube)));
    return gelu * up;
  }
  if (activation == 2u) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
  const float silu = gate / (1.0f + __expf(-alpha * gate));
  return silu * (up + up_shift);
}

extern "C" __global__ void libmir_cuda_selected_dense_gated_bf16(
    const unsigned short* input, const unsigned int* selected,
    const unsigned short* gate_weight, const unsigned short* gate_bias,
    const unsigned short* up_weight, const unsigned short* up_bias,
    unsigned short* output, unsigned int input_features,
    unsigned int output_features, unsigned int expert_count,
    unsigned int selected_count, unsigned int gate_up_layout,
    unsigned int gate_transposed, unsigned int up_transposed,
    unsigned int has_gate_bias, unsigned int has_up_bias,
    unsigned int activation, float alpha, float limit, float up_shift) {
  if (gate_transposed != 0u && up_transposed != 0u &&
      gate_up_layout == 2u) {
    extern __shared__ float partials[];
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * 32u + lane;
    const unsigned int fused_row = row * 2u;
    const unsigned int selection = blockIdx.z * selected_count + blockIdx.y;
    const unsigned int expert = selected[selection];
    const unsigned int fused_rows = output_features * 2u;
    const bool active = row < output_features && expert < expert_count;
    const unsigned short* token_input = input + blockIdx.z * input_features;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (active) {
      for (unsigned int column = warp; column < input_features; column += 2u) {
        const unsigned long long matrix =
            static_cast<unsigned long long>(expert) * fused_rows *
            input_features;
        const unsigned long long offset =
            matrix + static_cast<unsigned long long>(column) * fused_rows +
            fused_row;
        const unsigned int pair =
            *reinterpret_cast<const unsigned int*>(gate_weight + offset);
        const float value = dense_bf16_to_float(token_input[column]);
        gate_sum += value * dense_bf16_to_float(pair & 0xffffu);
        up_sum += value * dense_bf16_to_float(pair >> 16u);
      }
    }
    partials[warp * 32u + lane] = gate_sum;
    partials[(blockDim.y + warp) * 32u + lane] = up_sum;
    __syncthreads();
    if (warp == 0u) {
      gate_sum += partials[32u + lane];
      up_sum += partials[(blockDim.y + 1u) * 32u + lane];
      if (active) {
        if (has_gate_bias != 0u) {
          gate_sum += dense_bf16_to_float(
              gate_bias[expert * fused_rows + fused_row]);
        }
        if (has_up_bias != 0u) {
          up_sum += dense_bf16_to_float(
              up_bias[expert * fused_rows + fused_row + 1u]);
        }
      }
      if (row < output_features) {
        const float gate = active
                               ? dense_bf16_to_float(
                                     dense_float_to_bf16(gate_sum))
                               : 0.0f;
        const float up = active
                             ? dense_bf16_to_float(
                                   dense_float_to_bf16(up_sum))
                             : 0.0f;
        const unsigned int output_index =
            selection * output_features + row;
        output[output_index] =
            expert < expert_count
                ? dense_float_to_bf16(dense_activate(
                      gate, up, activation, alpha, limit, up_shift))
                : 0;
      }
    }
    return;
  }
  const unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
  const unsigned int slot = blockIdx.y;
  const unsigned int token = blockIdx.z;
  if (row >= output_features) return;
  const unsigned int selection = token * selected_count + slot;
  const unsigned int expert = selected[selection];
  const unsigned int output_index = selection * output_features + row;
  if (expert >= expert_count) {
    if (threadIdx.x == 0) output[output_index] = 0;
    return;
  }

  const unsigned int fused_rows = output_features * 2u;
  const unsigned int gate_row =
      gate_up_layout == 2u ? row * 2u : row;
  const unsigned int up_row =
      gate_up_layout == 2u ? row * 2u + 1u : row + output_features;
  const unsigned int gate_rows = gate_up_layout == 0u ? output_features : fused_rows;
  const unsigned int up_rows = gate_up_layout == 0u ? output_features : fused_rows;
  const unsigned short* token_input = input + token * input_features;
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < input_features;
       column += 32u) {
    const float value = dense_bf16_to_float(token_input[column]);
    gate_sum += value * dense_weight(
        gate_weight, expert, gate_row, column, gate_rows, input_features,
        gate_transposed);
    up_sum += value * dense_weight(
        up_weight, expert, gate_up_layout == 0u ? row : up_row, column,
        up_rows, input_features, up_transposed);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_sum += __shfl_down_sync(0xffffffffu, gate_sum, offset);
    up_sum += __shfl_down_sync(0xffffffffu, up_sum, offset);
  }
  if (threadIdx.x == 0) {
    if (has_gate_bias != 0u) {
      gate_sum += dense_bf16_to_float(
          gate_bias[expert * gate_rows + gate_row]);
    }
    if (has_up_bias != 0u) {
      const unsigned int bias_row = gate_up_layout == 0u ? row : up_row;
      up_sum += dense_bf16_to_float(up_bias[expert * up_rows + bias_row]);
    }
    const float gate =
        dense_bf16_to_float(dense_float_to_bf16(gate_sum));
    const float up = dense_bf16_to_float(dense_float_to_bf16(up_sum));
    output[output_index] =
        dense_float_to_bf16(dense_activate(
            gate, up, activation, alpha, limit, up_shift));
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_reduce_bf16(
    const unsigned short* input, const unsigned int* selected,
    const unsigned short* routing, const unsigned short* weight,
    const unsigned short* bias, unsigned short* output,
    unsigned int input_features, unsigned int output_features,
    unsigned int expert_count, unsigned int selected_count,
    unsigned int transposed, unsigned int has_bias) {
  if (transposed != 0u) {
    extern __shared__ float partials[];
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * 32u + lane;
    const unsigned int token = blockIdx.y;
    const bool active = row < output_features;
    float total = 0.0f;
    for (unsigned int slot = 0; slot < selected_count; ++slot) {
      const unsigned int selection = token * selected_count + slot;
      const unsigned int expert = selected[selection];
      float sum = 0.0f;
      if (active && expert < expert_count) {
        const unsigned short* selected_input =
            input + selection * input_features;
        for (unsigned int column = warp; column < input_features;
             column += 8u) {
          sum += dense_bf16_to_float(selected_input[column]) *
                 dense_weight(
                     weight, expert, row, column, output_features,
                     input_features, 1u);
        }
      }
      partials[warp * 32u + lane] = sum;
      __syncthreads();
      if (warp == 0u && active && expert < expert_count) {
        float even = partials[lane] + partials[4u * 32u + lane];
        even += partials[2u * 32u + lane] +
                partials[6u * 32u + lane];
        float odd = partials[32u + lane] +
                    partials[5u * 32u + lane];
        odd += partials[3u * 32u + lane] +
               partials[7u * 32u + lane];
        sum = even + odd;
        if (has_bias != 0u) {
          sum +=
              dense_bf16_to_float(bias[expert * output_features + row]);
        }
        const float projected =
            dense_bf16_to_float(dense_float_to_bf16(sum));
        total += projected * dense_bf16_to_float(routing[selection]);
      }
      __syncthreads();
    }
    if (warp == 0u && active) {
      output[token * output_features + row] = dense_float_to_bf16(total);
    }
    return;
  }
  const unsigned int row = blockIdx.x * 8u + threadIdx.y;
  const unsigned int token = blockIdx.y;
  if (row >= output_features) return;
  float total = 0.0f;
  for (unsigned int slot = 0; slot < selected_count; ++slot) {
    const unsigned int selection = token * selected_count + slot;
    const unsigned int expert = selected[selection];
    float sum = 0.0f;
    if (expert < expert_count) {
      const unsigned short* selected_input =
          input + selection * input_features;
      for (unsigned int column = threadIdx.x; column < input_features;
           column += 32u) {
        sum += dense_bf16_to_float(selected_input[column]) *
               dense_weight(
                   weight, expert, row, column, output_features,
                   input_features, transposed);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0 && expert < expert_count) {
      if (has_bias != 0u) {
        sum += dense_bf16_to_float(
            bias[expert * output_features + row]);
      }
      const float projected =
          dense_bf16_to_float(dense_float_to_bf16(sum));
      total += projected * dense_bf16_to_float(routing[selection]);
    }
  }
  if (threadIdx.x == 0) {
    output[token * output_features + row] = dense_float_to_bf16(total);
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_project_bf16(
    const unsigned short* input, const unsigned int* selected,
    const unsigned short* routing, const unsigned short* weight,
    const unsigned short* bias, float* partial, unsigned int input_features,
    unsigned int output_features, unsigned int expert_count,
    unsigned int selected_count, unsigned int has_bias,
    unsigned int transposed) {
  if (transposed == 0u) {
    const unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int selection =
        blockIdx.z * selected_count + blockIdx.y;
    const unsigned int expert = selected[selection];
    const bool active = row < output_features && expert < expert_count;
    float sum = 0.0f;
    if (active) {
      const unsigned short* selected_input =
          input + selection * input_features;
      const unsigned long long matrix =
          static_cast<unsigned long long>(expert) * output_features *
          input_features;
      const unsigned short* row_weight =
          weight + matrix +
          static_cast<unsigned long long>(row) * input_features;
      if ((input_features & 1u) == 0u) {
        for (unsigned int column = threadIdx.x * 2u;
             column < input_features; column += 64u) {
          const unsigned int values =
              *reinterpret_cast<const unsigned int*>(
                  selected_input + column);
          const unsigned int weights =
              *reinterpret_cast<const unsigned int*>(
                  row_weight + column);
          sum += dense_pair_dot(values, weights);
        }
      } else {
        for (unsigned int column = threadIdx.x; column < input_features;
             column += 32u) {
          sum += dense_bf16_to_float(selected_input[column]) *
                 dense_bf16_to_float(row_weight[column]);
        }
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0u && row < output_features) {
      if (active && has_bias != 0u) {
        sum += dense_bf16_to_float(
            bias[expert * output_features + row]);
      }
      const float projected =
          active ? dense_bf16_to_float(dense_float_to_bf16(sum)) : 0.0f;
      partial[selection * output_features + row] =
          projected * dense_bf16_to_float(routing[selection]);
    }
    return;
  }
  extern __shared__ float sums[];
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const bool paired = (output_features & 1u) == 0u;
  const unsigned int rows_per_block = paired ? 64u : 32u;
  const unsigned int row =
      blockIdx.x * rows_per_block + lane * (paired ? 2u : 1u);
  const unsigned int selection =
      blockIdx.z * selected_count + blockIdx.y;
  const unsigned int expert = selected[selection];
  const bool active = row < output_features && expert < expert_count;
  float partial_sum[2] = {};
  if (active) {
    const unsigned short* selected_input =
        input + selection * input_features;
    for (unsigned int column = warp; column < input_features;
         column += 8u) {
      const float value = dense_bf16_to_float(selected_input[column]);
      const unsigned long long matrix =
          static_cast<unsigned long long>(expert) * output_features *
          input_features;
      const unsigned long long offset =
          matrix + static_cast<unsigned long long>(column) *
                       output_features + row;
      if (paired) {
        const unsigned int weights =
            *reinterpret_cast<const unsigned int*>(weight + offset);
        partial_sum[0] += value * dense_bf16_to_float(weights & 0xffffu);
        partial_sum[1] += value * dense_bf16_to_float(weights >> 16u);
      } else {
        partial_sum[0] +=
            value * dense_bf16_to_float(weight[offset]);
      }
    }
  }
  const unsigned int items = paired ? 2u : 1u;
  for (unsigned int item = 0u; item < items; ++item) {
    sums[(warp * items + item) * 32u + lane] = partial_sum[item];
  }
  __syncthreads();
  if (warp == 0u) {
    for (unsigned int item = 0u; item < items; ++item) {
      const unsigned int output_row = row + item;
      if (output_row >= output_features) continue;
      float even = sums[item * 32u + lane] +
                   sums[(4u * items + item) * 32u + lane];
      even += sums[(2u * items + item) * 32u + lane] +
              sums[(6u * items + item) * 32u + lane];
      float odd = sums[(items + item) * 32u + lane] +
                  sums[(5u * items + item) * 32u + lane];
      odd += sums[(3u * items + item) * 32u + lane] +
             sums[(7u * items + item) * 32u + lane];
      float sum = even + odd;
      if (active && has_bias != 0u) {
        sum += dense_bf16_to_float(
            bias[expert * output_features + output_row]);
      }
      const float projected =
          active ? dense_bf16_to_float(dense_float_to_bf16(sum)) : 0.0f;
      partial[selection * output_features + output_row] =
          projected * dense_bf16_to_float(routing[selection]);
    }
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_finalize_bf16(
    const float* partial, unsigned short* output,
    unsigned int output_features, unsigned int selected_count) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int token = blockIdx.y;
  if (row >= output_features) return;
  float total = 0.0f;
  const unsigned int first = token * selected_count;
  for (unsigned int slot = 0u; slot < selected_count; ++slot) {
    total += partial[(first + slot) * output_features + row];
  }
  output[token * output_features + row] = dense_float_to_bf16(total);
}

extern "C" __global__ void libmir_cuda_selected_dense_dispatch_clear(
    unsigned int* counts, unsigned int* cursors, unsigned int experts) {
  const unsigned int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert < experts) {
    counts[expert] = 0u;
    cursors[expert] = 0u;
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_dispatch_count(
    const unsigned int* selected, unsigned int* counts,
    unsigned int assignments, unsigned int experts) {
  const unsigned int assignment = blockIdx.x * blockDim.x + threadIdx.x;
  if (assignment < assignments) {
    const unsigned int expert = selected[assignment];
    if (expert < experts) atomicAdd(counts + expert, 1u);
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_dispatch_prefix(
    const unsigned int* counts, unsigned int* offsets, unsigned int* cursors,
    unsigned int experts) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  unsigned int offset = 0u;
  for (unsigned int expert = 0u; expert < experts; ++expert) {
    offsets[expert] = offset;
    cursors[expert] = 0u;
    offset += counts[expert];
  }
}

extern "C" __global__ void libmir_cuda_selected_dense_dispatch_scatter(
    const unsigned int* selected, const unsigned int* offsets,
    unsigned int* cursors, unsigned int* assignments_out,
    unsigned int* experts_out, unsigned int assignments,
    unsigned int experts) {
  const unsigned int assignment = blockIdx.x * blockDim.x + threadIdx.x;
  if (assignment >= assignments) return;
  const unsigned int expert = selected[assignment];
  if (expert >= experts) return;
  const unsigned int target =
      offsets[expert] + atomicAdd(cursors + expert, 1u);
  assignments_out[target] = assignment;
  experts_out[target] = expert;
}

extern "C" __global__ void libmir_cuda_selected_dense_gated_expert_major_bf16(
    const unsigned short* input, const unsigned int* assignments,
    const unsigned int* experts, const unsigned short* gate_up_weight,
    const unsigned short* gate_up_bias, unsigned short* output,
    unsigned int input_features, unsigned int output_features,
    unsigned int selected_count, unsigned int has_gate_bias,
    unsigned int has_up_bias, unsigned int activation, float alpha,
    float limit, float up_shift) {
  extern __shared__ float partials[];
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int row = blockIdx.x * 32u + lane;
  const unsigned int assignment = assignments[blockIdx.z];
  const unsigned int expert = experts[blockIdx.z];
  const unsigned int token = assignment / selected_count;
  const unsigned int fused_row = row * 2u;
  const unsigned int fused_rows = output_features * 2u;
  const bool active = row < output_features;
  const unsigned short* token_input = input + token * input_features;
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  if (active) {
    for (unsigned int column = warp; column < input_features; column += 8u) {
      const unsigned long long matrix =
          static_cast<unsigned long long>(expert) * fused_rows *
          input_features;
      const unsigned long long offset =
          matrix + static_cast<unsigned long long>(column) * fused_rows +
          fused_row;
      const unsigned int pair =
          *reinterpret_cast<const unsigned int*>(gate_up_weight + offset);
      const float value = dense_bf16_to_float(token_input[column]);
      gate_sum += value * dense_bf16_to_float(pair & 0xffffu);
      up_sum += value * dense_bf16_to_float(pair >> 16u);
    }
  }
  partials[warp * 32u + lane] = gate_sum;
  partials[(blockDim.y + warp) * 32u + lane] = up_sum;
  __syncthreads();
  if (warp != 0u || !active) return;
  float gate_even = partials[lane] + partials[4u * 32u + lane];
  gate_even += partials[2u * 32u + lane] + partials[6u * 32u + lane];
  float gate_odd = partials[32u + lane] + partials[5u * 32u + lane];
  gate_odd += partials[3u * 32u + lane] + partials[7u * 32u + lane];
  float up_even = partials[8u * 32u + lane] + partials[12u * 32u + lane];
  up_even += partials[10u * 32u + lane] + partials[14u * 32u + lane];
  float up_odd = partials[9u * 32u + lane] + partials[13u * 32u + lane];
  up_odd += partials[11u * 32u + lane] + partials[15u * 32u + lane];
  gate_sum = gate_even + gate_odd;
  up_sum = up_even + up_odd;
  if (has_gate_bias != 0u)
    gate_sum += dense_bf16_to_float(
        gate_up_bias[expert * fused_rows + fused_row]);
  if (has_up_bias != 0u)
    up_sum += dense_bf16_to_float(
        gate_up_bias[expert * fused_rows + fused_row + 1u]);
  const float gate =
      dense_bf16_to_float(dense_float_to_bf16(gate_sum));
  const float up = dense_bf16_to_float(dense_float_to_bf16(up_sum));
  output[assignment * output_features + row] =
      dense_float_to_bf16(
          dense_activate(gate, up, activation, alpha, limit, up_shift));
}

extern "C" __global__ void libmir_cuda_selected_dense_project_expert_major_bf16(
    const unsigned short* input, const unsigned int* assignments,
    const unsigned int* experts, const unsigned short* routing,
    const unsigned short* weight, const unsigned short* bias, float* partial,
    unsigned int input_features, unsigned int output_features,
    unsigned int selected_count, unsigned int has_bias) {
  extern __shared__ float sums[];
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int row = blockIdx.x * 32u + lane;
  const unsigned int assignment = assignments[blockIdx.z];
  const unsigned int expert = experts[blockIdx.z];
  const bool active = row < output_features;
  float sum = 0.0f;
  if (active) {
    const unsigned short* selected_input =
        input + assignment * input_features;
    for (unsigned int column = warp; column < input_features;
         column += 8u) {
      sum += dense_bf16_to_float(selected_input[column]) *
             dense_weight(
                 weight, expert, row, column, output_features,
                 input_features, 1u);
    }
  }
  sums[warp * 32u + lane] = sum;
  __syncthreads();
  if (warp == 0u && active) {
    float even = sums[lane] + sums[4u * 32u + lane];
    even += sums[2u * 32u + lane] + sums[6u * 32u + lane];
    float odd = sums[32u + lane] + sums[5u * 32u + lane];
    odd += sums[3u * 32u + lane] + sums[7u * 32u + lane];
    sum = even + odd;
    if (has_bias != 0u)
      sum += dense_bf16_to_float(
          bias[expert * output_features + row]);
    const float projected =
        dense_bf16_to_float(dense_float_to_bf16(sum));
    partial[assignment * output_features + row] =
        projected * dense_bf16_to_float(routing[assignment]);
  }
}

extern "C" __global__ void libmir_cuda_dense_expert_canonicalize_bf16(
    const unsigned short* input, unsigned short* output,
    unsigned int matrices, unsigned int input_features,
    unsigned int output_features) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long matrix_elements =
      static_cast<unsigned long long>(input_features) * output_features;
  const unsigned long long elements =
      static_cast<unsigned long long>(matrices) * matrix_elements;
  if (index >= elements) return;
  const unsigned long long matrix = index / matrix_elements;
  const unsigned long long local = index - matrix * matrix_elements;
  const unsigned int row = local / input_features;
  const unsigned int column = local - row * input_features;
  const unsigned long long source =
      matrix * matrix_elements +
      static_cast<unsigned long long>(column) * output_features + row;
  output[index] = input[source];
}

extern "C" __global__ void libmir_cuda_selected_dense_compact_bf16(
    const unsigned short* input, const unsigned int* assignments,
    unsigned short* output, unsigned int features,
    unsigned int selected_count, unsigned int routes) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long elements =
      static_cast<unsigned long long>(routes) * features;
  if (index >= elements) return;
  const unsigned int route = index / features;
  const unsigned int column = index - static_cast<unsigned long long>(route) * features;
  const unsigned int token = assignments[route] / selected_count;
  output[index] =
      input[static_cast<unsigned long long>(token) * features + column];
}

extern "C" __global__ void libmir_cuda_selected_dense_fill_bias_bf16(
    const unsigned int* experts, const unsigned short* bias,
    unsigned short* output, unsigned int features, unsigned int routes,
    unsigned int has_bias) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long elements =
      static_cast<unsigned long long>(routes) * features;
  if (index >= elements) return;
  const unsigned int route = index / features;
  const unsigned int column = index - static_cast<unsigned long long>(route) * features;
  output[index] =
      has_bias != 0u ? bias[experts[route] * features + column] : 0u;
}

extern "C" __global__ void libmir_cuda_selected_dense_activate_compact_bf16(
    const unsigned short* fused, unsigned short* output,
    unsigned int features, unsigned int routes, unsigned int activation,
    float alpha, float limit, float up_shift) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long elements =
      static_cast<unsigned long long>(routes) * features;
  if (index >= elements) return;
  const unsigned int route = index / features;
  const unsigned int column = index - static_cast<unsigned long long>(route) * features;
  const unsigned long long pair =
      static_cast<unsigned long long>(route) * features * 2u + column * 2u;
  const float gate = dense_bf16_to_float(fused[pair]);
  const float up = dense_bf16_to_float(fused[pair + 1u]);
  output[index] = dense_float_to_bf16(
      dense_activate(gate, up, activation, alpha, limit, up_shift));
}

extern "C" __global__ void libmir_cuda_selected_dense_route_compact_bf16(
    const unsigned short* input, const unsigned int* assignments,
    const unsigned short* routing, float* partial, unsigned int features,
    unsigned int routes) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long elements =
      static_cast<unsigned long long>(routes) * features;
  if (index >= elements) return;
  const unsigned int route = index / features;
  const unsigned int column = index - static_cast<unsigned long long>(route) * features;
  const unsigned int assignment = assignments[route];
  partial[static_cast<unsigned long long>(assignment) * features + column] =
      dense_bf16_to_float(input[index]) *
      dense_bf16_to_float(routing[assignment]);
}
