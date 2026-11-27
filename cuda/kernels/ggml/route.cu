// Routing around the grouped GGML GEMM (`ggml/mmq.cu`), composed after
// `ggml/vector.cu`: the quantized input rows of each selection in expert
// order, and each token's weighted sum of its selections' output rows.
// Quantized rows of `input` in expert order: row i is token
// `assignments[i] / top_k`.
extern "C" __global__ void libmir_cuda_ggml_mmq_gather_q8(
    const __nv_bfloat16* input, const unsigned int* assignments, unsigned int* quantized,
    float4* scales, unsigned int rows, unsigned int columns, unsigned int top_k) {
  const unsigned int subs = columns / kSub;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(rows) * subs) return;
  const unsigned long long row = index / subs;
  const unsigned long long token = assignments[row] / top_k;
  ggml_quantize_sub(
      input + (token * subs + index % subs) * kSub, quantized + index * 8u, scales + index);
}

// The expert-order row of each (token, selection).
extern "C" __global__ void libmir_cuda_ggml_mmq_invert(
    const unsigned int* assignments, unsigned int* rows_of_routes, unsigned int routes) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < routes) rows_of_routes[assignments[row]] = row;
}

// Each token's selections weighted by routing, eight features per thread.
extern "C" __global__ void libmir_cuda_ggml_mmq_combine_bf16(
    const __nv_bfloat16* rows, const unsigned int* rows_of_routes, const unsigned int* selected,
    const __nv_bfloat16* routing, __nv_bfloat16* output, unsigned int tokens,
    unsigned int top_k, unsigned int features, unsigned int experts) {
  const unsigned int vectors = features / 8u;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(tokens) * vectors) return;
  const unsigned long long token = index / vectors;
  const unsigned int vector = static_cast<unsigned int>(index % vectors);
  float total[8] = {};
  for (unsigned int k = 0u; k < top_k; ++k) {
    const unsigned long long route = token * top_k + k;
    if (selected[route] >= experts) continue;
    const float weight = __bfloat162float(routing[route]);
    const uint4 packed = reinterpret_cast<const uint4*>(
        rows + static_cast<unsigned long long>(rows_of_routes[route]) * features)[vector];
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (unsigned int i = 0; i < 4u; ++i) {
      const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
      total[2u * i] = fmaf(weight, pair.x, total[2u * i]);
      total[2u * i + 1u] = fmaf(weight, pair.y, total[2u * i + 1u]);
    }
  }
  unsigned int words[4];
#pragma unroll
  for (unsigned int i = 0; i < 4u; ++i) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(total[2u * i], total[2u * i + 1u]);
    words[i] = *reinterpret_cast<const unsigned int*>(&pair);
  }
  reinterpret_cast<uint4*>(output + token * features)[vector] =
      make_uint4(words[0], words[1], words[2], words[3]);
}
