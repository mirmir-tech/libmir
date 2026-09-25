#include <cuda_bf16.h>
extern "C" __global__ void libmir_sampling_mask(
    const __nv_bfloat16* input, const unsigned int* mask,
    __nv_bfloat16* output, unsigned int vocab, unsigned int rows, unsigned int words) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab * rows) return;
    unsigned int row = idx / vocab, token = idx % vocab;
    bool allowed = (mask[row * words + token / 32] >> (token % 32)) & 1u;
    output[idx] = allowed ? input[idx] : __ushort_as_bfloat16(0xff80u);
}
