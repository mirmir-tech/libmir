#include <cuda_bf16.h>

extern "C" __global__ void libmir_cuda_vision_patch_cthw_to_thwc_bf16(
    const float* input, __nv_bfloat16* output, unsigned int elements,
    unsigned int channels, unsigned int temporal, unsigned int patch_area) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned int channel = index % channels;
  const unsigned int spatial_index = (index / channels) % patch_area;
  const unsigned int temporal_index = (index / (channels * patch_area)) % temporal;
  const unsigned int token = index / (channels * patch_area * temporal);
  const unsigned int source =
      ((token * channels + channel) * temporal + temporal_index) * patch_area
      + spatial_index;
  output[index] = __float2bfloat16_rn(input[source]);
}
