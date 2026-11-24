// Restores Hugging Face `rotate_half` row order in a query or key projection
// that llama.cpp's converter stored with each head's rotary halves
// interleaved pairwise: within the first `rotary_dim` rows r of a head,
// output row h*d + j*r/2 + i is source row h*d + 2i + j; the rows past the
// rotary part keep their place (GLM-4). Rows are opaque bytes, so dense and
// GGML block rows, and bias values, move alike.
extern "C" __global__ void libmir_cuda_rope_rows_u8(
    const unsigned char* source, unsigned char* output, unsigned int rows,
    unsigned int row_bytes, unsigned int head_dim, unsigned int rotary_dim) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned long long>(rows) * row_bytes) return;
  const unsigned long long row = index / row_bytes;
  const unsigned long long byte = index % row_bytes;
  const unsigned long long half = rotary_dim / 2u;
  const unsigned long long within = row % head_dim;
  const unsigned long long from =
      within < rotary_dim ? row - within + (within % half) * 2u + within / half : row;
  output[index] = source[from * row_bytes + byte];
}
