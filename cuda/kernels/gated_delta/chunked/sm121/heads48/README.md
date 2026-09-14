# SM121 chunked GDN, 16 key heads / 48 value heads

These ahead-of-time PTX kernels extend the existing 32-value-head implementation
for 128-dimensional keys and values. The runtime uses them for at least 64 tokens
(one complete algorithmic chunk), with BF16 activations and FP32 recurrent state.
Other geometries and shorter inputs retain their existing paths.

Generated with Workmir's `benchmarks/tools/qwen38/gdn/export.py`, vLLM 0.29.0,
PyTorch 2.13.0+cu130, and Triton 3.7.1 on an NVIDIA GB10. The generator fixes each
kernel's launch configuration and PTX version 8.8; the latter is needed by the
580.173.02 driver. Launch symbols, parameter types, warp counts and shared-memory
sizes match the existing native adapter. `manifest.json` records provenance and
the generator SHA-256. The empty `selected` lists reflect fixed configurations,
not missing exported kernels; each export requires exactly one compiled variant.

Only these generated PTX files are used at runtime. No Python, Triton or PyTorch
runtime dependency is introduced. Their Apache-2.0 OR MIT attribution is preserved
in each PTX header, as in the existing chunked kernels.

Validation compares outputs and final states against the serial recurrence on
nonzero pseudorandom data, at 64, 67, 128, 257 and 2048 tokens, including two
successive calls. The existing 32-head test is also retained and strengthened.

The native adapter must zero the inverse matrix before the triangular solve:
that kernel writes only its lower block triangle, and later operations read the
full matrix. The shared `initialize.cu` kernel enforces this for both head counts.
Tests poison reused scratch with NaNs to catch missing initialization.
