use mircuda::{DeviceBuffer, bf16, cuda_export};

cuda_export!(
    pub(super) AffineGemvInt4Kernel = "libmir_cuda_affine_gemv_bf16_int4"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, group_size: u32, matrix_index: u32,
    )
);

cuda_export!(
    pub(super) AffineGemvInt8Kernel = "libmir_cuda_affine_gemv_bf16_int8"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, group_size: u32, matrix_index: u32,
    )
);

cuda_export!(
    pub(super) AffineGemvFallbackKernel = "libmir_cuda_affine_gemv_bf16_fallback"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>,
        scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, input_features: u32,
        output_features: u32, group_size: u32, matrix_index: u32,
    )
);
