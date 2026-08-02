use mircuda::{DeviceBuffer, bf16, cuda_export};

macro_rules! qmm_kernel {
    ($name:ident, $symbol:literal) => {
        cuda_export!(
            pub(super) $name = $symbol(
                input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>,
                scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
                output: &mut DeviceBuffer<bf16>, tokens: u32,
                input_features: u32, output_features: u32, group_size: u32,
                matrix_index: u32,
            )
        );
    };
}

qmm_kernel!(AffineQmmInt4Kernel, "libmir_cuda_affine_qmm_bf16_int4");
qmm_kernel!(AffineQmmInt8Kernel, "libmir_cuda_affine_qmm_bf16_int8");
qmm_kernel!(AffineQmmScalarInt4Kernel, "libmir_cuda_affine_qmm_scalar_bf16_int4");
qmm_kernel!(AffineQmmScalarInt8Kernel, "libmir_cuda_affine_qmm_scalar_bf16_int8");
qmm_kernel!(AffineQmmFallbackKernel, "libmir_cuda_affine_qmm_scalar_bf16_fallback");
