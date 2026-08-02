use mircuda::{DeviceBuffer, bf16, cuda_export};

macro_rules! pair_kernel {
    ($name:ident, $symbol:literal) => {
        cuda_export!(
            pub(super) $name = $symbol(
                input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
                gate_weight: &DeviceBuffer<u32>, gate_scales: &DeviceBuffer<bf16>,
                gate_biases: &DeviceBuffer<bf16>, up_weight: &DeviceBuffer<u32>,
                up_scales: &DeviceBuffer<bf16>, up_biases: &DeviceBuffer<bf16>,
                gate_output: &mut DeviceBuffer<bf16>, up_output: &mut DeviceBuffer<bf16>,
                input_features: u32, output_features: u32, group_size: u32, expert_count: u32,
            )
        );
    };
}

macro_rules! gated_kernel {
    ($name:ident, $symbol:literal) => {
        cuda_export!(
            pub(super) $name = $symbol(
                input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
                gate_weight: &DeviceBuffer<u32>, gate_scales: &DeviceBuffer<bf16>,
                gate_biases: &DeviceBuffer<bf16>, up_weight: &DeviceBuffer<u32>,
                up_scales: &DeviceBuffer<bf16>, up_biases: &DeviceBuffer<bf16>,
                output: &mut DeviceBuffer<bf16>, input_features: u32,
                output_features: u32, group_size: u32, expert_count: u32, activation: u32,
            )
        );
    };
}

macro_rules! reduce_kernel {
    ($name:ident, $symbol:literal) => {
        cuda_export!(
            pub(super) $name = $symbol(
                input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
                routing_weights: &DeviceBuffer<bf16>, weight: &DeviceBuffer<u32>,
                scales: &DeviceBuffer<bf16>, biases: &DeviceBuffer<bf16>,
                output: &mut DeviceBuffer<bf16>, input_features: u32,
                output_features: u32, group_size: u32, expert_count: u32, selected_count: u32,
            )
        );
    };
}

pair_kernel!(SelectedPairInt4Kernel, "libmir_cuda_selected_affine_pair_bf16_int4");
pair_kernel!(SelectedPairInt8Kernel, "libmir_cuda_selected_affine_pair_bf16_int8");
pair_kernel!(SelectedPairFallbackKernel, "libmir_cuda_selected_affine_pair_bf16_fallback");
gated_kernel!(SelectedGatedInt4Kernel, "libmir_cuda_selected_affine_gated_bf16_int4");
gated_kernel!(SelectedGatedInt8Kernel, "libmir_cuda_selected_affine_gated_bf16_int8");
gated_kernel!(SelectedGatedFallbackKernel, "libmir_cuda_selected_affine_gated_bf16_fallback");
reduce_kernel!(SelectedReduceInt4Kernel, "libmir_cuda_selected_affine_reduce_bf16_int4");
reduce_kernel!(SelectedReduceInt8Kernel, "libmir_cuda_selected_affine_reduce_bf16_int8");
reduce_kernel!(SelectedReduceFallbackKernel, "libmir_cuda_selected_affine_reduce_bf16_fallback");
