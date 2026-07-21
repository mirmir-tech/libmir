use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, TypedKernel, bf16, cuda_export, cuda_kernel_files,
};

use super::super::{GatedActivation, NvFp4BankView};
use crate::Result;

mod down;
mod execute;
mod gate;
mod spec;

pub use spec::{
    NvFp4MicroBanks, NvFp4MicroDownLaunch, NvFp4MicroDownWorkspace, NvFp4MicroGateLaunch,
    NvFp4MicroGateWorkspace, NvFp4MicroLaunch, NvFp4MicroSpec, NvFp4MicroWorkspace,
};

cuda_export!(QuantizePairKernel = "libmir_cuda_nvfp4_micro_quantize_pair"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    gate_globals: &DeviceBuffer<f32>, up_globals: &DeviceBuffer<f32>,
    gate_packed: &mut DeviceBuffer<u8>, up_packed: &mut DeviceBuffer<u8>,
    gate_scales: &mut DeviceBuffer<u8>, up_scales: &mut DeviceBuffer<u8>,
    groups: u32, selected_count: u32, columns: u32,
));

cuda_export!(Fc1Kernel = "libmir_cuda_nvfp4_micro_fc1"(
    gate_input: &DeviceBuffer<u8>, gate_input_scales: &DeviceBuffer<u8>,
    up_input: &DeviceBuffer<u8>, up_input_scales: &DeviceBuffer<u8>,
    selected: &DeviceBuffer<u32>, gate_weight: &DeviceBuffer<u8>,
    gate_weight_scales: &DeviceBuffer<u8>, gate_combined: &DeviceBuffer<f32>,
    up_weight: &DeviceBuffer<u8>, up_weight_scales: &DeviceBuffer<u8>,
    up_combined: &DeviceBuffer<f32>, down_input_globals: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<u8>, output_scales: &mut DeviceBuffer<u8>,
    groups: u32, input_features: u32, output_features: u32,
    output_scale_stride: u32, activation: u32,
));

cuda_export!(GatedQuantizeKernel = "libmir_cuda_nvfp4_micro_gated_quantize"(
    gate: &DeviceBuffer<bf16>, up: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    input_globals: &DeviceBuffer<f32>, output: &mut DeviceBuffer<u8>,
    output_scales: &mut DeviceBuffer<u8>, groups: u32, columns: u32, activation: u32,
));

cuda_export!(Fc2Kernel = "libmir_cuda_nvfp4_micro_fc2"(
    input: &DeviceBuffer<u8>, input_scales: &DeviceBuffer<u8>,
    selected: &DeviceBuffer<u32>, routing: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<u8>, weight_scales: &DeviceBuffer<u8>,
    combined_scales: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
    input_features: u32, output_features: u32, selected_count: u32, tokens: u32,
));

#[derive(Clone, Debug)]
pub struct NvFp4MicroKernels {
    quantize_pair: TypedKernel<QuantizePairKernel>,
    fc1: TypedKernel<Fc1Kernel>,
    down: NvFp4MicroDownKernels,
    spec: NvFp4MicroSpec,
}

#[derive(Clone, Debug)]
pub struct NvFp4MicroDownKernels {
    gated_quantize: TypedKernel<GatedQuantizeKernel>,
    fc2: TypedKernel<Fc2Kernel>,
    spec: NvFp4MicroSpec,
}

impl NvFp4MicroKernels {
    pub fn compile(compiler: &Compiler, spec: NvFp4MicroSpec) -> Result<Self> {
        spec.validate()?;
        let source = cuda_kernel_files!(
            "nvfp4_micro_gate.cu";
            "../../../../kernels/nvfp4_micro_common.cuh",
            "../../../../kernels/nvfp4_micro_gate.cu",
        );
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            quantize_pair: module.kernel()?,
            fc1: module.kernel()?,
            down: NvFp4MicroDownKernels::compile(compiler, spec)?,
            spec,
        })
    }
}

const fn activation(value: GatedActivation) -> u32 {
    match value {
        GatedActivation::GeluTanh => 0,
        GatedActivation::Silu => 1,
    }
}

fn validate_bank(
    bank: NvFp4BankView<'_>,
    experts: usize,
    input: usize,
    output: usize,
) -> Result<()> {
    spec::require_len("micro bank weight", experts * output * input / 2, bank.weight.len())?;
    spec::require_len("micro bank scales", experts * output * input / 16, bank.scales.len())?;
    spec::require_len("micro bank combined scales", experts, bank.combined.len())?;
    spec::require_len("micro bank input scales", experts, bank.input_globals.len())
}
