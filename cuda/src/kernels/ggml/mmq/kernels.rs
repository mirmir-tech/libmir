//! The grouped GEMM exports of `kernels/ggml/mmq.cu`: one per GGML type,
//! all with one signature.

use mircuda::{DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16, cuda_export};

use super::super::GgmlKernelType;
use crate::{Error, Result};

pub(super) struct MmqArguments<'a> {
    pub(super) quantized: &'a DeviceBuffer<u32>,
    pub(super) scales: &'a DeviceBuffer<f32>,
    pub(super) offsets: &'a DeviceBuffer<u32>,
    pub(super) counts: &'a DeviceBuffer<u32>,
    pub(super) tile_experts: &'a DeviceBuffer<u32>,
    pub(super) tile_rows: &'a DeviceBuffer<u32>,
    pub(super) tile_count: &'a DeviceBuffer<u32>,
    pub(super) weight: &'a DeviceBuffer<u8>,
    pub(super) output: &'a mut DeviceBuffer<bf16>,
    pub(super) rows: u32,
    pub(super) columns: u32,
    pub(super) ld: u32,
    pub(super) stride: u32,
    pub(super) offset: u32,
}

macro_rules! mmq_kernels {
    ($($variant:ident = $symbol:literal, $kind:ident;)*) => {
        $(cuda_export!(pub(super) $variant = $symbol(
            quantized: &DeviceBuffer<u32>, scales: &DeviceBuffer<f32>,
            offsets: &DeviceBuffer<u32>, counts: &DeviceBuffer<u32>,
            tile_experts: &DeviceBuffer<u32>,
            tile_rows: &DeviceBuffer<u32>, tile_count: &DeviceBuffer<u32>,
            weight: &DeviceBuffer<u8>, output: &mut DeviceBuffer<bf16>, rows: u32,
            columns: u32, ld: u32, stride: u32, offset: u32,
        ));)*

        #[derive(Clone, Debug)]
        pub(super) enum MmqKernel {
            $($variant(TypedKernel<$variant>),)*
        }

        impl MmqKernel {
            pub(super) fn load(module: &Module, kind: GgmlKernelType) -> Result<Self> {
                match kind {
                    $(GgmlKernelType::$kind => Ok(Self::$variant(module.kernel()?)),)*
                    #[allow(unreachable_patterns)]
                    _ => Err(Error::InvalidDecoderKernel("no GGML grouped GEMM for this type")),
                }
            }

            pub(super) fn launch(
                &self,
                stream: &Stream,
                config: LaunchConfig,
                arguments: MmqArguments<'_>,
            ) -> Result<()> {
                let MmqArguments {
                    quantized, scales, offsets, counts, tile_experts, tile_rows, tile_count,
                    weight, output, rows, columns, ld, stride, offset,
                } = arguments;
                let arguments = (
                    quantized, scales, offsets, counts, tile_experts, tile_rows, tile_count,
                    weight, output, rows, columns, ld, stride, offset,
                );
                match self {
                    $(Self::$variant(kernel) => kernel.launch(stream, config, arguments)?,)*
                }
                Ok(())
            }
        }
    };
}

mmq_kernels! {
    Q40 = "libmir_cuda_ggml_mmq_bf16_q4_0", Q4_0;
    Q41 = "libmir_cuda_ggml_mmq_bf16_q4_1", Q4_1;
    Q50 = "libmir_cuda_ggml_mmq_bf16_q5_0", Q5_0;
    Q51 = "libmir_cuda_ggml_mmq_bf16_q5_1", Q5_1;
    Q80 = "libmir_cuda_ggml_mmq_bf16_q8_0", Q8_0;
    Q2K = "libmir_cuda_ggml_mmq_bf16_q2_k", Q2K;
    Q3K = "libmir_cuda_ggml_mmq_bf16_q3_k", Q3K;
    Q4K = "libmir_cuda_ggml_mmq_bf16_q4_k", Q4K;
    Q5K = "libmir_cuda_ggml_mmq_bf16_q5_k", Q5K;
    Q6K = "libmir_cuda_ggml_mmq_bf16_q6_k", Q6K;
    IQ4NL = "libmir_cuda_ggml_mmq_bf16_iq4_nl", IQ4NL;
    IQ4XS = "libmir_cuda_ggml_mmq_bf16_iq4_xs", IQ4XS;
    IQ2XXS = "libmir_cuda_ggml_mmq_bf16_iq2_xxs", IQ2XXS;
    IQ2XS = "libmir_cuda_ggml_mmq_bf16_iq2_xs", IQ2XS;
    IQ2S = "libmir_cuda_ggml_mmq_bf16_iq2_s", IQ2S;
    IQ3XXS = "libmir_cuda_ggml_mmq_bf16_iq3_xxs", IQ3XXS;
    IQ3S = "libmir_cuda_ggml_mmq_bf16_iq3_s", IQ3S;
    IQ1S = "libmir_cuda_ggml_mmq_bf16_iq1_s", IQ1S;
    IQ1M = "libmir_cuda_ggml_mmq_bf16_iq1_m", IQ1M;
    MXFP4 = "libmir_cuda_ggml_mmq_bf16_mxfp4", MXFP4;
}
