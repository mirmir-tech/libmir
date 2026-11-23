//! One export of `ggml_gemv_gathered` per GGML type, all with one signature.

use mircuda::{DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16, cuda_export};
use models::gguf::GgmlType;

use super::GgmlKernelType;
use crate::{Error, Result};

type GatheredArguments<'a> = (
    &'a DeviceBuffer<u32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

macro_rules! gathered_kernels {
    ($($variant:ident = $symbol:literal, $kind:ident;)*) => {
        $(cuda_export!(pub(super) $variant = $symbol(
            quantized: &DeviceBuffer<u32>, scales: &DeviceBuffer<f32>, weight: &DeviceBuffer<u8>,
            selected: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>, selections: u32,
            group: u32, rows: u32, columns: u32, expert_rows: u32, first_row: u32,
        ));)*

        #[derive(Clone, Debug)]
        pub(super) enum GatheredKernel {
            $($variant(TypedKernel<$variant>),)*
        }

        impl GatheredKernel {
            pub(super) fn load(module: &Module, kind: GgmlKernelType) -> Result<Self> {
                match kind.ggml_type() {
                    $(GgmlType::$kind => Ok(Self::$variant(module.kernel()?)),)*
                    _ => Err(Error::InvalidDecoderKernel("no GGML kernel for this type")),
                }
            }

            pub(super) fn launch(
                &self,
                stream: &Stream,
                config: LaunchConfig,
                arguments: GatheredArguments<'_>,
            ) -> Result<()> {
                match self {
                    $(Self::$variant(kernel) => kernel.launch(stream, config, arguments)?,)*
                }
                Ok(())
            }
        }
    };
}

gathered_kernels! {
    Q40 = "libmir_cuda_ggml_gemv_gathered_bf16_q4_0", Q4_0;
    Q41 = "libmir_cuda_ggml_gemv_gathered_bf16_q4_1", Q4_1;
    Q50 = "libmir_cuda_ggml_gemv_gathered_bf16_q5_0", Q5_0;
    Q51 = "libmir_cuda_ggml_gemv_gathered_bf16_q5_1", Q5_1;
    Q80 = "libmir_cuda_ggml_gemv_gathered_bf16_q8_0", Q8_0;
    Q2K = "libmir_cuda_ggml_gemv_gathered_bf16_q2_k", Q2_K;
    Q3K = "libmir_cuda_ggml_gemv_gathered_bf16_q3_k", Q3_K;
    Q4K = "libmir_cuda_ggml_gemv_gathered_bf16_q4_k", Q4_K;
    Q5K = "libmir_cuda_ggml_gemv_gathered_bf16_q5_k", Q5_K;
    Q6K = "libmir_cuda_ggml_gemv_gathered_bf16_q6_k", Q6_K;
    IQ4NL = "libmir_cuda_ggml_gemv_gathered_bf16_iq4_nl", IQ4_NL;
    IQ4XS = "libmir_cuda_ggml_gemv_gathered_bf16_iq4_xs", IQ4_XS;
    IQ2XXS = "libmir_cuda_ggml_gemv_gathered_bf16_iq2_xxs", IQ2_XXS;
    IQ2XS = "libmir_cuda_ggml_gemv_gathered_bf16_iq2_xs", IQ2_XS;
    IQ2S = "libmir_cuda_ggml_gemv_gathered_bf16_iq2_s", IQ2_S;
    IQ3XXS = "libmir_cuda_ggml_gemv_gathered_bf16_iq3_xxs", IQ3_XXS;
    IQ3S = "libmir_cuda_ggml_gemv_gathered_bf16_iq3_s", IQ3_S;
    IQ1S = "libmir_cuda_ggml_gemv_gathered_bf16_iq1_s", IQ1_S;
    IQ1M = "libmir_cuda_ggml_gemv_gathered_bf16_iq1_m", IQ1_M;
    MXFP4 = "libmir_cuda_ggml_gemv_gathered_bf16_mxfp4", MXFP4;
    Q10 = "libmir_cuda_ggml_gemv_gathered_bf16_q1_0", Q1_0;
    Q20 = "libmir_cuda_ggml_gemv_gathered_bf16_q2_0", Q2_0;
    TQ10 = "libmir_cuda_ggml_gemv_gathered_bf16_tq1_0", TQ1_0;
    TQ20 = "libmir_cuda_ggml_gemv_gathered_bf16_tq2_0", TQ2_0;
    NVFP4 = "libmir_cuda_ggml_gemv_gathered_bf16_nvfp4", NVFP4;
}
