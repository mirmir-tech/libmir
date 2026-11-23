use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_files,
};
use models::gguf::GgmlType;

use super::{
    super::geometry::{narrow, product, require},
    GgmlKernelType, GgmlRowPlacement,
};
use crate::{Error, Result};

type Arguments<'a> = (&'a DeviceBuffer<u8>, &'a mut DeviceBuffer<bf16>, u32, u32, u32, u32);

macro_rules! dequantize_kernels {
    ($($variant:ident = $symbol:literal, $kind:ident;)*) => {
        $(cuda_export!($variant = $symbol(
            weight: &DeviceBuffer<u8>, output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
            stride: u32, offset: u32,
        ));)*

        #[derive(Clone, Debug)]
        enum Kernel {
            $($variant(TypedKernel<$variant>),)*
        }

        impl Kernel {
            fn load(module: &Module, kind: GgmlKernelType) -> Result<Self> {
                match kind.ggml_type() {
                    $(GgmlType::$kind => Ok(Self::$variant(module.kernel()?)),)*
                    _ => Err(Error::InvalidDecoderKernel("no GGML kernel for this type")),
                }
            }

            fn launch(&self, stream: &Stream, config: LaunchConfig, arguments: Arguments<'_>) -> Result<()> {
                match self {
                    $(Self::$variant(kernel) => kernel.launch(stream, config, arguments)?,)*
                }
                Ok(())
            }
        }
    };
}

dequantize_kernels! {
    Q40 = "libmir_cuda_ggml_dequantize_bf16_q4_0", Q4_0;
    Q41 = "libmir_cuda_ggml_dequantize_bf16_q4_1", Q4_1;
    Q50 = "libmir_cuda_ggml_dequantize_bf16_q5_0", Q5_0;
    Q51 = "libmir_cuda_ggml_dequantize_bf16_q5_1", Q5_1;
    Q80 = "libmir_cuda_ggml_dequantize_bf16_q8_0", Q8_0;
    Q2K = "libmir_cuda_ggml_dequantize_bf16_q2_k", Q2_K;
    Q3K = "libmir_cuda_ggml_dequantize_bf16_q3_k", Q3_K;
    Q4K = "libmir_cuda_ggml_dequantize_bf16_q4_k", Q4_K;
    Q5K = "libmir_cuda_ggml_dequantize_bf16_q5_k", Q5_K;
    Q6K = "libmir_cuda_ggml_dequantize_bf16_q6_k", Q6_K;
    IQ4NL = "libmir_cuda_ggml_dequantize_bf16_iq4_nl", IQ4_NL;
    IQ4XS = "libmir_cuda_ggml_dequantize_bf16_iq4_xs", IQ4_XS;
    IQ2XXS = "libmir_cuda_ggml_dequantize_bf16_iq2_xxs", IQ2_XXS;
    IQ2XS = "libmir_cuda_ggml_dequantize_bf16_iq2_xs", IQ2_XS;
    IQ2S = "libmir_cuda_ggml_dequantize_bf16_iq2_s", IQ2_S;
    IQ3XXS = "libmir_cuda_ggml_dequantize_bf16_iq3_xxs", IQ3_XXS;
    IQ3S = "libmir_cuda_ggml_dequantize_bf16_iq3_s", IQ3_S;
    IQ1S = "libmir_cuda_ggml_dequantize_bf16_iq1_s", IQ1_S;
    IQ1M = "libmir_cuda_ggml_dequantize_bf16_iq1_m", IQ1_M;
    MXFP4 = "libmir_cuda_ggml_dequantize_bf16_mxfp4", MXFP4;
    NVFP4 = "libmir_cuda_ggml_dequantize_bf16_nvfp4", NVFP4;
    Q10 = "libmir_cuda_ggml_dequantize_bf16_q1_0", Q1_0;
    Q20 = "libmir_cuda_ggml_dequantize_bf16_q2_0", Q2_0;
    TQ10 = "libmir_cuda_ggml_dequantize_bf16_tq1_0", TQ1_0;
    TQ20 = "libmir_cuda_ggml_dequantize_bf16_tq2_0", TQ2_0;
}

/// A GGML matrix dequantized to row-major BF16 `[rows, columns]`, for the
/// dense GEMM of prefill batches.
#[derive(Clone, Debug)]
pub struct GgmlDequantize {
    kernel: Kernel,
    kind: GgmlKernelType,
    rows: usize,
    columns: usize,
}

impl GgmlDequantize {
    pub fn compile(
        compiler: &Compiler,
        kind: GgmlKernelType,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        Self::compile_kinds(compiler, &[kind], rows, columns)?
            .pop()
            .ok_or(Error::InvalidDecoderKernel("invalid GGML dequantization geometry"))
    }

    /// One dequantization per kind from a single module lookup: the cache
    /// key hashes every composed source, which costs more than the kernel
    /// lookups when a prefill plan prepares all twenty types.
    pub fn compile_kinds(
        compiler: &Compiler,
        kinds: &[GgmlKernelType],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Self>> {
        if rows == 0 || columns == 0 {
            return Err(Error::InvalidDecoderKernel("invalid GGML dequantization geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_files!(
                "ggml_matrix.cu";
                "../../../kernels/ggml/blocks.cuh",
                "../../../kernels/ggml/legacy.cuh",
                "../../../kernels/ggml/k_quants.cuh",
                "../../../kernels/ggml/iq4.cuh",
                "../../../kernels/ggml/low_bit.cuh",
                "../../../kernels/ggml/grids/iq2.cuh",
                "../../../kernels/ggml/grids/iq13.cuh",
                "../../../kernels/ggml/iq.cuh",
                "../../../kernels/ggml/matrix.cu",
            ),
            &CompileOptions::default(),
        )?;
        kinds
            .iter()
            .map(|&kind| {
                let block = usize::try_from(kind.ggml_type().block_elements())?;
                if !columns.is_multiple_of(block) {
                    return Err(Error::InvalidDecoderKernel(
                        "invalid GGML dequantization geometry",
                    ));
                }
                Ok(Self {
                    kernel: Kernel::load(&module, kind)?,
                    kind,
                    rows,
                    columns,
                })
            })
            .collect()
    }

    #[must_use]
    pub const fn kind(&self) -> GgmlKernelType {
        self.kind
    }

    pub fn elements(&self) -> Result<usize> {
        product(self.rows, self.columns)
    }

    pub fn execute(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("GGML dequantized output", self.elements()?, output.len())?;
        self.execute_placed(stream, weight, output, GgmlRowPlacement { stride: 1, offset: 0 })
    }

    /// Writes source row `r` to output row `r * stride + offset`, which
    /// interleaves the rows of matrices sharing one output.
    pub fn execute_placed(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        placement: GgmlRowPlacement,
    ) -> Result<()> {
        require("GGML weight", self.kind.bytes(self.rows, self.columns)?, weight.len())?;
        if placement.stride == 0 || placement.offset >= placement.stride {
            return Err(Error::InvalidDecoderKernel("invalid GGML row placement"));
        }
        let rows = product(self.rows, placement.stride)?;
        require("GGML placed output", product(rows, self.columns)?, output.len())?;
        let config = LaunchConfig {
            grid: (narrow(product(self.rows, self.columns / 32)?.div_ceil(256))?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        let arguments = (
            weight,
            output,
            narrow(self.rows)?,
            narrow(self.columns)?,
            narrow(placement.stride)?,
            narrow(placement.offset)?,
        );
        self.kernel.launch(stream, config, arguments)
    }
}
