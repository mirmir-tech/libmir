use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_files,
};

use super::{
    super::geometry::{narrow, product, require},
    GgmlKernelType, GgmlRowPlacement,
};
use crate::{Error, Result};

macro_rules! ggml_dequantize_exports {
    ($($name:ident = $symbol:literal),* $(,)?) => {$(
        cuda_export!($name = $symbol(
            weight: &DeviceBuffer<u8>, output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
            stride: u32, offset: u32,
        ));
    )*};
}

ggml_dequantize_exports!(
    DequantizeQ40 = "libmir_cuda_ggml_dequantize_bf16_q4_0",
    DequantizeQ41 = "libmir_cuda_ggml_dequantize_bf16_q4_1",
    DequantizeQ50 = "libmir_cuda_ggml_dequantize_bf16_q5_0",
    DequantizeQ51 = "libmir_cuda_ggml_dequantize_bf16_q5_1",
    DequantizeQ80 = "libmir_cuda_ggml_dequantize_bf16_q8_0",
    DequantizeQ3K = "libmir_cuda_ggml_dequantize_bf16_q3_k",
    DequantizeQ2K = "libmir_cuda_ggml_dequantize_bf16_q2_k",
    DequantizeQ4K = "libmir_cuda_ggml_dequantize_bf16_q4_k",
    DequantizeQ5K = "libmir_cuda_ggml_dequantize_bf16_q5_k",
    DequantizeQ6K = "libmir_cuda_ggml_dequantize_bf16_q6_k",
    DequantizeIQ4NL = "libmir_cuda_ggml_dequantize_bf16_iq4_nl",
    DequantizeIQ4XS = "libmir_cuda_ggml_dequantize_bf16_iq4_xs",
    DequantizeIQ2XXS = "libmir_cuda_ggml_dequantize_bf16_iq2_xxs",
    DequantizeIQ2XS = "libmir_cuda_ggml_dequantize_bf16_iq2_xs",
    DequantizeIQ2S = "libmir_cuda_ggml_dequantize_bf16_iq2_s",
    DequantizeIQ3XXS = "libmir_cuda_ggml_dequantize_bf16_iq3_xxs",
    DequantizeIQ3S = "libmir_cuda_ggml_dequantize_bf16_iq3_s",
    DequantizeIQ1S = "libmir_cuda_ggml_dequantize_bf16_iq1_s",
    DequantizeIQ1M = "libmir_cuda_ggml_dequantize_bf16_iq1_m",
    DequantizeMXFP4 = "libmir_cuda_ggml_dequantize_bf16_mxfp4",
);

#[derive(Clone, Debug)]
enum Kernel {
    Q4_0(TypedKernel<DequantizeQ40>),
    Q4_1(TypedKernel<DequantizeQ41>),
    Q5_0(TypedKernel<DequantizeQ50>),
    Q5_1(TypedKernel<DequantizeQ51>),
    Q8_0(TypedKernel<DequantizeQ80>),
    Q3K(TypedKernel<DequantizeQ3K>),
    Q2K(TypedKernel<DequantizeQ2K>),
    Q4K(TypedKernel<DequantizeQ4K>),
    Q5K(TypedKernel<DequantizeQ5K>),
    Q6K(TypedKernel<DequantizeQ6K>),
    IQ4NL(TypedKernel<DequantizeIQ4NL>),
    IQ4XS(TypedKernel<DequantizeIQ4XS>),
    IQ2XXS(TypedKernel<DequantizeIQ2XXS>),
    IQ2XS(TypedKernel<DequantizeIQ2XS>),
    IQ2S(TypedKernel<DequantizeIQ2S>),
    IQ3XXS(TypedKernel<DequantizeIQ3XXS>),
    IQ3S(TypedKernel<DequantizeIQ3S>),
    IQ1S(TypedKernel<DequantizeIQ1S>),
    IQ1M(TypedKernel<DequantizeIQ1M>),
    MXFP4(TypedKernel<DequantizeMXFP4>),
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
                    kernel: kernel(&module, kind)?,
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
        match &self.kernel {
            Kernel::Q4_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q4_1(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5_1(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q8_0(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q3K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q2K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q4K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q5K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::Q6K(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ4NL(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ4XS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2XXS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2XS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ2S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ3XXS(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ3S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ1S(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::IQ1M(kernel) => kernel.launch(stream, config, arguments)?,
            Kernel::MXFP4(kernel) => kernel.launch(stream, config, arguments)?,
        }
        Ok(())
    }
}

fn kernel(module: &Module, kind: GgmlKernelType) -> Result<Kernel> {
    Ok(match kind {
        GgmlKernelType::Q4_0 => Kernel::Q4_0(module.kernel()?),
        GgmlKernelType::Q4_1 => Kernel::Q4_1(module.kernel()?),
        GgmlKernelType::Q5_0 => Kernel::Q5_0(module.kernel()?),
        GgmlKernelType::Q5_1 => Kernel::Q5_1(module.kernel()?),
        GgmlKernelType::Q8_0 => Kernel::Q8_0(module.kernel()?),
        GgmlKernelType::Q3K => Kernel::Q3K(module.kernel()?),
        GgmlKernelType::Q2K => Kernel::Q2K(module.kernel()?),
        GgmlKernelType::Q4K => Kernel::Q4K(module.kernel()?),
        GgmlKernelType::Q5K => Kernel::Q5K(module.kernel()?),
        GgmlKernelType::Q6K => Kernel::Q6K(module.kernel()?),
        GgmlKernelType::IQ4NL => Kernel::IQ4NL(module.kernel()?),
        GgmlKernelType::IQ4XS => Kernel::IQ4XS(module.kernel()?),
        GgmlKernelType::IQ2XXS => Kernel::IQ2XXS(module.kernel()?),
        GgmlKernelType::IQ2XS => Kernel::IQ2XS(module.kernel()?),
        GgmlKernelType::IQ2S => Kernel::IQ2S(module.kernel()?),
        GgmlKernelType::IQ3XXS => Kernel::IQ3XXS(module.kernel()?),
        GgmlKernelType::IQ3S => Kernel::IQ3S(module.kernel()?),
        GgmlKernelType::IQ1S => Kernel::IQ1S(module.kernel()?),
        GgmlKernelType::IQ1M => Kernel::IQ1M(module.kernel()?),
        GgmlKernelType::MXFP4 => Kernel::MXFP4(module.kernel()?),
    })
}
