use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(AlphaBetaKernel = "libmir_cuda_gated_delta_alpha_beta_bf16"(
    input: &DeviceBuffer<bf16>, alpha_weight: &DeviceBuffer<bf16>,
    beta_weight: &DeviceBuffer<bf16>, alpha: &mut DeviceBuffer<bf16>,
    beta: &mut DeviceBuffer<bf16>, tokens: u32, columns: u32, heads: u32,
));

cuda_export!(SplitAlphaBetaKernel = "libmir_cuda_gated_delta_split_alpha_beta_bf16"(
    packed: &DeviceBuffer<bf16>, alpha: &mut DeviceBuffer<bf16>,
    beta: &mut DeviceBuffer<bf16>, elements: u32, heads: u32,
));

#[derive(Clone, Debug)]
pub struct GatedDeltaAlphaBeta {
    kernel: TypedKernel<AlphaBetaKernel>,
    tokens: usize,
    columns: usize,
    heads: usize,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaAlphaBetaSplit {
    kernel: TypedKernel<SplitAlphaBetaKernel>,
    tokens: usize,
    heads: usize,
}

impl GatedDeltaAlphaBeta {
    pub fn compile(
        compiler: &Compiler,
        tokens: usize,
        columns: usize,
        heads: usize,
    ) -> Result<Self> {
        if tokens == 0 || columns == 0 || heads == 0 {
            return Err(Error::InvalidDecoderKernel("empty Gated Delta alpha/beta projection"));
        }
        let source = cuda_kernel_file!("../../../kernels/gated_delta_gates_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            kernel: module.kernel()?,
            tokens,
            columns,
            heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        alpha_weight: &DeviceBuffer<bf16>,
        beta_weight: &DeviceBuffer<bf16>,
        alpha: &mut DeviceBuffer<bf16>,
        beta: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("Gated Delta gate input", product(self.tokens, self.columns)?, input.len())?;
        let weights = product(self.heads, self.columns)?;
        require("Gated Delta alpha weight", weights, alpha_weight.len())?;
        require("Gated Delta beta weight", weights, beta_weight.len())?;
        let output = product(self.tokens, self.heads)?;
        require("Gated Delta alpha", output, alpha.len())?;
        require("Gated Delta beta", output, beta.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.heads.div_ceil(8))?, narrow(self.tokens)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                alpha_weight,
                beta_weight,
                alpha,
                beta,
                narrow(self.tokens)?,
                narrow(self.columns)?,
                narrow(self.heads)?,
            ),
        )?)
    }
}

impl GatedDeltaAlphaBetaSplit {
    pub fn compile(compiler: &Compiler, tokens: usize, heads: usize) -> Result<Self> {
        if tokens == 0 || heads == 0 {
            return Err(Error::InvalidDecoderKernel("empty packed Gated Delta alpha/beta"));
        }
        let source = cuda_kernel_file!("../../../kernels/gated_delta_gates_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, tokens, heads })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        packed: &DeviceBuffer<bf16>,
        alpha: &mut DeviceBuffer<bf16>,
        beta: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.tokens, self.heads)?;
        require("packed Gated Delta alpha/beta", product(elements, 2)?, packed.len())?;
        require("Gated Delta alpha", elements, alpha.len())?;
        require("Gated Delta beta", elements, beta.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (packed, alpha, beta, narrow(elements)?, narrow(self.heads)?),
        )?)
    }
}
