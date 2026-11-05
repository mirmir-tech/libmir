//! Sigmoid router with correction-bias selection and the ungated `relu²`
//! activation of latent routed-expert blocks.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    LogitsKernel = "libmir_cuda_router_logits_bf16_f32"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        logits: &mut DeviceBuffer<f32>, tokens: u32, hidden: u32, experts: u32,
    )
);
cuda_export!(
    RowsLogitsKernel = "libmir_cuda_router_logits_rows_bf16_f32"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        logits: &mut DeviceBuffer<f32>, tokens: u32, hidden: u32, experts: u32,
    )
);
cuda_export!(
    SelectKernel = "libmir_cuda_router_sigmoid_topk_bf16"(
        logits: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>,
        selected: &mut DeviceBuffer<u32>, weights: &mut DeviceBuffer<bf16>,
        experts: u32, top_k: u32, routed_scale: f32,
    )
);
cuda_export!(
    ReluSquaredKernel = "libmir_cuda_relu_squared_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);

/// Largest chunk routed by the per-expert dot-product kernel; larger chunks
/// use tensor-core tiles. Must match `LIBMIR_ROUTER_ROWS`.
const ROW_TOKENS: usize = 8;

/// Geometry and scaling of a sigmoid router with correction-bias selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidRouterSpec {
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    pub routed_scale: f32,
}

/// Compiled router and activation kernels.
#[derive(Clone, Debug)]
pub struct SigmoidRouter {
    logits: TypedKernel<LogitsKernel>,
    rows_logits: TypedKernel<RowsLogitsKernel>,
    select: TypedKernel<SelectKernel>,
    spec: SigmoidRouterSpec,
}

/// Ungated `relu(x)^2` activation.
#[derive(Clone, Debug)]
pub struct ReluSquaredBf16 {
    kernel: TypedKernel<ReluSquaredKernel>,
}

impl ReluSquaredBf16 {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/latent_moe_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    /// Writes `relu(input)^2` for the first `elements` values.
    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        elements: usize,
    ) -> Result<()> {
        require("relu² input", elements, input.len())?;
        require("relu² output", elements, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, output, narrow(elements)?),
        )?)
    }
}

/// Router outputs for `tokens` rows.
pub struct SigmoidRouterOutputs<'a> {
    /// `tokens * experts` FP32 logits.
    pub logits: &'a mut DeviceBuffer<f32>,
    /// `tokens * top_k` selected expert indices.
    pub selected: &'a mut DeviceBuffer<u32>,
    /// `tokens * top_k` renormalized, scaled routing weights.
    pub weights: &'a mut DeviceBuffer<bf16>,
}

impl SigmoidRouter {
    pub fn compile(compiler: &Compiler, spec: SigmoidRouterSpec) -> Result<Self> {
        let valid = spec.hidden > 0
            && spec.hidden.is_multiple_of(256)
            && spec.experts > 0
            && spec.experts.is_multiple_of(64)
            && spec.experts <= 1024
            && spec.top_k > 0
            && spec.top_k <= spec.experts.min(64)
            && spec.routed_scale.is_finite()
            && spec.routed_scale > 0.0;
        if !valid {
            return Err(Error::InvalidDecoderKernel("unsupported sigmoid router geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/latent_moe_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            logits: module.kernel()?,
            rows_logits: module.kernel()?,
            select: module.kernel()?,
            spec,
        })
    }

    #[must_use]
    pub const fn spec(&self) -> SigmoidRouterSpec {
        self.spec
    }

    /// Computes FP32 logits and selects `top_k` experts per token.
    pub fn route(
        &self,
        stream: &Stream,
        tokens: usize,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<f32>,
        outputs: &mut SigmoidRouterOutputs<'_>,
    ) -> Result<()> {
        let spec = self.spec;
        require("router input", product(tokens, spec.hidden)?, input.len())?;
        require("router weight", product(spec.experts, spec.hidden)?, weight.len())?;
        require("router correction bias", spec.experts, bias.len())?;
        require("router logits", product(tokens, spec.experts)?, outputs.logits.len())?;
        require("router selection", product(tokens, spec.top_k)?, outputs.selected.len())?;
        require("router weights", product(tokens, spec.top_k)?, outputs.weights.len())?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("router has no tokens"));
        }
        let arguments = (narrow(tokens)?, narrow(spec.hidden)?, narrow(spec.experts)?);
        if tokens <= ROW_TOKENS {
            self.rows_logits.launch(
                stream,
                LaunchConfig {
                    grid: (narrow(spec.experts / 8)?, 1, 1),
                    block: (256, 1, 1),
                    shared_memory_bytes: 0,
                },
                (input, weight, &mut *outputs.logits, arguments.0, arguments.1, arguments.2),
            )?;
        } else {
            self.logits.launch(
                stream,
                LaunchConfig {
                    grid: (narrow(tokens.div_ceil(16))?, narrow(spec.experts / 64)?, 1),
                    block: (128, 1, 1),
                    shared_memory_bytes: 0,
                },
                (input, weight, &mut *outputs.logits, arguments.0, arguments.1, arguments.2),
            )?;
        }
        Ok(self.select.launch(
            stream,
            LaunchConfig {
                grid: (narrow(tokens)?, 1, 1),
                block: (narrow(spec.experts.div_ceil(32) * 32)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &*outputs.logits,
                bias,
                &mut *outputs.selected,
                &mut *outputs.weights,
                narrow(spec.experts)?,
                narrow(spec.top_k)?,
                spec.routed_scale,
            ),
        )?)
    }
}
