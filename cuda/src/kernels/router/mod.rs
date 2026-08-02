mod pattern;
mod unit;

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};
pub use pattern::{RoutePattern, RoutePatternGenerator, RoutePatternSpec};
pub use unit::{RouterUnitSpec, RouterUnitTopK};

use super::geometry::require;
use crate::{Error, Result};

cuda_export!(
    RouterNormalizeKernel = "libmir_cuda_router_normalize_bf16"(
        input: &DeviceBuffer<bf16>, norm_scale: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, hidden: u32, tokens: u32,
        epsilon: f32, norm_multiplier: f32,
    )
);

cuda_export!(
    RouterSelectKernel = "libmir_cuda_router_topk_fp32"(
        scores: &DeviceBuffer<f32>, expert_scale: &DeviceBuffer<bf16>,
        selected: &mut DeviceBuffer<u32>, weights: &mut DeviceBuffer<bf16>,
        experts: u32, top_k: u32, tokens: u32,
    )
);

#[derive(Clone, Copy, Debug)]
pub struct RouterSpec {
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    pub epsilon: f32,
    pub norm_multiplier: f32,
}

#[derive(Clone, Debug)]
pub struct RouterTopK {
    normalize: TypedKernel<RouterNormalizeKernel>,
    select: TypedKernel<RouterSelectKernel>,
    spec: RouterSpec,
}

impl RouterTopK {
    pub fn compile(compiler: &Compiler, spec: RouterSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/router_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            normalize: module.kernel()?,
            select: module.kernel()?,
            spec,
        })
    }

    pub fn normalize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        norm_scale: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
    ) -> Result<()> {
        require("router input", elements(tokens, self.spec.hidden)?, input.len())?;
        require("router norm scale", self.spec.hidden, norm_scale.len())?;
        require("router normalized", elements(tokens, self.spec.hidden)?, output.len())?;
        let config = LaunchConfig {
            grid: (u32::try_from(tokens)?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.normalize.launch(
            stream,
            config,
            (
                input,
                norm_scale,
                output,
                u32::try_from(self.spec.hidden)?,
                u32::try_from(tokens)?,
                self.spec.epsilon,
                self.spec.norm_multiplier,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<f32>,
        expert_scale: &DeviceBuffer<bf16>,
        selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>,
        tokens: usize,
    ) -> Result<()> {
        require("router scores", elements(tokens, self.spec.experts)?, scores.len())?;
        require("router expert scale", self.spec.experts, expert_scale.len())?;
        let selections = elements(tokens, self.spec.top_k)?;
        require("router selected", selections, selected.len())?;
        require("router weights", selections, weights.len())?;
        let config = LaunchConfig {
            grid: (u32::try_from(tokens)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok(self.select.launch(
            stream,
            config,
            (
                scores,
                expert_scale,
                selected,
                weights,
                u32::try_from(self.spec.experts)?,
                u32::try_from(self.spec.top_k)?,
                u32::try_from(tokens)?,
            ),
        )?)
    }
}

fn elements(rows: usize, columns: usize) -> Result<usize> {
    if rows == 0 {
        Err(Error::InvalidRouter("router batch is empty"))
    } else {
        rows.checked_mul(columns)
            .ok_or(Error::InvalidRouter("router buffer size overflow"))
    }
}

fn validate(spec: RouterSpec) -> Result<()> {
    if spec.hidden == 0
        || spec.experts == 0
        || spec.experts > 256
        || spec.top_k == 0
        || spec.top_k > spec.experts
        || !spec.epsilon.is_finite()
        || spec.epsilon < 0.0
        || !spec.norm_multiplier.is_finite()
    {
        Err(Error::InvalidRouter("invalid router geometry or numerical policy"))
    } else {
        Ok(())
    }
}
