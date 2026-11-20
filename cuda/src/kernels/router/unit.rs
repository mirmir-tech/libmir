use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RouterUnitKernel = "libmir_cuda_router_topk_unit_bf16"(
        scores: &DeviceBuffer<bf16>, selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>, experts: u32, top_k: u32,
        tokens: u32, scale: f32,
    )
);

/// How the selected experts are weighted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RouterUnitWeighting {
    /// Softmax over the selected logits.
    Renormalized,
    /// Softmax probability over every expert, not renormalized.
    Probability,
}

/// Weighting of the selected experts and a factor on every weight, which
/// scales the combined routed output.
#[derive(Clone, Copy, Debug)]
pub struct RouterUnitScoring {
    pub weighting: RouterUnitWeighting,
    pub scale: f32,
}

impl RouterUnitScoring {
    pub const RENORMALIZED: Self = Self {
        weighting: RouterUnitWeighting::Renormalized,
        scale: 1.0,
    };
}

// The scale is a fixed configuration constant; plans compare it exactly.
impl PartialEq for RouterUnitScoring {
    fn eq(&self, other: &Self) -> bool {
        self.weighting == other.weighting && self.scale.to_bits() == other.scale.to_bits()
    }
}

impl Eq for RouterUnitScoring {}

impl std::hash::Hash for RouterUnitScoring {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.weighting.hash(state);
        self.scale.to_bits().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RouterUnitSpec {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Debug)]
pub struct RouterUnitTopK {
    kernel: TypedKernel<RouterUnitKernel>,
    spec: RouterUnitSpec,
    scale: f32,
}

impl RouterUnitTopK {
    pub fn compile(compiler: &Compiler, spec: RouterUnitSpec) -> Result<Self> {
        Self::compile_scored(compiler, spec, RouterUnitScoring::RENORMALIZED)
    }

    pub fn compile_scored(
        compiler: &Compiler,
        spec: RouterUnitSpec,
        scoring: RouterUnitScoring,
    ) -> Result<Self> {
        if !scoring.scale.is_finite() || scoring.scale <= 0.0 {
            return Err(Error::InvalidRouter("invalid unit router scale"));
        }
        if spec.tokens == 0
            || spec.experts == 0
            || spec.experts > 256
            || spec.top_k == 0
            || spec.top_k > spec.experts
        {
            return Err(Error::InvalidRouter("invalid unit router geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/router_unit_bf16.cu");
        let mut options = CompileOptions::default();
        if scoring.weighting == RouterUnitWeighting::Probability {
            options.extra_options.push("-DLIBMIR_ROUTER_PROBABILITY=1".into());
        }
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            kernel: module.kernel()?,
            spec,
            scale: scoring.scale,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<bf16>,
        selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require(
            "unit router scores",
            product(self.spec.tokens, self.spec.experts)?,
            scores.len(),
        )?;
        let selections = product(self.spec.tokens, self.spec.top_k)?;
        require("unit router selected", selections, selected.len())?;
        require("unit router weights", selections, weights.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.tokens)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                scores,
                selected,
                weights,
                narrow(self.spec.experts)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.tokens)?,
                self.scale,
            ),
        )?)
    }
}
