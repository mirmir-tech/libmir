use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{
    super::geometry::{narrow, product, require},
    RouterUnitScoring, RouterUnitWeighting,
};
use crate::{Error, Result};

cuda_export!(
    RouterUnitKernel = "libmir_cuda_router_topk_unit_bf16"(
        scores: &DeviceBuffer<bf16>, selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>, experts: u32, top_k: u32,
        tokens: u32, scale: f32,
    )
);

cuda_export!(
    RouterSigmoidKernel = "libmir_cuda_router_topk_sigmoid_bf16"(
        scores: &DeviceBuffer<bf16>, bias: &DeviceBuffer<f32>,
        selected: &mut DeviceBuffer<u32>, weights: &mut DeviceBuffer<bf16>,
        experts: u32, top_k: u32, tokens: u32, scale: f32,
    )
);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RouterUnitSpec {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Debug)]
pub struct RouterUnitTopK {
    kernel: UnitKernel,
    spec: RouterUnitSpec,
    scale: f32,
}

#[derive(Clone, Debug)]
enum UnitKernel {
    Logits(TypedKernel<RouterUnitKernel>),
    /// Takes the layer's `[experts]` FP32 correction bias at execution.
    Sigmoid(TypedKernel<RouterSigmoidKernel>),
}

impl RouterUnitTopK {
    pub fn compile(compiler: &Compiler, spec: RouterUnitSpec) -> Result<Self> {
        Self::compile_scored(compiler, spec, RouterUnitScoring::RENORMALIZED)
    }

    /// A sigmoid router takes its layer's `[experts]` correction bias at
    /// [`Self::execute`], so layers can share one compiled router.
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
        match scoring.weighting {
            RouterUnitWeighting::Probability => {
                options.extra_options.push("-DLIBMIR_ROUTER_PROBABILITY=1".into());
            },
            RouterUnitWeighting::SparseMixer(jitter) => {
                if scoring.group_limit.is_some() {
                    return Err(Error::InvalidRouter("a sparsemixer router has no expert groups"));
                }
                options
                    .extra_options
                    .push(format!("-DLIBMIR_ROUTER_SPARSEMIXER_JITTER={:e}f", jitter.value()));
            },
            RouterUnitWeighting::SigmoidInputScaled => {
                if scoring.group_limit.is_some() {
                    return Err(Error::InvalidRouter(
                        "an input-scaled router has no expert groups",
                    ));
                }
                options.extra_options.push("-DLIBMIR_ROUTER_INPUT_SCALED=1".into());
            },
            RouterUnitWeighting::Renormalized | RouterUnitWeighting::SigmoidBiased => {},
        }
        if let Some(limit) = scoring.group_limit {
            // A lane scores each group; the sigmoid router sums two experts.
            let minimum = if scoring.weighting == RouterUnitWeighting::SigmoidBiased {
                2
            } else {
                1
            };
            let size = spec.experts.checked_div(limit.groups).unwrap_or(0);
            if limit.groups < 2
                || limit.groups > 32
                || !spec.experts.is_multiple_of(limit.groups)
                || size < minimum
                || limit.selected == 0
                || limit.selected > limit.groups
                || spec.top_k > limit.selected * size
            {
                return Err(Error::InvalidRouter("invalid unit router expert groups"));
            }
            options.extra_options.push(format!("-DLIBMIR_ROUTER_GROUPS={}", limit.groups));
            options
                .extra_options
                .push(format!("-DLIBMIR_ROUTER_GROUP_LIMIT={}", limit.selected));
        }
        let module = compiler.compile(source, &options)?;
        let kernel = match scoring.weighting {
            RouterUnitWeighting::SigmoidBiased => UnitKernel::Sigmoid(module.kernel()?),
            RouterUnitWeighting::Renormalized
            | RouterUnitWeighting::Probability
            | RouterUnitWeighting::SigmoidInputScaled
            | RouterUnitWeighting::SparseMixer(_) => UnitKernel::Logits(module.kernel()?),
        };
        Ok(Self { kernel, spec, scale: scoring.scale })
    }

    /// `correction_bias` is the layer's `[experts]` bias exactly when the
    /// router routes by sigmoid.
    pub fn execute(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<bf16>,
        correction_bias: Option<&DeviceBuffer<f32>>,
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
        let config = LaunchConfig {
            grid: (narrow(self.spec.tokens)?, 1, 1),
            block: (32, 1, 1),
            shared_memory_bytes: 0,
        };
        let (experts, top_k, tokens) =
            (narrow(self.spec.experts)?, narrow(self.spec.top_k)?, narrow(self.spec.tokens)?);
        match (&self.kernel, correction_bias) {
            (UnitKernel::Logits(kernel), None) => kernel.launch(
                stream,
                config,
                (scores, selected, weights, experts, top_k, tokens, self.scale),
            )?,
            (UnitKernel::Sigmoid(kernel), Some(bias)) if bias.len() >= self.spec.experts => kernel
                .launch(
                    stream,
                    config,
                    (scores, bias, selected, weights, experts, top_k, tokens, self.scale),
                )?,
            _ => {
                return Err(Error::InvalidRouter(
                    "a unit router takes a correction bias exactly when it routes by sigmoid",
                ));
            },
        }
        Ok(())
    }
}
