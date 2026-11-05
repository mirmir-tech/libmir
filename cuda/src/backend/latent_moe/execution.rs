use mircuda::{DeviceBuffer, MarlinNvFp4ThreadConfig, bf16};

use super::{LatentMoeConfig, scratch::LatentMoeScratch, weights::LatentMoeWeights};
use crate::{
    CudaBackend, DenseRole, Error, Result,
    backend::{
        linear::{CheckpointProjection, MarlinNvFp4UngatedMoeBf16},
        scratch_pool::SharedScratch,
    },
    kernels::{
        ElementwiseBf16, ReluSquaredBf16, SigmoidRouter, SigmoidRouterOutputs, SigmoidRouterSpec,
    },
};

/// Kernels of one latent routed-expert block for a fixed chunk; its row
/// buffers are shared with every block of the same shape.
#[derive(Debug)]
pub struct CudaLatentMoeExecution {
    backend: CudaBackend,
    config: LatentMoeConfig,
    weights: LatentMoeWeights,
    tokens: usize,
    router: SigmoidRouter,
    latent_input: CheckpointProjection,
    experts: MarlinNvFp4UngatedMoeBf16,
    latent_output: CheckpointProjection,
    shared: Option<SharedExpertExecution>,
    add: ElementwiseBf16,
    scratch: SharedScratch<LatentMoeScratch>,
}

#[derive(Debug)]
struct SharedExpertExecution {
    up: CheckpointProjection,
    activation: ReluSquaredBf16,
    down: CheckpointProjection,
}

impl CudaLatentMoeExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: LatentMoeConfig,
        weights: &LatentMoeWeights,
        tokens: usize,
        tiles: MarlinNvFp4ThreadConfig,
    ) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("latent routed-expert chunk has no tokens"));
        }
        let elements = |columns: usize| {
            tokens
                .checked_mul(columns)
                .ok_or(Error::InvalidDecoderKernel("latent MoE size overflow"))
        };
        let shared = weights
            .shared
            .as_ref()
            .zip(config.shared_intermediate)
            .map(|(shared, width)| {
                Ok::<_, Error>(SharedExpertExecution {
                    up: CheckpointProjection::new(
                        backend,
                        tokens,
                        config.hidden,
                        width,
                        DenseRole::DenseGateUp,
                        &shared.up,
                    )?,
                    activation: ReluSquaredBf16::compile(backend.compiler())?,
                    down: CheckpointProjection::new(
                        backend,
                        tokens,
                        width,
                        config.hidden,
                        DenseRole::DenseDown,
                        &shared.down,
                    )?,
                })
            })
            .transpose()?;
        Ok(Self {
            backend: backend.clone(),
            config,
            weights: weights.clone(),
            tokens,
            router: SigmoidRouter::compile(
                backend.compiler(),
                SigmoidRouterSpec {
                    hidden: config.hidden,
                    experts: config.experts,
                    top_k: config.top_k,
                    routed_scale: config.routed_scale,
                },
            )?,
            latent_input: CheckpointProjection::new(
                backend,
                tokens,
                config.hidden,
                config.latent,
                DenseRole::DenseGateUp,
                &weights.latent_input,
            )?,
            experts: backend.prepare_marlin_nvfp4_ungated_moe_bf16(
                tokens,
                config.top_k,
                tiles,
                [weights.up.clone(), weights.down.clone()],
            )?,
            latent_output: CheckpointProjection::new(
                backend,
                tokens,
                config.latent,
                config.hidden,
                DenseRole::DenseDown,
                &weights.latent_output,
            )?,
            shared,
            add: ElementwiseBf16::compile(backend.compiler(), elements(config.hidden)?)?,
            scratch: backend.latent_moe_scratch(config, tokens)?,
        })
    }

    pub(in crate::backend) fn stream(&self) -> mircuda::Stream {
        self.backend.stream().clone()
    }

    /// `input` holds `tokens` normalized hidden rows; `output` receives the
    /// block output of the same shape.
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = self.backend.stream();
        let router = self
            .weights
            .router
            .as_bf16()
            .ok_or(Error::InvalidExecutionPlan("router weight is not BF16"))?;
        let bias = self
            .weights
            .correction_bias
            .as_f32()
            .ok_or(Error::InvalidExecutionPlan("router correction bias is not FP32"))?;
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        self.router.route(
            stream,
            self.tokens,
            input,
            router,
            bias,
            &mut SigmoidRouterOutputs {
                logits: &mut scratch.logits,
                selected: &mut scratch.selected,
                weights: &mut scratch.routing,
            },
        )?;
        self.latent_input.execute(input, &mut scratch.latent)?;
        self.experts.execute(
            &scratch.latent,
            &scratch.selected,
            &scratch.routing,
            &mut scratch.routed_latent,
        )?;
        let (Some(shared), Some(rows)) = (&mut self.shared, &mut scratch.shared) else {
            return self.latent_output.execute(&scratch.routed_latent, output);
        };
        self.latent_output.execute(&scratch.routed_latent, &mut scratch.routed)?;
        let width = self.config.shared_intermediate.unwrap_or_default();
        shared.up.execute(input, &mut rows.intermediate)?;
        shared.activation.execute(
            stream,
            &rows.intermediate,
            &mut rows.activated,
            self.tokens * width,
        )?;
        shared.down.execute(&rows.activated, &mut rows.output)?;
        self.add.add(stream, &scratch.routed, &rows.output, output)
    }
}
