use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, scratch::AffineSharedMoeScratch,
};
use crate::{
    AffineQuantizedConfig, AffineQuantizedPairTensors, CudaBackend, Error, Result,
    SelectedAffineGatedBf16Linear, SelectedAffineReduceBf16Linear,
    backend::{AffineRouterBf16, linear::AffineProjection},
    kernels::{ElementwiseBf16, SigmoidMultiplyBf16},
};

#[derive(Debug)]
pub struct CudaAffineSharedExpertMoeExecution {
    backend: CudaBackend,
    config: AffineSharedExpertMoeConfig,
    tokens: usize,
    router: AffineRouterBf16,
    routed_gated: SelectedAffineGatedBf16Linear,
    routed_down: SelectedAffineReduceBf16Linear,
    shared_gate: AffineProjection,
    shared_up: AffineProjection,
    shared_down: AffineProjection,
    shared_output_gate: AffineProjection,
    shared_activation: ElementwiseBf16,
    output_add: ElementwiseBf16,
    sigmoid_multiply: SigmoidMultiplyBf16,
    weights: AffineSharedExpertMoeWeights,
    scratch: AffineSharedMoeScratch,
}

impl CudaAffineSharedExpertMoeExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        tokens: usize,
    ) -> Result<Self> {
        let expert = |input, output| {
            AffineQuantizedConfig::new(input, output, config.group_size, config.expert_bits)
        };
        let shared = |input, output, weight| {
            AffineProjection::new(
                backend,
                tokens,
                input,
                output,
                config.group_size,
                config.expert_bits,
                weight,
            )
        };
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            router: backend.prepare_affine_router_bf16(
                tokens,
                AffineQuantizedConfig::new(
                    config.hidden_size,
                    config.expert_count,
                    config.group_size,
                    config.router_bits,
                ),
                config.top_k,
            )?,
            routed_gated: backend.prepare_batched_selected_affine_gated_bf16_linear(
                tokens,
                expert(config.hidden_size, config.routed_intermediate_size),
                config.expert_count,
                config.top_k,
                config.activation,
            )?,
            routed_down: backend.prepare_batched_selected_affine_reduce_bf16_linear(
                tokens,
                expert(config.routed_intermediate_size, config.hidden_size),
                config.expert_count,
                config.top_k,
            )?,
            shared_gate: shared(
                config.hidden_size,
                config.shared_intermediate_size,
                &weights.shared_gate,
            )?,
            shared_up: shared(
                config.hidden_size,
                config.shared_intermediate_size,
                &weights.shared_up,
            )?,
            shared_down: shared(
                config.shared_intermediate_size,
                config.hidden_size,
                &weights.shared_down,
            )?,
            shared_output_gate: AffineProjection::new(
                backend,
                tokens,
                config.hidden_size,
                1,
                config.group_size,
                config.router_bits,
                &weights.shared_output_gate,
            )?,
            shared_activation: ElementwiseBf16::compile(
                &backend.inner.compiler,
                tokens * config.shared_intermediate_size,
            )?,
            output_add: ElementwiseBf16::compile(
                &backend.inner.compiler,
                tokens * config.hidden_size,
            )?,
            sigmoid_multiply: SigmoidMultiplyBf16::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
            )?,
            weights: weights.clone(),
            scratch: AffineSharedMoeScratch::new(backend, config, tokens)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let selection = self.router.execute(input, self.weights.router.tensors())?;
        self.routed_gated.execute(
            input,
            selection.indices,
            AffineQuantizedPairTensors {
                gate: self.weights.routed_gate.tensors(),
                up: self.weights.routed_up.tensors(),
            },
            &mut self.scratch.routed_intermediate,
        )?;
        self.routed_down.execute(
            &self.scratch.routed_intermediate,
            selection.indices,
            selection.weights,
            self.weights.routed_down.tensors(),
            &mut self.scratch.routed_output,
        )?;
        self.shared(input)?;
        self.output_add.add(
            &self.backend.inner.stream,
            &self.scratch.routed_output,
            &self.scratch.gated_shared_output,
            output,
        )
    }

    fn shared(&mut self, input: &DeviceBuffer<bf16>) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.shared_gate.execute(input, &mut self.scratch.shared_gate)?;
        self.shared_up.execute(input, &mut self.scratch.shared_up)?;
        self.shared_activation.gated(
            stream,
            &self.scratch.shared_gate,
            &self.scratch.shared_up,
            &mut self.scratch.shared_intermediate,
            self.config.activation.into(),
        )?;
        self.shared_down
            .execute(&self.scratch.shared_intermediate, &mut self.scratch.shared_output)?;
        self.shared_output_gate.execute(input, &mut self.scratch.shared_output_gate)?;
        self.sigmoid_multiply.execute(
            stream,
            &self.scratch.shared_output,
            &self.scratch.shared_output_gate,
            &mut self.scratch.gated_shared_output,
        )
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, output: &DeviceBuffer<bf16>) -> Result<()> {
        let expected = self.tokens * self.config.hidden_size;
        if input.len() != expected || output.len() != expected {
            return Err(Error::InvalidDecoderKernel("affine shared-expert MoE buffer mismatch"));
        }
        Ok(())
    }
}
