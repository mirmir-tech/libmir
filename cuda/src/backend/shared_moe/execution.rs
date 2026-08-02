use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, routed::SharedRoutedExecution,
    scratch::AffineSharedMoeScratch,
};
use crate::{
    CudaBackend, DenseRole, Error, Result,
    backend::linear::CheckpointProjection,
    kernels::{ElementwiseBf16, SigmoidMultiplyBf16},
};

#[derive(Debug)]
pub struct CudaAffineSharedExpertMoeExecution {
    backend: CudaBackend,
    config: AffineSharedExpertMoeConfig,
    tokens: usize,
    routed: SharedRoutedExecution,
    shared_gate: CheckpointProjection,
    shared_up: CheckpointProjection,
    shared_down: CheckpointProjection,
    shared_output_gate: CheckpointProjection,
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
        let shared = |input, output, role, weight| {
            CheckpointProjection::new(backend, tokens, input, output, role, weight)
        };
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            routed: SharedRoutedExecution::new(backend, config, weights, tokens)?,
            shared_gate: shared(
                config.hidden_size,
                config.shared_intermediate_size,
                DenseRole::DenseGateUp,
                &weights.shared_gate,
            )?,
            shared_up: shared(
                config.hidden_size,
                config.shared_intermediate_size,
                DenseRole::DenseGateUp,
                &weights.shared_up,
            )?,
            shared_down: shared(
                config.shared_intermediate_size,
                config.hidden_size,
                DenseRole::DenseDown,
                &weights.shared_down,
            )?,
            shared_output_gate: CheckpointProjection::new(
                backend,
                tokens,
                config.hidden_size,
                1,
                DenseRole::Router,
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
        self.routed.execute(&self.backend, &self.weights, input, &mut self.scratch)?;
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
