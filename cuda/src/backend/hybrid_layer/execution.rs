use mircuda::{DeviceBuffer, bf16};

use super::{AffineGatedDeltaMoeLayerConfig, scratch::HybridLayerScratch};
use crate::{
    CudaAffineGatedDeltaExecution, CudaAffineGatedDeltaLayer, CudaAffineSharedExpertMoe,
    CudaAffineSharedExpertMoeExecution, CudaBackend, CudaGatedDeltaState, CudaTensor, Error,
    ExecutionPhase, Result,
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub struct CudaAffineGatedDeltaMoeExecution {
    backend: CudaBackend,
    attention: CudaAffineGatedDeltaExecution,
    moe: CudaAffineSharedExpertMoeExecution,
    input_norm: ShiftedRmsNorm,
    post_attention_norm: ShiftedRmsNorm,
    residual: ElementwiseBf16,
    input_norm_weight: CudaTensor,
    post_attention_norm_weight: CudaTensor,
    scratch: HybridLayerScratch,
}

impl CudaAffineGatedDeltaMoeExecution {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaMoeLayerConfig,
        attention: &CudaAffineGatedDeltaLayer,
        moe: &CudaAffineSharedExpertMoe,
        input_norm_weight: &CudaTensor,
        post_attention_norm_weight: &CudaTensor,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let norm = || {
            ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                config.attention.hidden_size,
                config.rms_norm_epsilon,
                config.norm_weight_shift,
            )
        };
        Ok(Self {
            backend: backend.clone(),
            attention: attention.prepare(tokens)?,
            moe: moe.prepare_phase(tokens, phase)?,
            input_norm: norm()?,
            post_attention_norm: norm()?,
            residual: ElementwiseBf16::compile(
                &backend.inner.compiler,
                tokens * config.attention.hidden_size,
            )?,
            input_norm_weight: input_norm_weight.clone(),
            post_attention_norm_weight: post_attention_norm_weight.clone(),
            scratch: HybridLayerScratch::new(backend, tokens, config.attention.hidden_size)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut CudaGatedDeltaState,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16(&self.input_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.attention
            .execute(&self.scratch.normalized, state, &mut self.scratch.attention)?;
        self.residual
            .add(stream, input, &self.scratch.attention, &mut self.scratch.residual)?;
        self.post_attention_norm.execute(
            stream,
            &self.scratch.residual,
            bf16(&self.post_attention_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.moe.execute(&self.scratch.normalized, &mut self.scratch.moe)?;
        self.residual.add(stream, &self.scratch.residual, &self.scratch.moe, output)
    }

    pub(crate) fn execute_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &mut [&mut CudaGatedDeltaState],
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.prepare_packed(input, states, output)?;
        self.execute_prepared_packed(input, output)?;
        self.commit_packed(states)
    }

    pub(crate) fn prepare_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &[&mut CudaGatedDeltaState],
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        self.attention
            .prepare_packed(&self.scratch.normalized, states, &self.scratch.attention)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16(&self.input_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.attention
            .execute_prepared_packed(&self.scratch.normalized, &mut self.scratch.attention)?;
        self.residual
            .add(stream, input, &self.scratch.attention, &mut self.scratch.residual)?;
        self.post_attention_norm.execute(
            stream,
            &self.scratch.residual,
            bf16(&self.post_attention_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.moe.execute(&self.scratch.normalized, &mut self.scratch.moe)?;
        self.residual.add(stream, &self.scratch.residual, &self.scratch.moe, output)
    }

    pub(crate) fn commit_packed(&mut self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        self.attention.commit_packed(states)
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, output: &DeviceBuffer<bf16>) -> Result<()> {
        let expected = self.scratch.residual.len();
        if input.len() != expected || output.len() != expected {
            return Err(Error::InvalidDecoderKernel("affine hybrid layer buffer mismatch"));
        }
        Ok(())
    }
}

fn bf16(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
