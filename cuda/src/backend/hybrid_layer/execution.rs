use mircuda::{DeviceBuffer, bf16};

use crate::{
    CudaAffineGatedDeltaExecution, CudaAffineGatedDeltaLayer, CudaBackend, CudaGatedDeltaState,
    CudaTensor, Error, ExecutionPhase, Result,
    backend::{
        feed_forward::{FeedForward, FeedForwardExecution, LayerNormConfig},
        layer_scratch::SharedLayerScratch,
    },
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub struct CudaAffineGatedDeltaMoeExecution {
    backend: CudaBackend,
    attention: CudaAffineGatedDeltaExecution,
    moe: FeedForwardExecution,
    input_norm: ShiftedRmsNorm,
    post_attention_norm: ShiftedRmsNorm,
    residual: ElementwiseBf16,
    input_norm_weight: CudaTensor,
    post_attention_norm_weight: CudaTensor,
    scratch: SharedLayerScratch,
}

impl CudaAffineGatedDeltaMoeExecution {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        config: LayerNormConfig,
        attention: &CudaAffineGatedDeltaLayer,
        moe: &FeedForward,
        input_norm_weight: &CudaTensor,
        post_attention_norm_weight: &CudaTensor,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let norm = || {
            ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
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
                tokens * config.hidden_size,
            )?,
            input_norm_weight: input_norm_weight.clone(),
            post_attention_norm_weight: post_attention_norm_weight.clone(),
            scratch: backend.inner.layer_scratch.acquire(backend, tokens, config.hidden_size)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut CudaGatedDeltaState,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16(&self.input_norm_weight)?,
            &mut scratch.normalized,
        )?;
        self.attention.execute(&scratch.normalized, state, &mut scratch.attention)?;
        self.moe.execute_residual_norm(
            &self.post_attention_norm,
            input,
            &scratch.attention,
            bf16(&self.post_attention_norm_weight)?,
            &mut scratch.residual,
            &mut scratch.normalized,
            &mut scratch.moe,
            output,
            &self.residual,
        )?;
        drop(guard);
        Ok(())
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
        let scratch = self.scratch.lock()?;
        self.attention.prepare_packed(&scratch.normalized, states, &scratch.attention)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16(&self.input_norm_weight)?,
            &mut scratch.normalized,
        )?;
        self.attention
            .execute_prepared_packed(&scratch.normalized, &mut scratch.attention)?;
        self.moe.execute_residual_norm(
            &self.post_attention_norm,
            input,
            &scratch.attention,
            bf16(&self.post_attention_norm_weight)?,
            &mut scratch.residual,
            &mut scratch.normalized,
            &mut scratch.moe,
            output,
            &self.residual,
        )?;
        drop(guard);
        Ok(())
    }

    pub(crate) fn execute_ragged(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &mut [&mut CudaGatedDeltaState],
        counts: &[usize],
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16(&self.input_norm_weight)?,
            &mut scratch.normalized,
        )?;
        self.attention.execute_ragged(
            &scratch.normalized,
            states,
            counts,
            &mut scratch.attention,
        )?;
        self.moe.execute_residual_norm(
            &self.post_attention_norm,
            input,
            &scratch.attention,
            bf16(&self.post_attention_norm_weight)?,
            &mut scratch.residual,
            &mut scratch.normalized,
            &mut scratch.moe,
            output,
            &self.residual,
        )?;
        drop(guard);
        Ok(())
    }

    pub(crate) fn commit_packed(&mut self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        self.attention.commit_packed(states)
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, output: &DeviceBuffer<bf16>) -> Result<()> {
        let expected = self.scratch.elements();
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
