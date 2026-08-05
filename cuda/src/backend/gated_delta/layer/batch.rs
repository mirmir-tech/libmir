use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineGatedDeltaExecution, CudaGatedDeltaState};
use crate::{Error, GatedDeltaInputs, Result, backend::gated_delta::CudaGatedDeltaBatchState};

impl CudaAffineGatedDeltaExecution {
    pub(crate) fn prepare_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &[&mut CudaGatedDeltaState],
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let Some(first) = states.first() else {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed batch is empty"));
        };
        if !self.tokens.is_multiple_of(states.len()) {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed row mismatch"));
        }
        let row_tokens = self.tokens / states.len();
        self.validate(input, first, output)?;
        if !self
            .batch_state
            .as_ref()
            .is_some_and(|batch| batch.supports(states.len(), row_tokens))
        {
            self.batch_state = Some(CudaGatedDeltaBatchState::new(
                &self.backend,
                self.config.state()?,
                states.len(),
                row_tokens,
            )?);
        }
        self.batch()?.pack(states)
    }

    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let packed = self.project_qkv_gate(input)?;
        self.project_alpha_beta(input)?;
        let Self {
            backend,
            output: projection,
            transforms,
            weights,
            scratch,
            batch_state,
            config,
            ..
        } = self;
        let batch = batch_state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not prepared"))?;
        let stream = &backend.inner.stream;
        if packed {
            batch.convolve_strided(
                scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16_tensor(&weights.convolution)?,
                &mut scratch.convolved,
                config.mixed_width()? + config.value_width()?,
                0,
            )?;
        } else {
            batch.convolve(
                &scratch.mixed,
                bf16_tensor(&weights.convolution)?,
                &mut scratch.convolved,
            )?;
        }
        transforms.split_normalize(
            stream,
            &scratch.convolved,
            &mut scratch.normalized_query,
            &mut scratch.normalized_key,
            &mut scratch.value,
        )?;
        batch.recur(
            GatedDeltaInputs {
                query: &scratch.normalized_query,
                key: &scratch.normalized_key,
                value: &scratch.value,
                alpha: &scratch.alpha,
                beta: &scratch.beta,
                a_log: bf16_tensor(&weights.a_log)?,
                dt_bias: bf16_tensor(&weights.dt_bias)?,
            },
            &mut scratch.recurrent,
        )?;
        if packed {
            transforms.norm_gate_strided(
                stream,
                &scratch.recurrent,
                scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16_tensor(&weights.norm)?,
                &mut scratch.gated,
                config.mixed_width()? + config.value_width()?,
                config.mixed_width()?,
            )?;
        } else {
            transforms.norm_gate(
                stream,
                &scratch.recurrent,
                &scratch.gate,
                bf16_tensor(&weights.norm)?,
                &mut scratch.gated,
            )?;
        }
        projection.execute(&scratch.gated, output)
    }

    pub(crate) fn commit_packed(&mut self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        self.batch_state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not prepared"))?
            .commit(states)
    }

    fn batch(&mut self) -> Result<&mut CudaGatedDeltaBatchState> {
        self.batch_state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not prepared"))
    }
}

fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
