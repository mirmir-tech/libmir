use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaAffineGatedDeltaExecution, CudaGatedDeltaState,
    execution::bf16,
    segments::{RowSegment, row_segments},
};
use crate::{Error, GatedDeltaInputs, Result};

impl CudaAffineGatedDeltaExecution {
    pub(crate) fn execute_ragged(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &mut [&mut CudaGatedDeltaState],
        counts: &[usize],
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let total = counts.iter().try_fold(0_usize, |sum, count| sum.checked_add(*count));
        if states.is_empty()
            || states.len() != counts.len()
            || counts.contains(&0)
            || total.is_none_or(|total| total > self.tokens)
        {
            return Err(Error::InvalidDecoderKernel("Gated Delta ragged row mismatch"));
        }
        for state in states.iter() {
            self.validate(input, state, output)?;
        }
        let packed = self.project_qkv_gate(input)?;
        self.project_alpha_beta(input)?;
        let mixed = self.config.mixed_width()?;
        let key = self.config.key_width()?;
        let value = self.config.value_width()?;
        let heads = self.config.value_heads;
        let stride = if packed {
            mixed + value
        } else {
            mixed
        };
        let source = if packed {
            self.scratch
                .packed_qkv_gate
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?
        } else {
            &self.scratch.mixed
        };
        let segments = row_segments(states, counts)?;
        for RowSegment { row, offset, count, checkpoint } in segments.iter().copied() {
            let state = &mut *states[row];
            if self.config.key_dim == 128 {
                state.convolve_silu_split_normalize_strided(
                    count,
                    &source.slice(offset * stride..(offset + count) * stride)?,
                    bf16(&self.weights.convolution)?,
                    &mut self
                        .scratch
                        .normalized_query
                        .slice(offset * key..(offset + count) * key)?,
                    &mut self.scratch.normalized_key.slice(offset * key..(offset + count) * key)?,
                    &mut self.scratch.value.slice(offset * value..(offset + count) * value)?,
                    stride,
                    0,
                )?;
            } else {
                state.convolve_silu_strided(
                    count,
                    &source.slice(offset * stride..(offset + count) * stride)?,
                    bf16(&self.weights.convolution)?,
                    &mut self.scratch.convolved.slice(offset * mixed..(offset + count) * mixed)?,
                    stride,
                    0,
                )?;
            }
            if checkpoint {
                state.stage_convolution()?;
            }
        }
        if self.config.key_dim != 128 {
            self.transforms.split_normalize(
                &self.backend.inner.stream,
                &self.scratch.convolved,
                &mut self.scratch.normalized_query,
                &mut self.scratch.normalized_key,
                &mut self.scratch.value,
            )?;
        }
        for RowSegment { row, offset, count, checkpoint } in segments {
            let state = &mut *states[row];
            state.execute(
                count,
                GatedDeltaInputs {
                    query: &self
                        .scratch
                        .normalized_query
                        .slice(offset * key..(offset + count) * key)?,
                    key: &self
                        .scratch
                        .normalized_key
                        .slice(offset * key..(offset + count) * key)?,
                    value: &self.scratch.value.slice(offset * value..(offset + count) * value)?,
                    alpha: &self.scratch.alpha.slice(offset * heads..(offset + count) * heads)?,
                    beta: &self.scratch.beta.slice(offset * heads..(offset + count) * heads)?,
                    a_log: bf16(&self.weights.a_log)?,
                    dt_bias: bf16(&self.weights.dt_bias)?,
                },
                &mut self.scratch.recurrent.slice(offset * value..(offset + count) * value)?,
            )?;
            if checkpoint {
                state.stage_state()?;
            }
        }
        self.finish_projected(packed, output)
    }
}
