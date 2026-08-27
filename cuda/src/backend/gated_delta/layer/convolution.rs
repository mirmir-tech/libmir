use super::{CudaAffineGatedDeltaExecution, CudaGatedDeltaState, execution::bf16};
use crate::{Error, Result};

impl CudaAffineGatedDeltaExecution {
    pub(super) fn convolve_projected(
        &mut self,
        state: &mut CudaGatedDeltaState,
        packed: bool,
    ) -> Result<()> {
        let stride = if packed {
            self.config.mixed_width()? + self.config.value_width()?
        } else {
            self.config.mixed_width()?
        };
        let source = if packed {
            self.scratch
                .packed_qkv_gate
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?
        } else {
            &self.scratch.mixed
        };
        if self.config.key_dim == 128 {
            return state.convolve_silu_split_normalize_strided(
                self.tokens,
                source,
                bf16(&self.weights.convolution)?,
                &mut self.scratch.normalized_query,
                &mut self.scratch.normalized_key,
                &mut self.scratch.value,
                stride,
                0,
            );
        }
        state.convolve_silu_strided(
            self.tokens,
            source,
            bf16(&self.weights.convolution)?,
            &mut self.scratch.convolved,
            stride,
            0,
        )?;
        self.transforms.split_normalize(
            &self.backend.inner.stream,
            &self.scratch.convolved,
            &mut self.scratch.normalized_query,
            &mut self.scratch.normalized_key,
            &mut self.scratch.value,
        )
    }
}
