use super::{
    CudaAffineGatedDeltaExecution, CudaGatedDeltaState, execution::bf16, scratch::GatedDeltaScratch,
};
use crate::{Error, Result};

impl CudaAffineGatedDeltaExecution {
    pub(super) fn convolve_projected(
        &self,
        scratch: &mut GatedDeltaScratch,
        state: &mut CudaGatedDeltaState,
        packed: bool,
    ) -> Result<()> {
        let stride = if packed {
            self.config.mixed_width()? + self.config.value_width()?
        } else {
            self.config.mixed_width()?
        };
        let source = if packed {
            scratch
                .packed_qkv_gate
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?
        } else {
            &scratch.mixed
        };
        if self.config.key_dim == 128 {
            return state.convolve_silu_split_normalize_strided(
                self.tokens,
                source,
                bf16(&self.weights.convolution)?,
                &mut scratch.normalized_query,
                &mut scratch.normalized_key,
                &mut scratch.value,
                stride,
                0,
            );
        }
        state.convolve_silu_strided(
            self.tokens,
            source,
            bf16(&self.weights.convolution)?,
            &mut scratch.convolved,
            stride,
            0,
        )?;
        self.transforms.split_normalize(
            &self.backend.inner.stream,
            &scratch.convolved,
            &mut scratch.normalized_query,
            &mut scratch.normalized_key,
            &mut scratch.value,
        )
    }
}
