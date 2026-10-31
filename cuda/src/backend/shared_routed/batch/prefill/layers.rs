//! One layer of a packed prefill pass over several sequences.

use mircuda::{DeviceBuffer, bf16};

use super::super::{
    layer::SharedRoutedBatchLayer,
    states::{attention_states, full_states, linear_states, mamba_states},
};
use crate::{CudaSharedRoutedModelSession, Error, PagedPrefillBatch, Result};

/// Row layout of a packed prefill pass. Rows past the logical counts are
/// padding that only row-independent projections touch.
pub(super) struct PrefillRows<'a> {
    pub(super) counts: &'a [usize],
    /// Every row is logical and all sequences have the same length.
    pub(super) uniform: bool,
    pub(super) positions: &'a DeviceBuffer<u32>,
    pub(super) paging: &'a PagedPrefillBatch,
}

impl SharedRoutedBatchLayer {
    pub(super) fn execute_prefill(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        index: usize,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: &PrefillRows<'_>,
    ) -> Result<()> {
        match self {
            Self::Linear(layer) => {
                let mut states = linear_states(sessions, index)?;
                // A row with an armed checkpoint splits its recurrence,
                // which only the ragged path does.
                if rows.counts.len() > 1
                    && rows.uniform
                    && states.iter().all(|state| state.armed_checkpoint().is_none())
                {
                    layer.execute_packed(input, &mut states, output)
                } else {
                    layer.execute_ragged(input, &mut states, rows.counts, output)
                }
            },
            Self::Full(layer) => layer.execute_packed_prefill(
                input,
                rows.positions,
                &mut full_states(sessions, index)?,
                rows.paging,
                output,
            ),
            Self::Mamba2(layer) => layer.execute_ragged(
                input,
                &mut mamba_states(sessions, index)?,
                rows.counts,
                output,
            ),
            Self::Attention(layer) => {
                let mut states = attention_states(sessions, index)?;
                let state = states
                    .first_mut()
                    .ok_or(Error::InvalidDecoderKernel("packed prefill has no sessions"))?;
                layer.execute_packed_prefill(input, state, rows.paging, output)
            },
        }
    }
}
