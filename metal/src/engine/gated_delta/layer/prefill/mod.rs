mod diagnosis;
mod paired;

use super::{Array, Error, GatedDeltaLayer, GatedDeltaState, Result, Stream};
use crate::engine::gated_delta::state::StateArray;

impl GatedDeltaLayer {
    /// Candidate using the ordinary multi-token graph with a packed batch.
    /// Construct all new row handles before replacing live state; snapshots
    /// retain immutable parents and remain valid across cohort changes.
    pub(crate) fn forward_packed_prefill(
        &self,
        input: &Array,
        states: &mut [&mut GatedDeltaState],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let shape = input.shape()?;
        if shape.len() != 3 || usize::try_from(shape[0])? != states.len() || shape[1] <= 1 {
            return Ok(None);
        }
        let Some(first) = states.first() else {
            return Ok(None);
        };
        let offset = first.offset;
        if !states.iter().all(|state| state.offset == offset) {
            return Ok(None);
        }
        let initialized = states
            .iter()
            .map(|state| state.compiled_batch_state())
            .collect::<Option<Vec<_>>>();
        let (value, convolution) = if let Some(arrays) = initialized {
            let values = arrays.iter().map(|(value, _)| *value).collect::<Vec<_>>();
            let histories = arrays.iter().map(|(_, history)| *history).collect::<Vec<_>>();
            (
                Some(StateArray::join(&values, stream)?.into()),
                Some(StateArray::join(&histories, stream)?.into()),
            )
        } else if states
            .iter()
            .all(|state| state.offset == 0 && state.value.is_none() && state.convolution.is_none())
        {
            (None, None)
        } else {
            return Ok(None);
        };
        let mut packed = GatedDeltaState { value, convolution, offset };
        let gates = self.row_gates(input, stream)?;
        let output =
            self.forward_with_gates(input, &mut packed, Some(gates), stream, |output| {
                self.row_output(output, stream)
            })?;
        let (Some(value), Some(history)) = (packed.value.take(), packed.convolution.take()) else {
            return Err(Error::InvalidModel("packed Gated Delta prefill produced no state".into()));
        };
        let values =
            StateArray::split(Array::from_native(value.native().clone())?, states.len(), stream)?;
        let histories =
            StateArray::split(Array::from_native(history.native().clone())?, states.len(), stream)?;
        for ((state, value), history) in states.iter_mut().zip(values).zip(histories) {
            state.value = Some(value);
            state.convolution = Some(history);
            state.offset = packed.offset;
        }
        Ok(Some(output))
    }
}

impl GatedDeltaLayer {
    // MXFP4 alpha/beta and output QMM change arithmetic when rows are flattened.
    // Preserve their row geometry; pack QKV, z, convolution and recurrence.
    fn row_output(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let shape = input
            .shape()?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut rows = Vec::new();
        for row in 0..shape[0] {
            let input = input.slice(&[row, 0, 0], &[row + 1, shape[1], shape[2]], stream)?;
            rows.push(self.out_proj.forward(&input, stream)?);
        }
        Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)
    }

    fn row_gates(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        let shape = input
            .shape()?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut alpha = Vec::new();
        let mut beta = Vec::new();
        for row in 0..shape[0] {
            let input = input.slice(&[row, 0, 0], &[row + 1, shape[1], shape[2]], stream)?;
            alpha.push(self.in_proj_a.forward(&input, stream)?);
            beta.push(self.in_proj_b.forward(&input, stream)?);
        }
        Ok((
            Array::concatenate(&alpha.iter().collect::<Vec<_>>(), 0, stream)?,
            Array::concatenate(&beta.iter().collect::<Vec<_>>(), 0, stream)?,
        ))
    }
}
