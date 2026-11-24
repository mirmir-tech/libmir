//! Per-row states of a verification pass. An armed state keeps, for each row
//! of its next pass, the recurrent state after that row and the convolution
//! input the pass saw, so the cache can commit to any accepted row without
//! recomputing it.

use super::{GatedDeltaState, StateArray};
use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct RowTrail {
    /// The state's offset before the pass.
    start: usize,
    /// The recurrent state after every row, `[rows, batch, heads, dv, dk]`.
    values: Option<Array>,
    /// The convolution history followed by the pass's rows,
    /// `[batch, window + rows, channels]`.
    history: Option<Array>,
}

impl GatedDeltaState {
    /// Records the next pass's per-row states for [`Self::commit_rows`].
    pub(crate) fn arm_row_trail(&mut self) {
        self.trail = Some(RowTrail {
            start: self.offset,
            values: None,
            history: None,
        });
    }

    pub(super) const fn trail_armed(&self) -> bool {
        self.trail.is_some()
    }

    pub(super) fn record_history(&mut self, combined: &mirtal::Array) -> Result<()> {
        if let Some(trail) = self.trail.as_mut() {
            trail.history = Some(Array::from_native(combined.clone())?);
        }
        Ok(())
    }

    pub(super) fn record_values(&mut self, values: Array) {
        if let Some(trail) = self.trail.as_mut() {
            trail.values = Some(values);
        }
    }

    /// Keeps the first `rows` rows of the armed pass and disarms the trail.
    pub(crate) fn commit_rows(&mut self, rows: usize, stream: &Stream) -> Result<()> {
        let trail = self.trail.take().ok_or_else(|| missing("no armed trail"))?;
        let values = trail.values.ok_or_else(|| missing("the pass recorded no states"))?;
        let history = trail.history.ok_or_else(|| missing("the pass recorded no history"))?;
        let state = values.shape()?;
        let recorded = usize::try_from(state[0])?;
        if rows == 0 || rows > recorded || trail.start + recorded != self.offset {
            return Err(missing("the committed rows are outside the pass"));
        }
        let dimensions = state
            .iter()
            .map(|&value| usize::try_from(value))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut stop = dimensions.clone();
        stop[0] = rows;
        let start = [&[rows - 1][..], &vec![0; dimensions.len() - 1]].concat();
        let value = values.slice(&start, &stop, stream)?.reshape(&state[1..], stream)?;
        let window = history.shape()?;
        let [batch, combined, channels] = window[..] else {
            return Err(missing("the recorded history is not rank 3"));
        };
        let width = usize::try_from(combined)? - recorded;
        let convolution = history.slice(
            &[0, rows, 0],
            &[usize::try_from(batch)?, rows + width, usize::try_from(channels)?],
            stream,
        )?;
        self.value = Some(StateArray::from(value));
        self.convolution = Some(StateArray::from(convolution));
        self.offset = trail.start + rows;
        Ok(())
    }
}

fn missing(reason: &str) -> Error {
    Error::InvalidModel(format!("Gated Delta row trail: {reason}"))
}

#[cfg(test)]
mod tests;
