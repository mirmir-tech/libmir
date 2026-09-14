use super::CompiledDecode;
use crate::engine::{Array, GatedDeltaState, Result, Stream, gated_delta::state::StateArray};

impl CompiledDecode {
    pub(in crate::engine::gated_delta::layer) fn forward_batch(
        &self,
        input: &Array,
        states: &mut [&mut GatedDeltaState],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let Some((values, histories)) = state_arrays(states) else {
            return Ok(None);
        };
        let values = StateArray::join(&values, stream)?;
        let histories = StateArray::join(&histories, stream)?;
        let [output, next_values, next_histories] = self
            .selected(stream)
            .call(stream.native(), [input.native(), values.native(), histories.native()])?;
        let output = Array::from_native(output)?;
        let next_values =
            StateArray::split(Array::from_native(next_values)?, states.len(), stream)?;
        let next_histories =
            StateArray::split(Array::from_native(next_histories)?, states.len(), stream)?;
        for ((state, value), history) in states.iter_mut().zip(next_values).zip(next_histories) {
            state.commit_packed_decode(value, history);
        }
        Ok(Some(output))
    }
}

fn state_arrays<'a>(
    states: &'a [&mut GatedDeltaState],
) -> Option<(Vec<&'a StateArray>, Vec<&'a StateArray>)> {
    let mut values = Vec::with_capacity(states.len());
    let mut histories = Vec::with_capacity(states.len());
    for state in states {
        let (value, history) = state.compiled_batch_state()?;
        values.push(value);
        histories.push(history);
    }
    Some((values, histories))
}
