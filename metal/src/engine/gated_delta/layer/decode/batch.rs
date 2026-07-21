use super::CompiledDecode;
use crate::engine::{Array, GatedDeltaState, Result, Stream};

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
        let values = Array::concatenate(&values, 0, stream)?;
        let histories = Array::concatenate(&histories, 0, stream)?;
        let [output, next_values, next_histories] = self
            .graph
            .call(stream.native(), [input.native(), values.native(), histories.native()])?;
        let output = Array::from_native(output)?;
        let next_values = split_rows(&Array::from_native(next_values)?, states.len(), stream)?;
        let next_histories =
            split_rows(&Array::from_native(next_histories)?, states.len(), stream)?;
        for ((state, value), history) in states.iter_mut().zip(next_values).zip(next_histories) {
            state.commit_compiled_decode(value, history);
        }
        Ok(Some(output))
    }
}

fn state_arrays<'a>(
    states: &'a [&mut GatedDeltaState],
) -> Option<(Vec<&'a Array>, Vec<&'a Array>)> {
    let mut values = Vec::with_capacity(states.len());
    let mut histories = Vec::with_capacity(states.len());
    for state in states {
        let (value, history) = state.compiled_decode_state()?;
        values.push(value);
        histories.push(history);
    }
    Some((values, histories))
}

fn split_rows(input: &Array, rows: usize, stream: &Stream) -> Result<Vec<Array>> {
    let shape = input
        .shape()?
        .into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    (0..rows)
        .map(|row| {
            let mut start = vec![0; shape.len()];
            let mut stop = shape.clone();
            start[0] = row;
            stop[0] = row + 1;
            input.slice(&start, &stop, stream)
        })
        .collect()
}
