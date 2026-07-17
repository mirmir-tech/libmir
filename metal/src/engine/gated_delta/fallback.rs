use super::{GatedDeltaInputs, GatedDeltaState};
use crate::engine::{Array, Error, Result, Stream};

pub(super) fn update(
    state: &mut GatedDeltaState,
    inputs: GatedDeltaInputs<'_>,
    sequence: usize,
    stream: &Stream,
) -> Result<Array> {
    let graph = stream.native().graph();
    let query_shape = inputs.query.native().shape()?;
    let value_shape = inputs.value.native().shape()?;
    let repeats = value_shape.dimensions()[2] / query_shape.dimensions()[2];
    let queries = repeated(inputs.query, repeats, stream)?;
    let keys = repeated(inputs.key, repeats, stream)?;
    let [decays, updates] = stream.gated_delta_gates([
        inputs.alpha.native(),
        inputs.beta.native(),
        inputs.a_log.native(),
        inputs.dt_bias.native(),
    ])?;
    let mut current = state
        .value
        .take()
        .ok_or_else(|| Error::InvalidModel("Gated Delta state was not initialized".into()))?
        .native()
        .clone();
    let mut outputs = Vec::with_capacity(sequence);
    for time in 0..sequence {
        let query = time_slice(&queries, time, stream)?;
        let key = time_slice(&keys, time, stream)?;
        let value = time_slice(inputs.value.native(), time, stream)?;
        let decay = time_slice(&decays, time, stream)?;
        let update = time_slice(&updates, time, stream)?;
        let decay = graph.expand_dims(&decay, &[2, 3])?;
        let decayed = graph.multiply(&current, &decay)?;
        let key = graph.expand_dims(&key, &[2])?;
        let memory = graph.reduce_sum(&graph.multiply(&decayed, &key)?, -1, false)?;
        let delta = graph
            .multiply(&graph.subtract(&value, &memory)?, &graph.expand_dims(&update, &[2])?)?;
        current = graph.add(&decayed, &graph.multiply(&key, &graph.expand_dims(&delta, &[3])?)?)?;
        let output = graph.reduce_sum(
            &graph.multiply(&current, &graph.expand_dims(&query, &[2])?)?,
            -1,
            false,
        )?;
        outputs.push(graph.astype(&output, inputs.query.native().dtype()?)?);
    }
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let output = graph.stack(&output_refs, 1)?;
    state.value = Some(Array::from_native(current)?);
    state.offset += sequence;
    Array::from_native(output)
}

fn repeated(input: &Array, repeats: usize, stream: &Stream) -> Result<mirtal::Array> {
    if repeats == 1 {
        return Ok(input.native().clone());
    }
    Ok(stream.native().graph().repeat(input.native(), i32::try_from(repeats)?, 2)?)
}

fn time_slice(input: &mirtal::Array, time: usize, stream: &Stream) -> Result<mirtal::Array> {
    let shape = input.shape()?;
    let dimensions = shape.dimensions();
    let mut start = vec![0; dimensions.len()];
    let mut stop = dimensions.to_vec();
    start[1] = time;
    stop[1] = time + 1;
    let sliced = stream.native().graph().slice(input, &start, &stop)?;
    Ok(stream.native().graph().squeeze_axis(&sliced, 1)?)
}
