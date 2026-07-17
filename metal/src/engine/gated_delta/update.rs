use super::{GatedDeltaInputs, GatedDeltaState};
use crate::engine::{Array, Error, Result, Stream};

pub(super) fn update(
    state: &mut GatedDeltaState,
    inputs: GatedDeltaInputs<'_>,
    stream: &Stream,
) -> Result<Array> {
    let dimensions = validate(inputs)?;
    ensure_state(state, dimensions, stream)?;
    if dimensions.key_dimension % 32 != 0 {
        return super::fallback::update(state, inputs, dimensions.sequence, stream);
    }
    let [decay, update] = stream.gated_delta_gates([
        inputs.alpha.native(),
        inputs.beta.native(),
        inputs.a_log.native(),
        inputs.dt_bias.native(),
    ])?;
    let current = state
        .value
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("Gated Delta state was not initialized".into()))?;
    let [output, next_state] = stream.gated_delta_recurrence([
        inputs.query.native(),
        inputs.key.native(),
        inputs.value.native(),
        &decay,
        &update,
        current.native(),
    ])?;
    state.value = Some(Array::from_native(next_state)?);
    state.offset += dimensions.sequence;
    Array::from_native(output)
}

pub(super) fn decode(
    state: &mut GatedDeltaState,
    inputs: GatedDeltaInputs<'_>,
    normalize: bool,
    stream: &Stream,
) -> Result<Array> {
    let dimensions = validate(inputs)?;
    if dimensions.sequence != 1 || dimensions.key_dimension % 32 != 0 {
        return Err(Error::InvalidModel(
            "fused Gated Delta decode requires one token and a 32-aligned key dimension".into(),
        ));
    }
    ensure_state(state, dimensions, stream)?;
    let current = state
        .value
        .as_ref()
        .ok_or_else(|| Error::InvalidModel("Gated Delta state was not initialized".into()))?;
    let [output, next_state] = stream.gated_delta_decode(
        [
            inputs.query.native(),
            inputs.key.native(),
            inputs.value.native(),
            inputs.alpha.native(),
            inputs.beta.native(),
            inputs.a_log.native(),
            inputs.dt_bias.native(),
            current.native(),
        ],
        normalize,
    )?;
    state.value = Some(Array::from_native(next_state)?);
    state.offset += 1;
    Array::from_native(output)
}

#[derive(Clone, Copy)]
struct Dimensions {
    batch: usize,
    sequence: usize,
    value_heads: usize,
    key_dimension: usize,
    value_dimension: usize,
}

fn validate(inputs: GatedDeltaInputs<'_>) -> Result<Dimensions> {
    let query = inputs.query.native().shape()?;
    let key = inputs.key.native().shape()?;
    let value = inputs.value.native().shape()?;
    let alpha = inputs.alpha.native().shape()?;
    let beta = inputs.beta.native().shape()?;
    let a_log = inputs.a_log.native().shape()?;
    let dt_bias = inputs.dt_bias.native().shape()?;
    let query = query.dimensions();
    let value = value.dimensions();
    if query.len() != 4
        || value.len() != 4
        || inputs.query.native().shape()? != key
        || query[0..2] != value[0..2]
        || value[2] % query[2] != 0
        || alpha != beta
        || alpha.dimensions() != [value[0], value[1], value[2]]
        || a_log != dt_bias
        || a_log.dimensions() != [value[2]]
    {
        return Err(Error::InvalidModel("Gated Delta input shapes are incompatible".into()));
    }
    Ok(Dimensions {
        batch: value[0],
        sequence: value[1],
        value_heads: value[2],
        key_dimension: query[3],
        value_dimension: value[3],
    })
}

fn ensure_state(
    state: &mut GatedDeltaState,
    dimensions: Dimensions,
    stream: &Stream,
) -> Result<()> {
    let shape = mirtal::Shape::new([
        dimensions.batch,
        dimensions.value_heads,
        dimensions.value_dimension,
        dimensions.key_dimension,
    ])?;
    match state.value.as_ref() {
        Some(value) if value.native().shape()? == shape => Ok(()),
        Some(_) => Err(Error::InvalidModel(
            "Gated Delta cache shape differs from the current request".into(),
        )),
        None => {
            let value = stream.native().graph().full(&shape, 0.0, mirtal::DType::Float32)?;
            state.value = Some(Array::from_native(value)?);
            Ok(())
        },
    }
}
