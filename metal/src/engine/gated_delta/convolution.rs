use super::GatedDeltaState;
use crate::engine::{Array, Error, Result, Stream};

pub(super) fn convolve(
    state: &mut GatedDeltaState,
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Array> {
    let input_shape = input.native().shape()?;
    let weight_shape = weight.native().shape()?;
    let input_dimensions = input_shape.dimensions();
    let weight_dimensions = weight_shape.dimensions();
    if input_dimensions.len() != 3
        || weight_dimensions.len() != 3
        || weight_dimensions[0] != input_dimensions[2]
        || weight_dimensions[1] < 2
        || weight_dimensions[2] != 1
    {
        return Err(Error::InvalidModel("Gated Delta convolution shapes are incompatible".into()));
    }
    let graph = stream.native().graph();
    let history_shape =
        mirtal::Shape::new([input_dimensions[0], weight_dimensions[1] - 1, input_dimensions[2]])?;
    let history = match state.convolution.take() {
        Some(history) if history.native().shape()? == history_shape => history,
        Some(_) => {
            return Err(Error::InvalidModel(
                "Gated Delta convolution cache shape differs from the request".into(),
            ));
        },
        None => Array::from_native(graph.full(&history_shape, 0.0, input.native().dtype()?)?)?,
    };
    let combined = graph.concatenate(&[history.native(), input.native()], 1)?;
    let start = [0, input_dimensions[1], 0];
    let stop =
        [input_dimensions[0], input_dimensions[1] + weight_dimensions[1] - 1, input_dimensions[2]];
    state.convolution = Some(Array::from_native(graph.slice(&combined, &start, &stop)?)?);
    Array::from_native(graph.silu(&graph.conv1d(
        &combined,
        weight.native(),
        1,
        0,
        1,
        i32::try_from(input_dimensions[2])?,
    )?)?)
}
