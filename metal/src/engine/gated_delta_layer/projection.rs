use super::{Array, Error, Result, Stream};

pub(super) fn split_qkv(
    input: &Array,
    key_width: usize,
    value_width: usize,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let shape = input.native().shape()?;
    let dimensions = shape.dimensions();
    if dimensions.len() != 3 || dimensions[2] != key_width * 2 + value_width {
        return Err(Error::InvalidModel(
            "Gated Delta QKV projection has an unexpected width".into(),
        ));
    }
    let graph = stream.native().graph();
    let slice = |start, stop| {
        Array::from_native(graph.slice(
            input.native(),
            &[0, 0, start],
            &[dimensions[0], dimensions[1], stop],
        )?)
    };
    Ok((
        slice(0, key_width)?,
        slice(key_width, key_width * 2)?,
        slice(key_width * 2, dimensions[2])?,
    ))
}
