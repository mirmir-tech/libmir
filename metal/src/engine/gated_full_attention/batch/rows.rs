use crate::engine::{Array, Error, Result, Stream};

pub(super) fn sequence_row(
    input: &Array,
    row: usize,
    sequence: i32,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(sequence)?, usize::try_from(heads)?, usize::try_from(head_dim)?],
        stream,
    )
}

pub(super) fn head_row(
    input: &Array,
    row: usize,
    sequence: i32,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array> {
    input.slice(
        &[row, 0, 0, 0],
        &[row + 1, usize::try_from(heads)?, usize::try_from(sequence)?, usize::try_from(head_dim)?],
        stream,
    )
}

pub(super) fn dimension(shape: &[i32], axis: usize) -> Result<i32> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::InvalidModel("packed attention input rank is invalid".into()))
}
