mod attention;
mod embedding;
mod layer;
mod merger;
mod mlp;
mod prefill;
mod rope;
mod tower;

#[cfg(test)]
mod tests;

pub use tower::SpatialMergeVisionTower;

use crate::engine::{Array, Error, Result, Stream};

pub(super) fn slice_axis(
    input: &Array,
    axis: usize,
    start: usize,
    stop: usize,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape()?;
    if axis >= shape.len() {
        return Err(Error::InvalidModel("spatial-merge vision slice axis exceeds rank".into()));
    }
    let mut starts = vec![0; shape.len()];
    let mut stops = shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    starts[axis] = start;
    stops[axis] = stop;
    input.slice(&starts, &stops, stream)
}

pub(super) fn dimension(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidModel(format!("spatial-merge vision {label} exceeds i32")))
}
