mod attention;
mod embedding;
mod layer;
mod mlp;
mod pooler;
mod prefill;
mod rope;
mod tower;

#[cfg(test)]
mod tests;

pub use tower::PooledVisionTower;

use crate::engine::{Array, Error, Result, Stream};

fn slice_axis(
    input: &Array,
    axis: usize,
    start: usize,
    stop: usize,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape()?;
    if axis >= shape.len() {
        return Err(Error::InvalidModel(format!(
            "vision slice axis {axis} exceeds rank {}",
            shape.len()
        )));
    }
    let mut starts = vec![0; shape.len()];
    let mut stops = shape
        .into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    starts[axis] = start;
    stops[axis] = stop;
    input.slice(&starts, &stops, stream)
}

fn dimension(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_or_else(|_| Err(Error::InvalidModel(format!("pooled vision {label} exceeds i32"))), Ok)
}
