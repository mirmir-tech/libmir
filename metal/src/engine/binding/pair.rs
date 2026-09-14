use std::cell::Cell;

use super::BoundLinear;
use crate::engine::{Array, Result, Stream, fused_gate_up::split_last};

thread_local! {
    static JOINED_CALLS: Cell<usize> = const { Cell::new(0) };
}

pub fn joined_calls() -> usize {
    JOINED_CALLS.get()
}

#[derive(Debug)]
pub(in crate::engine) struct KeyValueJoin {
    projection: BoundLinear,
    split: usize,
}

impl KeyValueJoin {
    pub fn new(first: &BoundLinear, second: &BoundLinear, stream: &Stream) -> Result<Option<Self>> {
        let (projection, split) = match (first, second) {
            (BoundLinear::MxFp4(first), BoundLinear::MxFp4(second)) => (
                first.join_outputs(second, stream)?.map(BoundLinear::MxFp4),
                usize::try_from(first.weight.shape()?[0])?,
            ),
            (BoundLinear::Dense(first), BoundLinear::Dense(second)) => (
                first.join_outputs(second, stream)?.map(BoundLinear::Dense),
                first.joined_probe_layout()?.0,
            ),
            _ => return Ok(None),
        };
        Ok(projection.map(|projection| Self { projection, split }))
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        let output = split_last(&self.projection.forward(input, stream)?, self.split, stream)?;
        JOINED_CALLS.update(|calls| calls + 1);
        Ok(output)
    }
}
