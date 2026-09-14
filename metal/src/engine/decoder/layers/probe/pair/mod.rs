use super::{ACTIVE, ProjectionKind};
use crate::engine::{Array, Error, Result, Stream, binding::BoundLinear};

mod cost;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Plan {
    Separate,
    Joined,
}

pub struct ProjectionPair {
    pub layer: usize,
    input: Array,
    first: BoundLinear,
    second: BoundLinear,
    joined: BoundLinear,
    split: usize,
}

impl ProjectionPair {
    pub(in crate::engine) fn capture(
        first: &BoundLinear,
        second: &BoundLinear,
        input: &Array,
        stream: &Stream,
    ) -> Result<()> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            let Some(trace) = active.as_mut() else {
                return Ok(());
            };
            let Some(last) = trace.values.last() else {
                return Ok(());
            };
            if !trace.targets.contains(&(last.layer, ProjectionKind::KeyValue)) {
                return Ok(());
            }
            trace.pairs.push(Self::snapshot(last.layer, first, second, input, stream)?);
            Ok(())
        })
    }

    fn snapshot(
        layer: usize,
        first: &BoundLinear,
        second: &BoundLinear,
        input: &Array,
        stream: &Stream,
    ) -> Result<Self> {
        let input = input.snapshot()?;
        match (first, second) {
            (BoundLinear::MxFp4(first), BoundLinear::MxFp4(second)) => {
                let joined = first.join_outputs(second, stream)?.ok_or_else(|| {
                    Error::InvalidQuantization("incompatible MXFP4 K/V join".into())
                })?;
                Ok(Self {
                    layer,
                    input,
                    split: usize::try_from(first.weight.shape()?[0])?,
                    first: BoundLinear::MxFp4(first.snapshot()?),
                    second: BoundLinear::MxFp4(second.snapshot()?),
                    joined: BoundLinear::MxFp4(joined),
                })
            },
            (BoundLinear::Dense(first), BoundLinear::Dense(second)) => {
                let joined = first
                    .join_outputs(second, stream)?
                    .ok_or_else(|| Error::InvalidModel("incompatible dense K/V join".into()))?;
                Ok(Self {
                    layer,
                    input,
                    split: first.joined_probe_layout()?.0,
                    first: BoundLinear::Dense(first.snapshot_unclipped()?),
                    second: BoundLinear::Dense(second.snapshot_unclipped()?),
                    joined: BoundLinear::Dense(joined),
                })
            },
            _ => Err(Error::InvalidQuantization(
                "K/V probe requires matching dense or MXFP4 matrices".into(),
            )),
        }
    }

    fn forward(&self, plan: Plan, stream: &Stream) -> Result<(Array, Array)> {
        if plan == Plan::Joined {
            crate::engine::fused_gate_up::split_last(
                &self.joined.forward(&self.input, stream)?,
                self.split,
                stream,
            )
        } else {
            Ok((
                self.first.forward(&self.input, stream)?,
                self.second.forward(&self.input, stream)?,
            ))
        }
    }
}
