use super::ACTIVE;
use crate::engine::{Array, Result, binding::BoundLinear};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    QueryGate,
    Output,
    KeyValue,
}

pub struct Projection {
    pub layer: usize,
    pub kind: ProjectionKind,
    pub input: Array,
    pub weight: Array,
    pub scales: Array,
}

impl Projection {
    pub(in crate::engine) fn capture(
        kind: ProjectionKind,
        linear: &BoundLinear,
        input: &Array,
    ) -> Result<()> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            let Some(trace) = active.as_mut() else {
                return Ok(());
            };
            let Some(last) = trace.values.last() else {
                return Ok(());
            };
            if !trace.targets.contains(&(last.layer, kind)) {
                return Ok(());
            }
            let BoundLinear::MxFp4(linear) = linear else {
                return Ok(());
            };
            assert!(!linear.has_bias, "projection probe requires unbiased MXFP4");
            trace.projections.push(Self {
                layer: last.layer,
                kind,
                input: input.snapshot()?,
                weight: linear.weight.snapshot()?,
                scales: linear.scales.snapshot()?,
            });
            Ok(())
        })
    }
}
