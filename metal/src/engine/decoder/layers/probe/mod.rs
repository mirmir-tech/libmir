pub use crate::engine::binding::pair::joined_calls;
pub mod history;
mod pair;
mod projection;
pub mod routing;
use std::cell::RefCell;

pub use pair::ProjectionPair;
pub use projection::{Projection, ProjectionKind};
use serde::Serialize;

use crate::engine::{Array, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Input,
    Mixer,
    FeedForward,
    AttentionNorm,
    Query,
    Key,
    Value,
    AttentionGate,
    RotatedQuery,
    RotatedKey,
    RawAttention,
    Attended,
    AttentionProjection,
}

pub struct Value {
    pub layer: usize,
    pub stage: Stage,
    pub array: Array,
}

thread_local! {
    static ACTIVE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub struct Trace {
    pub values: Vec<Value>,
    pub projections: Vec<Projection>,
    pub pairs: Vec<ProjectionPair>,
    targets: Vec<(usize, ProjectionKind)>,
}

pub struct Capture;

impl Capture {
    pub fn begin(targets: &[(usize, ProjectionKind)]) -> Result<Self> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                return Err(Error::InvalidModel("nested layer capture".into()));
            }
            *active = Some(Trace {
                values: Vec::new(),
                projections: Vec::new(),
                pairs: Vec::new(),
                targets: targets.to_vec(),
            });
            Ok(Self)
        })
    }

    pub fn finish(self) -> Result<Trace> {
        let values = ACTIVE.with(|active| {
            active
                .borrow_mut()
                .take()
                .ok_or_else(|| Error::InvalidModel("inactive layer capture".into()))
        });
        drop(self);
        values
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

pub fn record(layer: usize, stage: Stage, array: &Array) -> Result<()> {
    if stage == Stage::Input {
        routing::enter_layer(layer);
        history::enter_layer(layer);
    }
    ACTIVE.with(|active| {
        if let Some(trace) = &mut *active.borrow_mut() {
            trace.values.push(Value { layer, stage, array: array.snapshot()? });
        }
        Ok(())
    })
}

pub fn detail(stage: Stage, array: &Array) -> Result<()> {
    ACTIVE.with(|active| {
        if let Some(trace) = &mut *active.borrow_mut()
            && let Some(previous) = trace.values.last()
        {
            trace.values.push(Value {
                layer: previous.layer,
                stage,
                array: array.snapshot()?,
            });
        }
        Ok(())
    })
}
