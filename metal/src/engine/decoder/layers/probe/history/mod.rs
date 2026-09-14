use std::cell::RefCell;

use crate::engine::{Array, Error, Result};
pub mod pages;
mod replay;
mod tests;
pub use replay::Replay;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reader {
    JoinedView,
    RowView,
    NativePaged,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mask {
    None,
    Causal,
    Explicit,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bias {
    None,
    Sinks,
}

#[derive(serde::Serialize)]
pub struct Record {
    pub layer: usize,
    pub reader: Reader,
    pub query: Vec<i32>,
    pub keys: Vec<i32>,
    pub values: Vec<i32>,
    pub mask: Mask,
    pub bias: Bias,
}

#[derive(Default)]
struct Trace {
    layer: Option<usize>,
    records: Vec<Record>,
    replays: Vec<Replay>,
}
thread_local! { static ACTIVE:RefCell<Option<Trace>>=const {RefCell::new(None)}; }
pub struct Capture;
impl Capture {
    pub fn begin() -> Result<Self> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                return Err(Error::InvalidModel("nested K/V history capture".into()));
            }
            *active = Some(Trace::default());
            Ok(Self)
        })
    }

    pub fn finish(self) -> Result<(Vec<Record>, Vec<Replay>)> {
        let trace = ACTIVE
            .with(|a| a.borrow_mut().take())
            .ok_or_else(|| Error::InvalidModel("inactive K/V history capture".into()))?;
        drop(self);
        Ok((trace.records, trace.replays))
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with(|a| *a.borrow_mut() = None);
    }
}
pub fn enter_layer(layer: usize) {
    ACTIVE.with(|a| {
        if let Some(trace) = a.borrow_mut().as_mut() {
            trace.layer = Some(layer);
        }
    });
}

pub fn row(
    query: &Array,
    keys: &Array,
    values: &Array,
    mask: Mask,
    bias: Bias,
    reader: Reader,
) -> Result<()> {
    ACTIVE.with(|active| {
        if let Some(trace) = active.borrow_mut().as_mut() {
            trace.records.push(Record {
                layer: layer(trace)?,
                reader,
                query: query.shape()?,
                keys: keys.shape()?,
                values: values.shape()?,
                mask,
                bias,
            });
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn joined(
    query: &Array,
    keys: &[&Array],
    values: &[&Array],
    joined: [&Array; 2],
    output: &Array,
    scale: f32,
    causal: bool,
) -> Result<()> {
    ACTIVE.with(|active| {
        if let Some(trace) = active.borrow_mut().as_mut() {
            let layer = layer(trace)?;
            trace.records.push(Record {
                layer,
                reader: Reader::JoinedView,
                query: query.shape()?,
                keys: joined[0].shape()?,
                values: joined[1].shape()?,
                mask: if causal {
                    Mask::Causal
                } else {
                    Mask::None
                },
                bias: Bias::None,
            });
            trace
                .replays
                .push(Replay::new(layer, query, keys, values, joined, output, scale, causal)?);
        }
        Ok(())
    })
}
fn layer(trace: &Trace) -> Result<usize> {
    trace
        .layer
        .ok_or_else(|| Error::InvalidModel("K/V history capture requires a layer".into()))
}

thread_local! { static ROW_BATCHES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
pub fn row_batches() -> usize {
    ROW_BATCHES.get()
}
pub fn record_row_batch() {
    ROW_BATCHES.set(ROW_BATCHES.get() + 1);
}

thread_local! { static GATHER_BATCHES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
pub fn gather_batches() -> usize {
    GATHER_BATCHES.get()
}
pub fn record_gather_batch() {
    GATHER_BATCHES.set(GATHER_BATCHES.get() + 1);
}
