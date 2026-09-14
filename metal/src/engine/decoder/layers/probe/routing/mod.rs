use std::cell::RefCell;

use crate::engine::{Array, Error, Result};

pub struct Route {
    pub layer: usize,
    pub indices: Array,
}

#[derive(Default)]
struct Trace {
    layer: Option<usize>,
    routes: Vec<Route>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub struct Capture;

impl Capture {
    pub fn begin() -> Result<Self> {
        ACTIVE.with(|active| {
            let mut trace = active.borrow_mut();
            if trace.is_some() {
                return Err(Error::InvalidModel("nested route capture".into()));
            }
            *trace = Some(Trace::default());
            Ok(Self)
        })
    }

    pub fn finish(self) -> Result<Vec<Route>> {
        let routes = ACTIVE.with(|active| {
            active
                .borrow_mut()
                .take()
                .map(|trace| trace.routes)
                .ok_or_else(|| Error::InvalidModel("inactive route capture".into()))
        });
        drop(self);
        routes
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

pub fn enter_layer(layer: usize) {
    ACTIVE.with(|active| {
        if let Some(trace) = active.borrow_mut().as_mut() {
            trace.layer = Some(layer);
        }
    });
}

pub fn record(indices: &Array) -> Result<()> {
    ACTIVE.with(|active| {
        if let Some(trace) = active.borrow_mut().as_mut() {
            let layer = trace.layer.ok_or_else(|| {
                Error::InvalidModel("route capture requires an entered decoder layer".into())
            })?;
            trace.routes.push(Route { layer, indices: indices.snapshot()? });
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests;
