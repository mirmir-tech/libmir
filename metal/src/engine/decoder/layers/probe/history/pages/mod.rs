use std::{cell::RefCell, io::Write};

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Contiguous { first: u32, pages: usize },
    Gathered { runs: usize, pages: usize },
}

pub struct View {
    layer: usize,
    layout: Layout,
    page_ids: Vec<u32>,
    arenas: [Array; 2],
    views: [Array; 2],
}

thread_local! { static ACTIVE: RefCell<Option<Vec<View>>> = const { RefCell::new(None) }; }

pub struct Capture;
impl Capture {
    pub fn begin() -> Result<Self> {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                return Err(Error::InvalidModel("nested page-view capture".into()));
            }
            *active = Some(Vec::new());
            Ok(Self)
        })
    }

    pub fn finish(self) -> Result<Vec<View>> {
        let records = ACTIVE.with(|active| {
            active
                .borrow_mut()
                .take()
                .ok_or_else(|| Error::InvalidModel("inactive page-view capture".into()))
        });
        drop(self);
        records
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

pub fn record(
    layer: usize,
    page_ids: &[u32],
    arenas: [&mirtal::Array; 2],
    views: [&mirtal::Array; 2],
) -> Result<()> {
    ACTIVE.with(|active| {
        if let Some(records) = active.borrow_mut().as_mut() {
            let first = *page_ids.first().ok_or(Error::NullHandle("captured pages"))?;
            let runs = 1 + page_ids
                .windows(2)
                .filter(|pair| pair[0].checked_add(1) != Some(pair[1]))
                .count();
            let layout = if runs == 1 {
                Layout::Contiguous { first, pages: page_ids.len() }
            } else {
                Layout::Gathered { runs, pages: page_ids.len() }
            };
            records.push(View {
                layer,
                layout,
                page_ids: page_ids.to_vec(),
                arenas: [
                    Array::from_native(arenas[0].clone())?,
                    Array::from_native(arenas[1].clone())?,
                ],
                views: [
                    Array::from_native(views[0].clone())?,
                    Array::from_native(views[1].clone())?,
                ],
            });
        }
        Ok(())
    })
}

impl View {
    pub fn aliases_arena(&self) -> Result<bool> {
        for (arena, view) in self.arenas.iter().zip(&self.views) {
            let source = arena
                .native()
                .allocation()?
                .ok_or(Error::NullHandle("captured arena allocation"))?;
            let output = view
                .native()
                .allocation()?
                .ok_or(Error::NullHandle("captured view allocation"))?;
            if source != output {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn roots(&self) -> Vec<&Array> {
        self.arenas.iter().chain(&self.views).collect()
    }

    pub fn inspect(&self, stream: &Stream) -> Result<()> {
        let mut buffers = Vec::new();
        for (kind, arena, view) in
            [("key", &self.arenas[0], &self.views[0]), ("value", &self.arenas[1], &self.views[1])]
        {
            let source = arena
                .native()
                .allocation()?
                .ok_or(Error::NullHandle("captured arena allocation"))?;
            let output = view
                .native()
                .allocation()?
                .ok_or(Error::NullHandle("captured view allocation"))?;
            let aliases = source == output;
            // The allocation observation is independent of the graph-operation label.
            let shape = arena.shape()?;
            let view_shape = view.shape()?;
            let values = arena.to_vec_f32(stream)?;
            let actual = view.to_vec_f32(stream)?;
            let capacity = usize::try_from(shape[1])?;
            let page_size = usize::try_from(shape[2])?;
            let dim = usize::try_from(shape[3])?;
            let tokens = usize::try_from(view_shape[2])?;
            for (index, value) in actual.iter().enumerate() {
                let head = index / (tokens * dim);
                let token = (index / dim) % tokens;
                let page = usize::try_from(self.page_ids[token / page_size])?;
                let source =
                    ((head * capacity + page) * page_size + token % page_size) * dim + index % dim;
                assert!(
                    value.is_finite() && value.to_bits() == values[source].to_bits(),
                    "page view changed logical chronology"
                );
            }
            buffers.push(serde_json::json!({"kind":kind, "arena_shape":shape,
                "view_shape":view_shape,"aliases_arena":aliases,"arena_bytes":source.bytes(),
                "view_allocation_bytes":output.bytes(),"logical_bytes":view.byte_len()?,"bitwise_equal":true}));
        }
        writeln!(
            std::io::stderr().lock(),
            "history.pages: {}",
            serde_json::json!({
                "layer":self.layer,"layout":self.layout,"page_ids":self.page_ids,"buffers":buffers,
            })
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
