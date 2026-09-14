// Experimental duplicate batch history. Cache rows own the cohort; no global
// session identity or pointer-based session lookup is needed.
mod append;
pub mod budget;
mod buffer;
mod stats;
#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use append::Append;
pub use stats::counts;

use super::{KvCache, KvContext};
use crate::engine::{Array, Result, Stream};

#[derive(Debug)]
pub struct Row {
    index: usize,
    history: Arc<History>,
}

#[derive(Debug)]
pub struct History {
    buffers: [Array; 2],
    shape: [usize; 4],
    tokens: usize,
    reservation: Arc<budget::Lease>,
}

impl Row {
    pub(super) fn extend_allocations(
        &self,
        allocations: &mut std::collections::HashSet<mirtal::memory::Allocation>,
    ) -> Result<()> {
        for buffer in &self.history.buffers {
            allocations.insert(
                buffer
                    .native()
                    .allocation()?
                    .ok_or(crate::engine::Error::NullHandle("materialized batch history"))?,
            );
        }
        Ok(())
    }
}

pub fn enabled(
    caches: &[&mut KvCache],
    contexts: &[KvContext],
    causal: bool,
    sequence: i32,
    stream: &Stream,
) -> bool {
    stream.config().diagnostics.history_batching == crate::config::HistoryBatching::Persistent
        && !causal
        && sequence == 1
        && caches.len() > 1
        && caches.iter().all(|c| c.max_context.is_none())
        && contexts.iter().all(|c| c.paged.is_none() && c.mask.is_none())
}

// Remove ownership BEFORE any cache mutation. Every ordinary update/reset
// invalidates its handle, and snapshots never inherit it. Partial failures thus
// cannot publish a cohort with only some rows updated.
pub fn take(caches: &mut [&mut KvCache]) -> Option<Arc<History>> {
    let mut rows = caches.iter_mut().map(|c| c.history.take()).collect::<Vec<_>>();
    let first = rows.first_mut()?.take()?;
    if first.index != 0 || first.history.shape[0] != rows.len() {
        return None;
    }
    rows.iter()
        .enumerate()
        .skip(1)
        .all(|(index, row)| {
            row.as_ref()
                .is_some_and(|row| row.index == index && Arc::ptr_eq(&row.history, &first.history))
        })
        .then_some(first.history)
}

#[allow(clippy::too_many_arguments)]
pub fn execute(
    previous: Option<&History>,
    caches: &mut [&mut KvCache],
    contexts: &[KvContext],
    updates: [&Array; 2],
    queries: &Array,
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let history = match History::next(previous, contexts, updates, stream) {
        Ok(history) => Arc::new(history),
        Err(crate::engine::Error::HistoryBudgetUnavailable) => {
            let keys = Array::concatenate(
                &contexts.iter().map(|c| &c.keys).collect::<Vec<_>>(),
                0,
                stream,
            )?;
            let values = Array::concatenate(
                &contexts.iter().map(|c| &c.values).collect::<Vec<_>>(),
                0,
                stream,
            )?;
            return queries.scaled_dot_product_attention(&keys, &values, scale, false, stream);
        },
        Err(error) => return Err(error),
    };
    let [keys, values] = history.views(stream)?;
    let output = queries.scaled_dot_product_attention(&keys, &values, scale, false, stream)?;
    for (index, cache) in caches.iter_mut().enumerate() {
        cache.history = Some(Row { index, history: Arc::clone(&history) });
    }
    Ok(output)
}
