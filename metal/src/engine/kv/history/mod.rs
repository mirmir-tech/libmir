// Experimental duplicate batch history. Cache rows own the cohort; no global
// session identity or pointer-based session lookup is needed.
mod append;
mod attend;
pub mod budget;
mod buffer;
#[cfg(test)]
mod stats;
#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use append::Append;
pub use attend::{Attend, attend};
#[cfg(test)]
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
    /// Valid tokens per row; rows may differ and are right-padded to
    /// `shape[2]`.
    tokens: Vec<usize>,
    reservation: Arc<budget::Lease>,
}

impl Row {
    #[cfg(test)]
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
) -> bool {
    !causal
        && sequence == 1
        && caches.len() > 1
        && caches.iter().all(|c| c.max_context.is_none())
        && contexts.iter().all(|c| c.paged.is_none() && c.mask.is_none())
}

/// Predicts whether the next step can keep a batched history within budget, so
/// callers can select per-row attention before any cache mutation.
pub fn admits(
    previous: Option<&History>,
    [rows, heads, dim]: [usize; 3],
    tokens: &[usize],
    dtype: mirtal::DType,
    stream: &Stream,
) -> Result<bool> {
    let budget = stream.history_budget();
    if previous.is_some_and(|history| history.extends_to(rows, heads, tokens, dim)) {
        return Ok(budget.allows_reuse());
    }
    let capacity = buffer::capacity_for(tokens.iter().copied().max().unwrap_or(0))?;
    Ok(budget.admits(budget::storage_bytes([rows, heads, capacity, dim], dtype)?))
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
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let history = match History::next(previous, contexts, updates, stream) {
        Ok(history) => Arc::new(history),
        Err(crate::engine::Error::HistoryBudgetUnavailable) => {
            return row_attention(queries, contexts, scale, sinks, stream);
        },
        Err(error) => return Err(error),
    };
    let [keys, values] = history.views(stream)?;
    let mask = history.padding_mask(queries, stream)?;
    let output = match (mask.as_ref(), sinks) {
        (Some(mask), Some(sinks)) => queries.masked_scaled_dot_product_attention_with_sinks(
            &keys, &values, scale, mask, sinks, stream,
        )?,
        (Some(mask), None) => {
            queries.scaled_dot_product_attention_with_mask(&keys, &values, scale, mask, stream)?
        },
        (None, Some(sinks)) => queries
            .scaled_dot_product_attention_with_sinks(&keys, &values, scale, false, sinks, stream)?,
        (None, None) => {
            queries.scaled_dot_product_attention(&keys, &values, scale, false, stream)?
        },
    };
    for (index, cache) in caches.iter_mut().enumerate() {
        cache.history = Some(Row { index, history: Arc::clone(&history) });
    }
    Ok(output)
}

/// Fallback when the budget refuses a batched history; rows may differ in
/// length.
fn row_attention(
    queries: &Array,
    contexts: &[KvContext],
    scale: f32,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let shape = queries.native().shape()?;
    let dimensions = shape.dimensions();
    let rows = contexts
        .iter()
        .enumerate()
        .map(|(row, context)| {
            let query = queries.slice(
                &[row, 0, 0, 0],
                &[row + 1, dimensions[1], dimensions[2], dimensions[3]],
                stream,
            )?;
            sinks.map_or_else(
                || {
                    query.scaled_dot_product_attention(
                        &context.keys, &context.values, scale, false, stream,
                    )
                },
                |sinks| {
                    query.scaled_dot_product_attention_with_sinks(
                        &context.keys, &context.values, scale, false, sinks, stream,
                    )
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)
}
