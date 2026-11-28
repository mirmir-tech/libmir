//! The partial last page of a prefix snapshot. The runtime caches a prompt's
//! whole blocks only, and the request that wrote the partial one goes on
//! decoding into it, so a snapshot keeps a copy of that page and a request
//! restoring the snapshot writes the copy into its own block: a repeated
//! prompt then resumes without recomputing a token.

use std::sync::Arc;

use super::{PagedStore, lock, pool::ArenaPages};
use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct PagedTail {
    /// The page's keys and values, `[kv_heads, 1, page_size, head_dim]`.
    keys: Array,
    values: Array,
    /// The page's index in the sequence.
    page: usize,
}

impl PagedStore {
    /// Gathers a copy of the partial page at `tokens`, evaluated with the
    /// others through [`Self::tail_roots`]. A page ending there needs no copy,
    /// and quantized pages keep none: their tokens are written anew.
    pub(in crate::engine::kv) fn capture_tail(
        &mut self,
        tokens: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.tail = None;
        let Some(storage) =
            self.storage.as_ref().filter(|_| !tokens.is_multiple_of(self.page_size))
        else {
            return Ok(());
        };
        let page = tokens / self.page_size;
        let id = *storage
            .page_ids
            .get(page)
            .ok_or_else(|| Error::InvalidModel("paged tail exceeds initialized storage".into()))?;
        let arena = lock(&storage.arena)?;
        let ArenaPages::Native { keys, values } = &arena.pages else {
            return Ok(());
        };
        // Gathering allocates the copy, which no later write to the page reaches.
        let index = Array::from_u32(&[id], &[1])?;
        let keys = keys.take(&index, 1, stream)?;
        let values = values.take(&index, 1, stream)?;
        drop(arena);
        self.tail = Some(Arc::new(PagedTail { keys, values, page }));
        Ok(())
    }

    /// The copy [`Self::capture_tail`] gathers, to evaluate before the
    /// request that wrote the page decodes into it.
    pub(in crate::engine::kv) fn tail_roots(&self) -> impl Iterator<Item = &Array> {
        self.tail.iter().flat_map(|tail| [&tail.keys, &tail.values])
    }

    /// The copy a snapshot at `tokens` keeps: the partial page's, when
    /// `tokens` ends inside it.
    pub(super) fn tail_at(&self, tokens: usize) -> Option<Arc<PagedTail>> {
        self.tail
            .as_ref()
            .filter(|tail| {
                !tokens.is_multiple_of(self.page_size) && tokens / self.page_size == tail.page
            })
            .map(Arc::clone)
    }

    /// Whether this store resumes at `tokens` from the runtime's first
    /// `cached` tokens: whole pages, with any partial one copied.
    pub(in crate::engine::kv) fn restorable_within(&self, tokens: usize, cached: usize) -> bool {
        tokens <= cached
            || tokens / self.page_size * self.page_size <= cached && self.tail_at(tokens).is_some()
    }

    /// Writes the copied page's first tokens up to `tokens` into the block
    /// bound at its index.
    pub(in crate::engine::kv) fn restore_tail(
        &mut self,
        tokens: usize,
        stream: &Stream,
    ) -> Result<()> {
        let Some(tail) = self.tail_at(tokens) else {
            self.tail = None;
            return Ok(());
        };
        self.tail = None;
        let start = tail.page * self.page_size;
        let count = tokens - start;
        let graph = stream.native().graph();
        let tokens_of = |page: &Array| -> Result<Array> {
            let mut stop = page.native().shape()?.dimensions().to_vec();
            stop[2] = count;
            let rows = page.slice(&[0, 0, 0, 0], &stop, stream)?;
            let rows = graph.transpose(rows.native(), &[1, 0, 2, 3])?;
            Array::from_native(graph.contiguous(&rows)?)
        };
        let keys = tokens_of(&tail.keys)?;
        let values = tokens_of(&tail.values)?;
        self.update(&keys, &values, start, stream)
    }
}
