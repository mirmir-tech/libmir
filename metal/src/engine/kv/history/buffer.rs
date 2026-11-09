use std::sync::Arc;

#[cfg(test)]
use super::stats;
use super::{History, KvContext, budget};
use crate::engine::{Array, Error, Result, Stream};

impl History {
    pub(super) fn next(
        previous: Option<&Self>,
        contexts: &[KvContext],
        updates: [&Array; 2],
        stream: &Stream,
    ) -> Result<Self> {
        let first = contexts.first().ok_or(Error::NullHandle("history rows"))?;
        let first_shape = first.keys.native().shape()?;
        let &[1, heads, _, dim] = first_shape.dimensions() else {
            return Err(invalid());
        };
        let dtype = first.keys.native().dtype()?;
        if heads == 0 || dim == 0 || contexts.iter().any(|c| c.mask.is_some() || c.paged.is_some())
        {
            return Err(invalid());
        }
        let mut tokens = Vec::with_capacity(contexts.len());
        for context in contexts {
            let shape = context.keys.native().shape()?;
            let &[1, row_heads, row_tokens, row_dim] = shape.dimensions() else {
                return Err(invalid());
            };
            if row_heads != heads
                || row_dim != dim
                || row_tokens == 0
                || context.values.native().shape()? != shape
                || context.keys.native().dtype()? != dtype
                || context.values.native().dtype()? != dtype
            {
                return Err(invalid());
            }
            tokens.push(row_tokens);
        }
        for update in updates {
            if update.native().shape()?.dimensions() != [contexts.len(), heads, 1, dim]
                || update.native().dtype()? != dtype
            {
                return Err(invalid());
            }
        }
        if !stream.history_budget().allows_reuse() {
            return Err(Error::HistoryBudgetUnavailable);
        }
        if let Some(previous) =
            previous.filter(|p| p.extends_to(contexts.len(), heads, &tokens, dim))
            && previous.buffers[0].native().dtype()? == dtype
        {
            let buffers = stream.kernels().history_append.execute(previous, updates, stream)?;
            #[cfg(test)]
            stats::record(true);
            return Ok(Self {
                buffers,
                shape: previous.shape,
                tokens,
                reservation: Arc::clone(&previous.reservation),
            });
        }
        let capacity = capacity_for(tokens.iter().copied().max().unwrap_or(0))?;
        let shape = [contexts.len(), heads, capacity, dim];
        let bytes = budget::storage_bytes(shape, dtype)?;
        let reservation =
            stream.history_budget().reserve(bytes).ok_or(Error::HistoryBudgetUnavailable)?;
        let padded = |row: usize, array: &Array| -> Result<Array> {
            let empty = Array::from_native(stream.native().graph().full(
                &mirtal::Shape::new([1, heads, capacity - tokens[row], dim])?,
                0.0,
                dtype,
            )?)?;
            Array::concatenate(&[array, &empty], 2, stream)
        };
        let keys = contexts
            .iter()
            .enumerate()
            .map(|(row, context)| padded(row, &context.keys))
            .collect::<Result<Vec<_>>>()?;
        let values = contexts
            .iter()
            .enumerate()
            .map(|(row, context)| padded(row, &context.values))
            .collect::<Result<Vec<_>>>()?;
        let buffers = [
            Array::concatenate(&keys.iter().collect::<Vec<_>>(), 0, stream)?,
            Array::concatenate(&values.iter().collect::<Vec<_>>(), 0, stream)?,
        ];
        #[cfg(test)]
        stats::record(false);
        Ok(Self { buffers, shape, tokens, reservation })
    }

    /// Every row must have grown by exactly one token within the allocation.
    pub(super) fn extends_to(
        &self,
        rows: usize,
        heads: usize,
        tokens: &[usize],
        dim: usize,
    ) -> bool {
        self.shape[0] == rows
            && self.shape[1] == heads
            && self.shape[3] == dim
            && tokens.len() == self.tokens.len()
            && tokens
                .iter()
                .zip(&self.tokens)
                .all(|(next, current)| current + 1 == *next && *next <= self.shape[2])
    }

    pub(super) fn views(&self, stream: &Stream) -> Result<[Array; 2]> {
        let mut stop = self.shape;
        stop[2] = self.tokens.iter().copied().max().unwrap_or(0);
        Ok([
            self.buffers[0].slice(&[0; 4], &stop, stream)?,
            self.buffers[1].slice(&[0; 4], &stop, stream)?,
        ])
    }

    /// Hides the padded tail of shorter rows; equal rows need no mask.
    pub(super) fn padding_mask(&self, queries: &Array, stream: &Stream) -> Result<Option<Array>> {
        if self.tokens.windows(2).all(|pair| pair[0] == pair[1]) {
            return Ok(None);
        }
        let mask = stream.kernels().history_append.padding_mask(&self.tokens)?;
        mask.astype_like(queries, stream).map(Some)
    }
}

pub(super) fn capacity_for(tokens: usize) -> Result<usize> {
    Ok(tokens.checked_add(255).ok_or(Error::ShapeOverflow)? / 256 * 256)
}

fn invalid() -> Error {
    Error::InvalidModel(
        "persistent history requires unmasked native K/V rows and single-token updates".into(),
    )
}
