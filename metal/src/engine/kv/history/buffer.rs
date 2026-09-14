use super::*;
use crate::engine::Error;

impl History {
    pub(super) fn next(
        previous: Option<&Self>,
        contexts: &[KvContext],
        updates: [&Array; 2],
        stream: &Stream,
    ) -> Result<Self> {
        let first = contexts.first().ok_or(Error::NullHandle("history rows"))?;
        let shape = first.keys.native().shape()?;
        let &[1, heads, tokens, dim] = shape.dimensions() else {
            return Err(invalid());
        };
        let dtype = first.keys.native().dtype()?;
        if heads == 0
            || tokens == 0
            || dim == 0
            || contexts.iter().any(|c| c.mask.is_some() || c.paged.is_some())
        {
            return Err(invalid());
        }
        for context in contexts {
            for array in [&context.keys, &context.values] {
                if array.native().shape()? != shape || array.native().dtype()? != dtype {
                    return Err(invalid());
                }
            }
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
        if let Some(previous) = previous.filter(|p| {
            p.tokens + 1 == tokens
                && p.shape[0] == contexts.len()
                && p.shape[1] == heads
                && p.shape[3] == dim
                && tokens <= p.shape[2]
        }) && previous.buffers[0].native().dtype()? == dtype
        {
            let buffers = stream.kernels().history_append.execute(previous, updates, stream)?;
            stats::record(true);
            return Ok(Self {
                buffers,
                shape: previous.shape,
                tokens,
                reservation: Arc::clone(&previous.reservation),
            });
        }
        let capacity = tokens.checked_add(255).ok_or(Error::ShapeOverflow)? / 256 * 256;
        let shape = [contexts.len(), heads, capacity, dim];
        let bytes = budget::storage_bytes(shape, dtype)?;
        let reservation =
            stream.history_budget().reserve(bytes).ok_or(Error::HistoryBudgetUnavailable)?;
        let graph = stream.native().graph();
        let empty = Array::from_native(graph.full(
            &mirtal::Shape::new([contexts.len(), heads, capacity - tokens, dim])?,
            0.0,
            dtype,
        )?)?;
        let keys =
            Array::concatenate(&contexts.iter().map(|c| &c.keys).collect::<Vec<_>>(), 0, stream)?;
        let values =
            Array::concatenate(&contexts.iter().map(|c| &c.values).collect::<Vec<_>>(), 0, stream)?;
        let buffers = [
            Array::concatenate(&[&keys, &empty], 2, stream)?,
            Array::concatenate(&[&values, &empty], 2, stream)?,
        ];
        stats::record(false);
        Ok(Self { buffers, shape, tokens, reservation })
    }

    pub(super) fn views(&self, stream: &Stream) -> Result<[Array; 2]> {
        let mut stop = self.shape;
        stop[2] = self.tokens;
        Ok([
            self.buffers[0].slice(&[0; 4], &stop, stream)?,
            self.buffers[1].slice(&[0; 4], &stop, stream)?,
        ])
    }
}

fn invalid() -> Error {
    Error::InvalidModel(
        "persistent history requires equal unmasked native K/V rows and single-token updates"
            .into(),
    )
}
