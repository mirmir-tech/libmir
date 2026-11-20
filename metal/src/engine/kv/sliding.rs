use mirtal::DType;

use super::{KvCache, KvContext, clone_array};
use crate::engine::{Array, Error, Result, Stream};

/// History a short sliding update keeps beyond its window (masked out), so a
/// prefix snapshot rewinds past the prompt's last template tokens and a block:
/// the next chat turn, or llama-benchy's measured request after its context
/// probe. Decode compacts the ring back to the window.
const SLIDING_REWIND_TOKENS: usize = 64;

impl KvCache {
    pub(super) fn update_sliding(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        tokens: usize,
        needed: usize,
    ) -> Result<KvContext> {
        let limit = self.max_context.ok_or_else(|| Error::InvalidModel("missing window".into()))?;
        // A prompt's last token (right after its chunk) keeps the chronological
        // view and its rewind margin for the prefix snapshot; decode compacts
        // it, or a rewound snapshot's longer arrays, into the in-place ring.
        let prompt_tail = tokens == 1 && self.last_update_tokens > 1;
        self.last_update_tokens = tokens;
        if tokens == 1 && !prompt_tail && self.capacity >= limit {
            return self.update_sliding_token(keys, values, stream, limit, needed);
        }
        self.update_sliding_chunk(keys, values, stream, limit, tokens, needed)
    }

    fn update_sliding_token(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        limit: usize,
        needed: usize,
    ) -> Result<KvContext> {
        if self.capacity > limit || self.slots()? != limit {
            self.compact_sliding(limit, stream)?;
        }
        if self.write_index == limit {
            self.write_index = 0;
        }
        self.write(keys, values, self.write_index, stream)?;
        self.write_index += 1;
        self.offset = needed;
        Ok(KvContext {
            keys: clone_array(self.keys.as_ref())?.ok_or(Error::NullHandle("KV keys"))?,
            values: clone_array(self.values.as_ref())?.ok_or(Error::NullHandle("KV values"))?,
            paged: None,
            mask: None,
        })
    }

    fn update_sliding_chunk(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        limit: usize,
        tokens: usize,
        needed: usize,
    ) -> Result<KvContext> {
        let previous_keys = self.temporal(self.keys.as_ref(), limit, stream)?;
        let previous_values = self.temporal(self.values.as_ref(), limit, stream)?;
        // A compacted decode ring holds just the window, less than the margin.
        let held = previous_keys.as_ref().map_or(Ok(0), |keys| -> Result<usize> {
            Ok(usize::try_from(keys.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?)?)
        })?;
        // Only short updates (a prompt's tail and last token) keep the margin:
        // a full chunk itself holds as many past tokens for its successor.
        let margin = if tokens < SLIDING_REWIND_TOKENS {
            SLIDING_REWIND_TOKENS
        } else {
            0
        };
        let retained = self.offset.min(limit.saturating_sub(1) + margin).min(held);
        let context_keys = append(previous_keys, keys, retained, stream)?;
        let context_values = append(previous_values, values, retained, stream)?;
        self.capacity = retained + tokens;
        self.write_index = self.capacity;
        self.keys = Some(Array::from_native(context_keys.native().clone())?);
        self.values = Some(Array::from_native(context_values.native().clone())?);
        self.offset = needed;
        Ok(KvContext {
            keys: context_keys,
            values: context_values,
            paged: None,
            mask: Some(sliding_mask(retained, tokens, limit, stream)?),
        })
    }

    fn temporal(
        &self,
        current: Option<&Array>,
        limit: usize,
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let Some(current) = current else {
            return Ok(None);
        };
        let valid = if self.offset < limit {
            self.offset
        } else {
            self.capacity
        };
        if self.write_index >= valid {
            return slice_tokens(current, 0, valid, stream).map(Some);
        }
        concatenate_slices(current, self.write_index, valid, stream).map(Some)
    }

    fn slots(&self) -> Result<usize> {
        let Some(keys) = self.keys.as_ref() else {
            return Ok(0);
        };
        Ok(usize::try_from(keys.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?)?)
    }

    fn compact_sliding(&mut self, limit: usize, stream: &Stream) -> Result<()> {
        let keys = self
            .temporal(self.keys.as_ref(), limit, stream)?
            .ok_or(Error::NullHandle("KV keys"))?;
        let values = self
            .temporal(self.values.as_ref(), limit, stream)?
            .ok_or(Error::NullHandle("KV values"))?;
        let length = usize::try_from(keys.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?)?;
        self.keys = Some(slice_tokens(&keys, length - limit, length, stream)?);
        self.values = Some(slice_tokens(&values, length - limit, length, stream)?);
        self.capacity = limit;
        self.write_index = limit;
        Ok(())
    }
}

fn append(
    previous: Option<Array>,
    update: &Array,
    retained: usize,
    stream: &Stream,
) -> Result<Array> {
    let Some(previous) = previous else {
        return Array::from_native(update.native().clone());
    };
    let length = previous.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?;
    let start = usize::try_from(length)?.saturating_sub(retained);
    let previous = slice_tokens(&previous, start, usize::try_from(length)?, stream)?;
    Array::from_native(
        stream.native().graph().concatenate(&[previous.native(), update.native()], 2)?,
    )
}

fn concatenate_slices(input: &Array, split: usize, stop: usize, stream: &Stream) -> Result<Array> {
    let tail = slice_tokens(input, split, stop, stream)?;
    let head = slice_tokens(input, 0, split, stream)?;
    Array::from_native(stream.native().graph().concatenate(&[tail.native(), head.native()], 2)?)
}

fn slice_tokens(input: &Array, start: usize, stop: usize, stream: &Stream) -> Result<Array> {
    let shape = input.native().shape()?;
    let mut begin = vec![0; shape.dimensions().len()];
    let mut end = shape.dimensions().to_vec();
    begin[2] = start;
    end[2] = stop;
    Array::from_native(stream.native().graph().slice(input.native(), &begin, &end)?)
}

fn sliding_mask(retained: usize, tokens: usize, window: usize, stream: &Stream) -> Result<Array> {
    let graph = stream.native().graph();
    let keys = graph.arange(0.0, as_f32(retained + tokens)?, 1.0, DType::Uint32)?;
    let queries =
        graph.arange(as_f32(retained)?, as_f32(retained + tokens)?, 1.0, DType::Uint32)?;
    let keys = graph.expand_dims(&keys, &[0])?;
    let queries = graph.expand_dims(&queries, &[-1])?;
    let causal = graph.greater_equal(&queries, &keys)?;
    let lower = graph.less(&queries, &graph.add_scalar(&keys, as_f32(window)?)?)?;
    Array::from_native(graph.logical_and(&causal, &lower)?)
}

fn as_f32(value: usize) -> Result<f32> {
    Ok(value.to_string().parse()?)
}
