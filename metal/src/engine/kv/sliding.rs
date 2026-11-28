use mirtal::DType;

use super::{
    KvCache, KvContext, KvStorage, SlidingStore, clone_array,
    contiguous::{clone_required, context, grow_array, write_array},
};
use crate::engine::{Array, Error, Result, Stream};

/// History a short sliding update keeps beyond its window (masked out), so a
/// prefix snapshot rewinds past the prompt's last template tokens and a block:
/// the next chat turn, or llama-benchy's measured request after its context
/// probe. Decode compacts the ring back to the window.
const SLIDING_REWIND_TOKENS: usize = 64;

impl KvCache {
    /// The view slot holding the newest token, for a windowed cache after
    /// an update: earlier slots hold the tokens before it, cyclically, in
    /// the chronological view and in the decode ring alike.
    pub(crate) fn window_newest_slot(&self) -> Option<usize> {
        match &self.storage {
            KvStorage::Sliding(store) => store.write_index.checked_sub(1),
            KvStorage::Contiguous(_) | KvStorage::Paged(_) => None,
        }
    }
}

impl SlidingStore {
    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        tokens: usize,
        needed: usize,
        stream: &Stream,
    ) -> Result<KvContext> {
        let limit = self.max_context;
        if needed > limit {
            let prompt_tail = tokens == 1 && self.last_update_tokens > 1;
            self.last_update_tokens = tokens;
            if tokens == 1 && !prompt_tail && self.capacity >= limit {
                return self.update_token(keys, values, offset, limit, stream);
            }
            return self.update_chunk(keys, values, offset, limit, tokens, stream);
        }
        if self.keys.is_none() && tokens > 1 {
            self.keys = Some(clone_required(keys)?);
            self.values = Some(clone_required(values)?);
            self.capacity = needed;
            self.write_index = needed;
            return Ok(KvContext {
                keys: clone_required(keys)?,
                values: clone_required(values)?,
                paged: None,
                mask: None,
            });
        }
        if needed > self.capacity {
            let target = (needed.div_ceil(self.step) * self.step).min(limit);
            self.keys = Some(grow_array(self.keys.take(), keys, target, self.capacity, stream)?);
            self.values =
                Some(grow_array(self.values.take(), values, target, self.capacity, stream)?);
            self.capacity = target;
        }
        self.keys = Some(write_array(self.keys.as_ref(), keys, offset, stream)?);
        self.values = Some(write_array(self.values.as_ref(), values, offset, stream)?);
        self.write_index = needed;
        Ok(KvContext {
            keys: context(self.keys.as_ref(), needed, stream)?,
            values: context(self.values.as_ref(), needed, stream)?,
            paged: None,
            mask: None,
        })
    }

    fn update_token(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        limit: usize,
        stream: &Stream,
    ) -> Result<KvContext> {
        if self.capacity > limit || self.slots()? != limit {
            self.compact(offset, limit, stream)?;
        }
        if self.write_index == limit {
            self.write_index = 0;
        }
        self.keys = Some(write_array(self.keys.as_ref(), keys, self.write_index, stream)?);
        self.values = Some(write_array(self.values.as_ref(), values, self.write_index, stream)?);
        self.write_index += 1;
        Ok(KvContext {
            keys: clone_array(self.keys.as_ref())?.ok_or(Error::NullHandle("KV keys"))?,
            values: clone_array(self.values.as_ref())?.ok_or(Error::NullHandle("KV values"))?,
            paged: None,
            mask: None,
        })
    }

    fn update_chunk(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        limit: usize,
        tokens: usize,
        stream: &Stream,
    ) -> Result<KvContext> {
        let previous_keys = self.temporal(self.keys.as_ref(), offset, stream)?;
        let previous_values = self.temporal(self.values.as_ref(), offset, stream)?;
        let held = previous_keys.as_ref().map_or(Ok(0), |keys| -> Result<usize> {
            Ok(usize::try_from(keys.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?)?)
        })?;
        let margin = if tokens < SLIDING_REWIND_TOKENS {
            SLIDING_REWIND_TOKENS
        } else {
            0
        };
        let retained = offset.min(limit.saturating_sub(1) + margin).min(held);
        let context_keys = append(previous_keys, keys, retained, stream)?;
        let context_values = append(previous_values, values, retained, stream)?;
        self.capacity = retained + tokens;
        self.write_index = self.capacity;
        self.keys = Some(Array::from_native(context_keys.native().clone())?);
        self.values = Some(Array::from_native(context_values.native().clone())?);
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
        offset: usize,
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let Some(current) = current else {
            return Ok(None);
        };
        let valid = self.valid_tokens(offset);
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

    fn compact(&mut self, offset: usize, limit: usize, stream: &Stream) -> Result<()> {
        let keys = self
            .temporal(self.keys.as_ref(), offset, stream)?
            .ok_or(Error::NullHandle("KV keys"))?;
        let values = self
            .temporal(self.values.as_ref(), offset, stream)?
            .ok_or(Error::NullHandle("KV values"))?;
        let length = usize::try_from(keys.shape()?.get(2).copied().ok_or(Error::ShapeOverflow)?)?;
        self.keys = Some(slice_tokens(&keys, length - limit, length, stream)?);
        self.values = Some(slice_tokens(&values, length - limit, length, stream)?);
        self.capacity = limit;
        self.write_index = limit;
        Ok(())
    }

    /// Tokens of the sliding chronological view (`temporal`).
    pub(super) const fn valid_tokens(&self, offset: usize) -> usize {
        if offset < self.max_context {
            offset
        } else {
            self.capacity
        }
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
