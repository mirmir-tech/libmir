use super::{ContiguousStore, KvCache, KvContext, KvStorage, PagedContextMode};
use crate::engine::{Array, Error, Result, Stream};

impl ContiguousStore {
    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        needed: usize,
        stream: &Stream,
    ) -> Result<KvContext> {
        self.grow(keys, values, needed, None, stream)?;
        self.write(keys, values, offset, stream)?;
        let context_keys = context(self.keys.as_ref(), needed, stream)?;
        let context_values = context(self.values.as_ref(), needed, stream)?;
        Ok(KvContext {
            keys: context_keys,
            values: context_values,
            paged: None,
            mask: None,
        })
    }

    pub(super) fn grow(
        &mut self,
        keys: &Array,
        values: &Array,
        needed: usize,
        limit: Option<usize>,
        stream: &Stream,
    ) -> Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }
        let rounded = needed.div_ceil(self.step) * self.step;
        let target = limit.map_or(rounded, |limit| rounded.min(limit));
        self.keys = Some(grow_array(self.keys.take(), keys, target, self.capacity, stream)?);
        self.values = Some(grow_array(self.values.take(), values, target, self.capacity, stream)?);
        self.capacity = target;
        Ok(())
    }

    pub(super) fn write(
        &mut self,
        keys: &Array,
        values: &Array,
        offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.keys = Some(write_array(self.keys.as_ref(), keys, offset, stream)?);
        self.values = Some(write_array(self.values.as_ref(), values, offset, stream)?);
        Ok(())
    }

    pub(super) fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.capacity = 0;
    }
}

impl KvCache {
    pub(super) fn update_inner(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        _page_min_context: usize,
        mode: PagedContextMode,
    ) -> Result<KvContext> {
        let token_count = self.validate_update(keys, values)?;
        self.history = None;
        let needed = self.offset.checked_add(token_count).ok_or(Error::ShapeOverflow)?;
        match &mut self.storage {
            KvStorage::Contiguous(store) => {
                let output = store.update(keys, values, self.offset, needed, stream)?;
                self.offset = needed;
                Ok(output)
            },
            KvStorage::Sliding(store) => {
                let output =
                    store.update(keys, values, self.offset, token_count, needed, stream)?;
                self.offset = needed;
                Ok(output)
            },
            KvStorage::Paged(pages) => {
                pages.update(keys, values, self.offset, stream)?;
                self.offset = needed;
                page_context(pages, keys, values, needed, stream, mode)
            },
        }
    }
}

fn page_context(
    pages: &super::paged::PagedStore,
    update_keys: &Array,
    update_values: &Array,
    tokens: usize,
    stream: &Stream,
    mode: PagedContextMode,
) -> Result<KvContext> {
    let native = pages.quantized()
        || match mode {
            PagedContextMode::Native => true,
            PagedContextMode::NativeIfFragmented => pages.fragmented(),
            PagedContextMode::View | PagedContextMode::Both => false,
        };
    let (keys, values) = if native {
        (clone_required(update_keys)?, clone_required(update_values)?)
    } else {
        pages.context(tokens, stream)?
    };
    Ok(KvContext {
        keys,
        values,
        paged: (native || mode == PagedContextMode::Both)
            .then(|| pages.context_for_attention(tokens, stream))
            .transpose()?,
        mask: None,
    })
}

pub(super) fn grow_array(
    current: Option<Array>,
    update: &Array,
    target: usize,
    capacity: usize,
    stream: &Stream,
) -> Result<Array> {
    let graph = stream.native().graph();
    let mut shape = update.native().shape()?.dimensions().to_vec();
    shape[2] = if current.is_some() {
        target - capacity
    } else {
        target
    };
    let empty = graph.full(&mirtal::Shape::new(shape)?, 0.0, update.native().dtype()?)?;
    Array::from_native(match current {
        Some(current) => graph.concatenate(&[current.native(), &empty], 2)?,
        None => empty,
    })
}

pub(super) fn write_array(
    current: Option<&Array>,
    update: &Array,
    offset: usize,
    stream: &Stream,
) -> Result<Array> {
    let current = current.ok_or(Error::NullHandle("KV cache array"))?;
    let mut start = vec![0; 4];
    let mut stop = update.native().shape()?.dimensions().to_vec();
    start[2] = offset;
    stop[2] += offset;
    Array::from_native(stream.native().graph().slice_update(
        current.native(),
        update.native(),
        &start,
        &stop,
    )?)
}

pub(super) fn context(current: Option<&Array>, tokens: usize, stream: &Stream) -> Result<Array> {
    let current = current.ok_or(Error::NullHandle("KV cache context"))?;
    let shape = current.native().shape()?;
    let mut stop = shape.dimensions().to_vec();
    stop[2] = tokens;
    Array::from_native(stream.native().graph().slice(current.native(), &[0, 0, 0, 0], &stop)?)
}

pub(super) fn clone_required(input: &Array) -> Result<Array> {
    Array::from_native(input.native().clone())
}
