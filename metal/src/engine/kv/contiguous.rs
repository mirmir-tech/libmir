use super::{KvCache, KvContext, PagedContextMode};
use crate::engine::{Array, Error, Result, Stream};

impl KvCache {
    pub(super) fn update_inner(
        &mut self,
        keys: &Array,
        values: &Array,
        stream: &Stream,
        page_min_context: usize,
        mode: PagedContextMode,
    ) -> Result<KvContext> {
        let token_count = validate(keys, values)?;
        let needed = self.offset.checked_add(token_count).ok_or(Error::ShapeOverflow)?;
        if self.max_context.is_some_and(|limit| needed > limit) {
            return self.update_sliding(keys, values, stream, token_count, needed);
        }
        if let Some(pages) = self.pages.as_mut().filter(|pages| pages.active()) {
            pages.update(keys, values, self.offset, stream)?;
            self.offset = needed;
            return page_context(pages, keys, values, needed, stream, mode);
        }
        if self.keys.is_none() && self.max_context.is_some() && token_count > 1 {
            self.keys = Some(clone_required(keys)?);
            self.values = Some(clone_required(values)?);
            self.offset = needed;
            self.capacity = needed;
            self.write_index = needed;
            return Ok(KvContext {
                keys: clone_required(keys)?,
                values: clone_required(values)?,
                paged: None,
                mask: None,
            });
        }
        let activate_pages =
            (page_min_context == 0 || needed >= page_min_context) && self.pages.is_some();
        if activate_pages && self.offset == 0 && self.keys.is_none() {
            let pages = self.pages.as_mut().ok_or(Error::NullHandle("paged storage"))?;
            pages.update(keys, values, 0, stream)?;
            self.offset = needed;
            return page_context(pages, keys, values, needed, stream, mode);
        }
        self.grow(keys, values, needed, stream)?;
        self.write(keys, values, self.offset, stream)?;
        self.offset = needed;
        if self.max_context.is_some() {
            self.write_index = needed;
        }
        let context_keys = context(self.keys.as_ref(), needed, stream)?;
        let context_values = context(self.values.as_ref(), needed, stream)?;
        if activate_pages && let Some(pages) = self.pages.as_mut() {
            pages.update(&context_keys, &context_values, 0, stream)?;
            self.keys = None;
            self.values = None;
            self.capacity = 0;
            return page_context(pages, keys, values, needed, stream, mode);
        }
        Ok(KvContext {
            keys: context_keys,
            values: context_values,
            paged: None,
            mask: None,
        })
    }

    fn grow(&mut self, keys: &Array, values: &Array, needed: usize, stream: &Stream) -> Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }
        let rounded = needed.div_ceil(self.step) * self.step;
        let target = self.max_context.map_or(rounded, |limit| rounded.min(limit));
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

fn validate(keys: &Array, values: &Array) -> Result<usize> {
    let key_shape = keys.native().shape()?;
    let value_shape = values.native().shape()?;
    if key_shape.dimensions().len() != 4
        || value_shape.dimensions().len() != 4
        || key_shape.dimensions()[2] != value_shape.dimensions()[2]
    {
        return Err(Error::InvalidModel("KV cache expects matching rank-four arrays".into()));
    }
    Ok(key_shape.dimensions()[2])
}

fn grow_array(
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

fn write_array(
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

fn context(current: Option<&Array>, tokens: usize, stream: &Stream) -> Result<Array> {
    let current = current.ok_or(Error::NullHandle("KV cache context"))?;
    let shape = current.native().shape()?;
    let mut stop = shape.dimensions().to_vec();
    stop[2] = tokens;
    Array::from_native(stream.native().graph().slice(current.native(), &[0, 0, 0, 0], &stop)?)
}

fn clone_required(input: &Array) -> Result<Array> {
    Array::from_native(input.native().clone())
}
