use std::sync::Arc;

use super::{PagedStore, allocation, lock};
use crate::engine::{Array, Error, KvPageFormat, Result, Stream, kernels::PageRowsOptions};

/// Appends `[B, H, 1, D]` keys and values, one token per store at its
/// `offset`. Pages are allocated (and copied on write) row by row on the host;
/// stores sharing one arena then take a single write.
pub(in crate::engine::kv) fn write_rows(
    stores: &mut [&mut PagedStore],
    offsets: &[usize],
    [keys, values]: [&Array; 2],
    stream: &Stream,
) -> Result<()> {
    let shape = keys.shape()?;
    let &[_, heads, 1, dim] = shape.as_slice() else {
        return Err(Error::InvalidModel("row page writes take one token per row".into()));
    };
    let [heads, dim] = [usize::try_from(heads)?, usize::try_from(dim)?];
    let row = |array: &Array, index: usize| {
        array.slice(&[index, 0, 0, 0], &[index + 1, heads, 1, dim], stream)
    };
    let mut slots = Vec::with_capacity(stores.len());
    for (index, store) in stores.iter_mut().enumerate() {
        slots.push(store.prepare_row(
            &row(keys, index)?,
            &row(values, index)?,
            offsets[index],
            stream,
        )?);
    }
    let arena = stores
        .first()
        .and_then(|store| store.storage.as_ref())
        .map(|storage| Arc::clone(&storage.arena));
    let shared = arena.as_ref().is_some_and(|arena| {
        stores.iter().all(|store| {
            store.storage.as_ref().is_some_and(|storage| Arc::ptr_eq(arena, &storage.arena))
        })
    });
    let Some(arena) = arena.filter(|_| shared) else {
        // Pages are already in place; rows in different arenas write one by one.
        for (index, store) in stores.iter_mut().enumerate() {
            store.update(&row(keys, index)?, &row(values, index)?, offsets[index], stream)?;
        }
        return Ok(());
    };
    let slots = mirtal::Array::from_slice(&slots, [slots.len()])?;
    let mut arena = lock(&arena)?;
    let [page_keys, page_values] = stream.page_rows_write(
        [keys.native(), values.native(), arena.keys.native(), arena.values.native(), &slots],
        PageRowsOptions {
            rows: stores.len(),
            kv_heads: arena.kv_heads,
            page_capacity: arena.capacity,
            page_size: arena.page_size,
            head_dim: arena.head_dim,
        },
    )?;
    arena.keys = Array::from_native(page_keys)?;
    arena.values = Array::from_native(page_values)?;
    drop(arena);
    Ok(())
}

impl PagedStore {
    /// Allocates the page for `offset` and returns that token's physical slot.
    fn prepare_row(
        &mut self,
        key: &Array,
        value: &Array,
        offset: usize,
        stream: &Stream,
    ) -> Result<u32> {
        if self.format != KvPageFormat::Native {
            return Err(Error::InvalidModel("row page writes need native K/V pages".into()));
        }
        let logical = offset / self.page_size;
        let before = self.storage.as_ref().map(|storage| {
            (
                storage.page_ids.len(),
                storage.page_ids.get(logical).copied(),
                storage.table_capacity,
            )
        });
        allocation::ensure(self, key, value, offset, stream)?;
        let storage = self.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let after = (
            storage.page_ids.len(),
            storage.page_ids.get(logical).copied(),
            storage.table_capacity,
        );
        // The row kernel takes host slots, so nothing in this step reads a
        // changed page table: schedule it now, before a later detach cuts it.
        if before != Some(after) {
            storage.table.async_eval(stream)?;
        }
        let page = storage
            .page_ids
            .get(offset / self.page_size)
            .ok_or_else(|| Error::InvalidModel("row page is not allocated".into()))?;
        let slot = usize::try_from(*page)?
            .checked_mul(self.page_size)
            .and_then(|start| start.checked_add(offset % self.page_size))
            .ok_or(Error::ShapeOverflow)?;
        Ok(u32::try_from(slot)?)
    }
}
