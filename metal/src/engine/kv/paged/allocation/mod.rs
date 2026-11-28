use super::{PagedStore, Storage, lock, single_run};
use crate::engine::{Array, Error, Result, Stream};

const PAGE_TABLE_STEP: usize = 64;

pub(super) fn ensure(
    store: &mut PagedStore,
    keys: &Array,
    values: &Array,
    offset: usize,
    stream: &Stream,
) -> Result<()> {
    store.validate_update(keys, values)?;
    let sequence = keys.native().shape()?.dimensions()[2];
    let end = offset.checked_add(sequence).ok_or(Error::ShapeOverflow)?;
    let needed = end.div_ceil(store.page_size);
    if store.storage.is_none() {
        let initial = super::pool::growth_target(0, needed.max(1), store.capacity)?;
        let arena = store
            .pool
            .acquire(store.layer, store.page_size, store.format, keys, initial, stream)?;
        let (arena_capacity, initial_ids) = {
            let mut guard = lock(&arena)?;
            let ids = desired_page_ids(store, &[], needed, &mut guard)?;
            guard.grow(required_pages(&ids), store.capacity, stream)?;
            let cap = guard.capacity;
            drop(guard);
            (cap, ids)
        };
        let table_capacity = table_capacity(initial_ids.len(), arena_capacity);
        let table = page_table(&initial_ids, table_capacity)?;
        let contiguous = single_run(&initial_ids);
        store.storage = Some(Storage {
            input_dtype: keys.native().dtype()?,
            arena,
            table,
            page_ids: initial_ids,
            table_capacity,
            contiguous,
        });
        return Ok(());
    }
    let (arena_capacity, desired) = {
        let storage = store.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let mut guard = lock(&storage.arena)?;
        let ids = desired_page_ids(store, &storage.page_ids, needed, &mut guard)?;
        guard.grow(required_pages(&ids), store.capacity, stream)?;
        let cap = guard.capacity;
        drop(guard);
        (cap, ids)
    };
    let storage = store.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
    sync_page_table(storage, &desired, arena_capacity, stream)
}

pub(super) fn sync_page_table(
    storage: &mut Storage,
    desired: &[u32],
    arena_capacity: usize,
    stream: &Stream,
) -> Result<()> {
    if storage.page_ids == desired {
        return Ok(());
    }
    let is_prefix_append =
        desired.len() > storage.page_ids.len() && desired.starts_with(&storage.page_ids);
    if is_prefix_append && desired.len() <= storage.table_capacity {
        let start = storage.page_ids.len();
        let added = &desired[start..];
        let update = Array::from_u32(added, &[i32::try_from(added.len())?])?;
        storage.table = Array::from_native(stream.native().graph().slice_update(
            storage.table.native(),
            update.native(),
            &[start],
            &[start + added.len()],
        )?)?;
    } else {
        let next_capacity = table_capacity(desired.len(), arena_capacity);
        storage.table = page_table(desired, next_capacity)?;
        storage.table_capacity = next_capacity;
    }
    storage.page_ids.clear();
    storage.page_ids.extend_from_slice(desired);
    storage.contiguous = single_run(&storage.page_ids);
    Ok(())
}

#[cfg_attr(not(test), allow(clippy::needless_pass_by_ref_mut))]
fn desired_page_ids(
    store: &PagedStore,
    existing: &[u32],
    needed: usize,
    arena: &mut super::Arena,
) -> Result<Vec<u32>> {
    if let Some(bound) = &store.bound_blocks {
        if needed > bound.len() {
            return Err(Error::KvPageCapacity {
                layer: store.layer,
                required: needed,
                maximum: bound.len(),
            });
        }
        for &block in bound {
            if usize::try_from(block)? >= store.capacity {
                return Err(Error::KvBlockOutOfRange {
                    layer: store.layer,
                    block,
                    capacity: store.capacity,
                });
            }
        }
        return Ok(bound.clone());
    }
    #[cfg(not(test))]
    {
        let _ = (existing, arena);
        if needed > store.capacity {
            return Err(Error::KvPageCapacity {
                layer: store.layer,
                required: needed,
                maximum: store.capacity,
            });
        }
        (0..needed).map(|index| Ok(u32::try_from(index)?)).collect()
    }
    #[cfg(test)]
    {
        let mut ids = existing.to_vec();
        let extra = needed.saturating_sub(ids.len());
        // Slots count to the configured capacity; the arena grows to them.
        if arena.unbound_refs.len() < store.capacity {
            arena.unbound_refs.resize(store.capacity, 0);
        }
        if extra > 0 {
            let free = arena.unbound_refs.iter().filter(|&&refs| refs == 0).count();
            if extra > free {
                return Err(Error::KvPageCapacity {
                    layer: store.layer,
                    required: store.capacity - free + extra,
                    maximum: store.capacity,
                });
            }
            for _ in 0..extra {
                let slot = arena.unbound_refs.iter().position(|&refs| refs == 0).ok_or(
                    Error::KvPageCapacity {
                        layer: store.layer,
                        required: needed,
                        maximum: store.capacity,
                    },
                )?;
                arena.unbound_refs[slot] = 1;
                ids.push(u32::try_from(slot)?);
            }
        }
        Ok(ids)
    }
}

/// The pages an arena holds for `ids`: up to the highest, as it is indexed
/// by them directly.
fn required_pages(ids: &[u32]) -> usize {
    ids.iter().max().map_or(0, |&id| id as usize + 1)
}

fn table_capacity(count: usize, arena_capacity: usize) -> usize {
    let min_cap = count.max(1).div_ceil(PAGE_TABLE_STEP) * PAGE_TABLE_STEP;
    min_cap.min(arena_capacity.max(count.max(1)))
}

fn page_table(ids: &[u32], capacity: usize) -> Result<Array> {
    let mut values =
        (0..capacity).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
    values[..ids.len()].copy_from_slice(ids);
    Array::from_u32(&values, &[i32::try_from(capacity)?])
}
