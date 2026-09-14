use std::sync::Arc;

use super::{Arena, PagedStore, Storage, lock};
use crate::engine::{Array, Error, Result, Stream};

mod copy;
use copy::copy_page;

pub(super) fn ensure(
    store: &mut PagedStore,
    keys: &Array,
    values: &Array,
    offset: usize,
    stream: &Stream,
) -> Result<()> {
    store.validate_update(keys, values)?;
    let shape = keys.native().shape()?;
    let dimensions = shape.dimensions();
    let sequence = dimensions[2];
    let end = offset.checked_add(sequence).ok_or(Error::ShapeOverflow)?;
    let needed = end.div_ceil(store.page_size);
    let candidate = store
        .storage
        .is_none()
        .then(|| create(store, keys, needed, stream))
        .transpose()?;
    let storage = store
        .storage
        .as_ref()
        .or(candidate.as_ref())
        .ok_or(Error::NullHandle("paged storage"))?;
    let arena_handle = Arc::clone(&storage.arena);
    let mut arena = lock(&arena_handle)?;
    let first = offset / store.page_size;
    let last = (end - 1) / store.page_size;
    let existing = storage.page_ids.len();
    let shared = (first..=last.min(existing.saturating_sub(1)))
        .filter(|logical| {
            storage
                .page_ids
                .get(*logical)
                .and_then(|page| usize::try_from(*page).ok())
                .is_some_and(|page| arena.references[page] > 1)
        })
        .count();
    let free = arena.references.iter().filter(|count| **count == 0).count();
    let planned = store.reserve_pages.max(needed);
    let reservation = storage.reservation_needed(planned);
    let additional = storage.additional_owned_pages(planned, needed, shared);
    let mut required = arena.capacity.saturating_add(additional.saturating_sub(free));
    if reservation > 0
        && !arena.has_contiguous_free(reservation)
        && arena.capacity.saturating_add(reservation) <= store.max_pages
    {
        required = required.max(arena.capacity + reservation);
    }
    let target = growth_target(
        arena.capacity,
        required.max(store.reserve_pages),
        store.allocation_step,
        store.max_pages,
    )?;
    if target > arena.capacity {
        tracing::debug!(
            target: "libmir::metal::kv",
            layer = store.layer,
            capacity_pages = arena.capacity,
            target_pages = target,
            used_pages = arena.references.iter().filter(|count| **count > 0).count(),
            free_pages = free,
            "growing Metal paged K/V arena"
        );
    }
    ensure_capacity(&mut arena, target, stream)?;
    if let Some(candidate) = candidate {
        store.storage = Some(candidate);
    }
    let storage = store.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
    storage.reserve_contiguous(&mut arena, planned)?;
    let table_resized = needed > storage.table_capacity;
    if table_resized {
        storage.table_capacity = round(needed + store.allocation_step, store.allocation_step);
    }
    let appended = needed.saturating_sub(storage.page_ids.len());
    if appended > 0 {
        storage.append_pages(&mut arena, appended)?;
    }
    let mut remapped = false;
    for page_id in &mut storage.page_ids[first..=last] {
        let source = usize::try_from(*page_id)?;
        if arena.references[source] == 1 {
            continue;
        }
        let target = arena.allocate()?;
        copy_page(&mut arena, source, usize::try_from(target)?, stream)?;
        arena.references[source] -= 1;
        *page_id = target;
        remapped = true;
    }
    if table_resized || remapped {
        storage.table = page_table(&storage.page_ids, storage.table_capacity)?;
    } else if appended > 0 {
        append_page_table(storage, needed - appended, stream)?;
    }
    if table_resized || remapped || appended > 0 {
        storage.identity = storage
            .page_ids
            .iter()
            .enumerate()
            .all(|(index, page)| usize::try_from(*page) == Ok(index));
    }
    drop(arena);
    Ok(())
}

fn append_page_table(storage: &mut Storage, start: usize, stream: &Stream) -> Result<()> {
    let ids = &storage.page_ids[start..];
    let update = Array::from_u32(ids, &[i32::try_from(ids.len())?])?;
    storage.table = Array::from_native(stream.native().graph().slice_update(
        storage.table.native(),
        update.native(),
        &[start],
        &[start + ids.len()],
    )?)?;
    Ok(())
}

fn create(store: &PagedStore, keys: &Array, needed: usize, stream: &Stream) -> Result<Storage> {
    let capacity =
        growth_target(0, needed.max(store.reserve_pages), store.allocation_step, store.max_pages)?;
    let arena = store
        .pool
        .acquire(store.layer, store.page_size, store.format, keys, capacity, stream)?;
    Ok(Storage {
        input_dtype: keys.native().dtype()?,
        arena,
        table: page_table(&[], capacity)?,
        page_ids: Vec::new(),
        reserved_page_ids: Vec::new(),
        table_capacity: capacity,
        identity: true,
    })
}

fn ensure_capacity(arena: &mut Arena, required: usize, stream: &Stream) -> Result<()> {
    if required <= arena.capacity {
        return Ok(());
    }
    let capacity = required;
    let shape = mirtal::Shape::new([
        arena.kv_heads,
        capacity - arena.capacity,
        arena.page_size,
        arena.keys.native().shape()?.dimensions()[3],
    ])?;
    let graph = stream.native().graph();
    let dtype = arena.keys.native().dtype()?;
    let extra_keys = graph.full(&shape, 0.0, dtype)?;
    let extra_values = graph.full(&shape, 0.0, dtype)?;
    arena.keys = Array::from_native(graph.concatenate(&[arena.keys.native(), &extra_keys], 1)?)?;
    arena.values =
        Array::from_native(graph.concatenate(&[arena.values.native(), &extra_values], 1)?)?;
    grow_scales(arena, capacity, graph)?;
    arena.references.resize(capacity, 0);
    arena.capacity = capacity;
    Ok(())
}

fn grow_scales(arena: &mut Arena, capacity: usize, graph: mirtal::Graph<'_>) -> Result<()> {
    let shape = mirtal::Shape::new([arena.kv_heads, capacity - arena.capacity, arena.page_size])?;
    for scales in [&mut arena.key_scales, &mut arena.value_scales] {
        if let Some(current) = scales.take() {
            let extra = graph.full(&shape, 1.0, mirtal::DType::Float32)?;
            *scales = Some(Array::from_native(graph.concatenate(&[current.native(), &extra], 1)?)?);
        }
    }
    Ok(())
}

fn page_table(ids: &[u32], capacity: usize) -> Result<Array> {
    let mut values =
        (0..capacity).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
    values[..ids.len()].copy_from_slice(ids);
    Array::from_u32(&values, &[i32::try_from(capacity)?])
}

fn round(value: usize, step: usize) -> usize {
    value.div_ceil(step) * step
}

fn growth_target(current: usize, required: usize, step: usize, maximum: usize) -> Result<usize> {
    if required > maximum {
        return Err(Error::InvalidModel(format!(
            "paged arena requires {required} pages but the configured K/V capacity is {maximum}"
        )));
    }
    if required <= current {
        return Ok(current);
    }
    let geometric = current.saturating_mul(2).max(required);
    Ok(round(geometric, step).min(maximum))
}

#[cfg(test)]
mod tests;
