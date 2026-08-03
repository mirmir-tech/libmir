use std::sync::Arc;

use super::{Arena, PagedStore, Storage, lock};
use crate::engine::{Array, Error, Result, Stream};

pub(super) fn ensure(
    store: &mut PagedStore,
    keys: &Array,
    values: &Array,
    offset: usize,
    stream: &Stream,
) -> Result<()> {
    let shape = keys.native().shape()?;
    let dimensions = shape.dimensions();
    if dimensions.len() != 4
        || keys.native().shape()? != values.native().shape()?
        || keys.native().dtype()? != values.native().dtype()?
        || dimensions[0] != 1
    {
        return Err(Error::InvalidModel("paged K/V update arrays are incompatible".into()));
    }
    let sequence = dimensions[2];
    let needed = (offset + sequence).div_ceil(store.page_size);
    if store.storage.is_none() {
        store.storage = Some(create(store, keys, needed, stream)?);
    }
    let storage = store.storage.as_mut().ok_or(Error::NullHandle("paged storage"))?;
    let arena_handle = Arc::clone(&storage.arena);
    let mut arena = lock(&arena_handle)?;
    let first = offset / store.page_size;
    let last = (offset + sequence - 1) / store.page_size;
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
    ensure_capacity(&mut arena, target, store.allocation_step, stream)?;
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
    for logical in first..=last {
        let source = usize::try_from(storage.page_ids[logical])?;
        if arena.references[source] == 1 {
            continue;
        }
        let target = arena.allocate()?;
        copy_page(&mut arena, source, usize::try_from(target)?, stream)?;
        arena.references[source] -= 1;
        storage.page_ids[logical] = target;
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
        arena,
        table: page_table(&[], capacity)?,
        page_ids: Vec::new(),
        reserved_page_ids: Vec::new(),
        table_capacity: capacity,
        identity: true,
    })
}

fn ensure_capacity(arena: &mut Arena, required: usize, step: usize, stream: &Stream) -> Result<()> {
    if required <= arena.capacity {
        return Ok(());
    }
    let capacity = round(required, step);
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

fn copy_page(arena: &mut Arena, source: usize, target: usize, stream: &Stream) -> Result<()> {
    let graph = stream.native().graph();
    let stored_dim = arena.keys.native().shape()?.dimensions()[3];
    let stop = [arena.kv_heads, source + 1, arena.page_size, stored_dim];
    let keys = graph.slice(arena.keys.native(), &[0, source, 0, 0], &stop)?;
    let values = graph.slice(arena.values.native(), &[0, source, 0, 0], &stop)?;
    let target_stop = [arena.kv_heads, target + 1, arena.page_size, stored_dim];
    arena.keys = Array::from_native(graph.slice_update(
        arena.keys.native(),
        &keys,
        &[0, target, 0, 0],
        &target_stop,
    )?)?;
    arena.values = Array::from_native(graph.slice_update(
        arena.values.native(),
        &values,
        &[0, target, 0, 0],
        &target_stop,
    )?)?;
    copy_scales(arena, source, target, graph)?;
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

fn copy_scales(
    arena: &mut Arena,
    source: usize,
    target: usize,
    graph: mirtal::Graph<'_>,
) -> Result<()> {
    for scales in [&mut arena.key_scales, &mut arena.value_scales] {
        if let Some(current) = scales.take() {
            let values = graph.slice(
                current.native(),
                &[0, source, 0],
                &[arena.kv_heads, source + 1, arena.page_size],
            )?;
            let next = graph.slice_update(
                current.native(),
                &values,
                &[0, target, 0],
                &[arena.kv_heads, target + 1, arena.page_size],
            )?;
            *scales = Some(Array::from_native(next)?);
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
        return Err(Error::InvalidModel(
            format!(
                "paged arena requires {required} pages but the configured K/V capacity is {maximum}"
            )
            .into(),
        ));
    }
    if required <= current {
        return Ok(current);
    }
    let geometric = current.saturating_mul(2).max(required);
    Ok(round(geometric, step).min(maximum))
}

#[cfg(test)]
#[path = "allocation/tests.rs"]
mod tests;
