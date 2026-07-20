use std::sync::{Arc, Mutex};

use super::{Arena, PagedStore, Storage, lock};
use crate::engine::{Array, Error, KvPageFormat, Result, Stream};

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
    if needed > storage.table_capacity {
        storage.table_capacity = round(needed + store.allocation_step, store.allocation_step);
        storage.table = page_table(&storage.page_ids, storage.table_capacity)?;
    }
    while storage.page_ids.len() < needed {
        ensure_free(&mut arena, store.allocation_step, stream)?;
        let logical = storage.page_ids.len();
        let physical = allocate(&mut arena)?;
        storage.page_ids.push(physical);
        if usize::try_from(physical)? != logical {
            map(storage, logical, physical, stream)?;
        }
    }
    let first = offset / store.page_size;
    let last = (offset + sequence - 1) / store.page_size;
    for logical in first..=last {
        let source = usize::try_from(storage.page_ids[logical])?;
        if arena.references[source] == 1 {
            continue;
        }
        ensure_free(&mut arena, store.allocation_step, stream)?;
        let target = allocate(&mut arena)?;
        copy_page(&mut arena, source, usize::try_from(target)?, stream)?;
        arena.references[source] -= 1;
        map(storage, logical, target, stream)?;
    }
    drop(arena);
    Ok(())
}

fn create(store: &PagedStore, keys: &Array, needed: usize, stream: &Stream) -> Result<Storage> {
    let dimensions = keys.native().shape()?.dimensions().to_vec();
    let capacity = round(needed.max(store.reserve_pages), store.allocation_step);
    let packed_head_dim = store.format.packed_words(dimensions[3])?;
    let shape = mirtal::Shape::new([dimensions[1], capacity, store.page_size, packed_head_dim])?;
    let graph = stream.native().graph();
    let dtype = match store.format {
        KvPageFormat::Native => keys.native().dtype()?,
        KvPageFormat::Int8PerTokenHead => mirtal::DType::Uint32,
    };
    let scale_shape = mirtal::Shape::new([dimensions[1], capacity, store.page_size])?;
    let (key_scales, value_scales) = if store.format.quantized() {
        (
            Some(Array::from_native(graph.full(&scale_shape, 1.0, mirtal::DType::Float32)?)?),
            Some(Array::from_native(graph.full(&scale_shape, 1.0, mirtal::DType::Float32)?)?),
        )
    } else {
        (None, None)
    };
    let arena = Arena {
        keys: Array::from_native(graph.full(&shape, 0.0, dtype)?)?,
        values: Array::from_native(graph.full(&shape, 0.0, dtype)?)?,
        key_scales,
        value_scales,
        capacity,
        page_size: store.page_size,
        kv_heads: dimensions[1],
        head_dim: dimensions[3],
        references: vec![0; capacity],
    };
    Ok(Storage {
        arena: Arc::new(Mutex::new(arena)),
        table: page_table(&[], capacity)?,
        page_ids: Vec::new(),
        table_capacity: capacity,
        identity: true,
    })
}

fn ensure_free(arena: &mut Arena, step: usize, stream: &Stream) -> Result<()> {
    if arena.references.contains(&0) {
        return Ok(());
    }
    let capacity = (arena.capacity * 2).max(arena.capacity + step);
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

fn allocate(arena: &mut Arena) -> Result<u32> {
    let index = arena
        .references
        .iter()
        .position(|count| *count == 0)
        .ok_or_else(|| Error::InvalidModel("paged arena has no free page".into()))?;
    arena.references[index] = 1;
    Ok(u32::try_from(index)?)
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

fn map(storage: &mut Storage, logical: usize, physical: u32, stream: &Stream) -> Result<()> {
    storage.page_ids[logical] = physical;
    let update = mirtal::Array::from_slice(&[physical], [1])?;
    storage.table = Array::from_native(stream.native().graph().slice_update(
        storage.table.native(),
        &update,
        &[logical],
        &[logical + 1],
    )?)?;
    storage.identity = storage
        .page_ids
        .iter()
        .enumerate()
        .all(|(index, page)| usize::try_from(*page) == Ok(index));
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
