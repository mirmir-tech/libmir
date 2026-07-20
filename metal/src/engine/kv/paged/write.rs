use super::{Storage, lock};
use crate::engine::{
    Array, Error, Result, Stream,
    kernels::{
        PageWriteOptions, PreparedPageWrite, PreparedQuantizedPageWrite, QuantizedPageWriteOptions,
    },
};

pub(super) fn native(
    keys: &Array,
    values: &Array,
    offset: usize,
    stream: &Stream,
    storage: &Storage,
    prepared: &mut PreparedPageWrite,
) -> Result<()> {
    let mut arena = lock(&storage.arena)?;
    let [page_keys, page_values] = stream.page_write(
        [
            keys.native(),
            values.native(),
            arena.keys.native(),
            arena.values.native(),
            storage.table.native(),
        ],
        PageWriteOptions {
            sequence: keys.native().shape()?.dimensions()[2],
            offset,
            kv_heads: arena.kv_heads,
            page_capacity: arena.capacity,
            page_size: arena.page_size,
            head_dim: arena.head_dim,
        },
        prepared,
    )?;
    arena.keys = Array::from_native(page_keys)?;
    arena.values = Array::from_native(page_values)?;
    drop(arena);
    Ok(())
}

pub(super) fn quantized(
    keys: &Array,
    values: &Array,
    offset: usize,
    stream: &Stream,
    storage: &Storage,
    prepared: &mut PreparedQuantizedPageWrite,
) -> Result<()> {
    let mut arena = lock(&storage.arena)?;
    let key_scales = arena.key_scales.as_ref().ok_or(Error::NullHandle("K/V key scales"))?;
    let value_scales = arena.value_scales.as_ref().ok_or(Error::NullHandle("K/V value scales"))?;
    let [page_keys, page_values, next_key_scales, next_value_scales] = stream
        .quantized_page_write(
            [
                keys.native(),
                values.native(),
                arena.keys.native(),
                arena.values.native(),
                key_scales.native(),
                value_scales.native(),
                storage.table.native(),
            ],
            QuantizedPageWriteOptions {
                sequence: keys.native().shape()?.dimensions()[2],
                offset,
                kv_heads: arena.kv_heads,
                page_capacity: arena.capacity,
                page_size: arena.page_size,
                head_dim: arena.head_dim,
            },
            prepared,
        )?;
    arena.keys = Array::from_native(page_keys)?;
    arena.values = Array::from_native(page_values)?;
    arena.key_scales = Some(Array::from_native(next_key_scales)?);
    arena.value_scales = Some(Array::from_native(next_value_scales)?);
    drop(arena);
    Ok(())
}
