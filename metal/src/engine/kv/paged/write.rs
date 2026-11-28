use super::{Storage, lock, pool::ArenaPages};
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
    let options = PageWriteOptions {
        sequence: keys.native().shape()?.dimensions()[2],
        offset,
        kv_heads: arena.kv_heads,
        page_capacity: arena.capacity,
        page_size: arena.page_size,
        head_dim: arena.head_dim,
    };
    let ArenaPages::Native { keys: page_keys, values: page_values } = &mut arena.pages else {
        return Err(Error::InvalidModel("native write targeted a quantized K/V arena".into()));
    };
    let [next_keys, next_values] = stream.page_write(
        [
            keys.native(),
            values.native(),
            page_keys.native(),
            page_values.native(),
            storage.table.native(),
        ],
        options,
        prepared,
    )?;
    *page_keys = Array::from_native(next_keys)?;
    *page_values = Array::from_native(next_values)?;
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
    let options = QuantizedPageWriteOptions {
        sequence: keys.native().shape()?.dimensions()[2],
        offset,
        kv_heads: arena.kv_heads,
        page_capacity: arena.capacity,
        page_size: arena.page_size,
        head_dim: arena.head_dim,
    };
    let ArenaPages::Quantized {
        keys: page_keys,
        values: page_values,
        key_scales,
        value_scales,
    } = &mut arena.pages
    else {
        return Err(Error::NullHandle("K/V key scales"));
    };
    let [next_keys, next_values, next_key_scales, next_value_scales] = stream
        .quantized_page_write(
            [
                keys.native(),
                values.native(),
                page_keys.native(),
                page_values.native(),
                key_scales.native(),
                value_scales.native(),
                storage.table.native(),
            ],
            options,
            prepared,
        )?;
    *page_keys = Array::from_native(next_keys)?;
    *page_values = Array::from_native(next_values)?;
    *key_scales = Array::from_native(next_key_scales)?;
    *value_scales = Array::from_native(next_value_scales)?;
    drop(arena);
    Ok(())
}
