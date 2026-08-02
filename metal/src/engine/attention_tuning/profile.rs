use super::super::{
    Error, Result,
    kernels::{PagedExecution, partial_blocks},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AttentionKey {
    pub context_bucket: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub dtype: u8,
}

pub(super) fn key(
    [queries, key_pages, _, _, _]: [&mirtal::Array; 5],
    page_size: usize,
    context_tokens: usize,
) -> Result<AttentionKey> {
    let query = queries.shape()?;
    let keys = key_pages.shape()?;
    let query = query.dimensions();
    let keys = keys.dimensions();
    Ok(AttentionKey {
        context_bucket: context_bucket(context_tokens),
        query_heads: *query.get(1).ok_or(Error::ShapeOverflow)?,
        kv_heads: *keys.first().ok_or(Error::ShapeOverflow)?,
        head_dim: *query.get(3).ok_or(Error::ShapeOverflow)?,
        page_size,
        dtype: dtype_key(queries.dtype()?),
    })
}

const fn dtype_key(dtype: mirtal::DType) -> u8 {
    match dtype {
        mirtal::DType::Bool => 0,
        mirtal::DType::Uint8 => 1,
        mirtal::DType::Uint32 => 2,
        mirtal::DType::Int32 => 3,
        mirtal::DType::Float16 => 4,
        mirtal::DType::Bfloat16 => 5,
        mirtal::DType::Float32 => 6,
    }
}

pub(super) fn context_bucket(tokens: usize) -> usize {
    tokens.max(1_024).checked_next_power_of_two().unwrap_or(usize::MAX)
}

pub(super) fn fallback(key: AttentionKey, context_tokens: usize) -> PagedExecution {
    PagedExecution::TwoPass {
        blocks: partial_blocks(context_tokens, key.query_heads / key.kv_heads.max(1), key.kv_heads),
        reduction_groups: 32,
    }
}
