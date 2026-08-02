use super::super::{Array, KvContext, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowKey {
    sequence: usize,
    context: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype: u8,
    causal: bool,
    fragmented: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct BatchAttentionKey {
    pub batch: usize,
    pub sequence: usize,
    pub context_bucket: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dtype: u8,
    pub causal: bool,
    pub fragmented: bool,
}

pub(super) fn key(
    queries: &[&Array],
    contexts: &[&KvContext],
    causal: bool,
) -> Result<Option<BatchAttentionKey>> {
    if queries.len() < 2 || queries.len() != contexts.len() {
        return Ok(None);
    }
    let Some(first) = row_key(queries[0], contexts[0], causal)? else {
        return Ok(None);
    };
    for (&query, &context) in queries[1..].iter().zip(&contexts[1..]) {
        if row_key(query, context, causal)? != Some(first) {
            return Ok(None);
        }
    }
    Ok(Some(BatchAttentionKey {
        batch: queries.len(),
        sequence: first.sequence,
        context_bucket: context_bucket(first.context),
        query_heads: first.query_heads,
        kv_heads: first.kv_heads,
        head_dim: first.head_dim,
        dtype: first.dtype,
        causal,
        fragmented: first.fragmented,
    }))
}

pub(in crate::engine) fn compatible_groups(
    queries: &[&Array],
    contexts: &[&KvContext],
    causal: bool,
) -> Result<Vec<Vec<usize>>> {
    if queries.len() != contexts.len() {
        return Ok(Vec::new());
    }
    let mut groups = Vec::<(RowKey, Vec<usize>)>::new();
    let mut singles = Vec::new();
    for (index, (&query, &context)) in queries.iter().zip(contexts).enumerate() {
        let Some(key) = row_key(query, context, causal)? else {
            singles.push(vec![index]);
            continue;
        };
        if let Some((_, rows)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
            rows.push(index);
        } else {
            groups.push((key, vec![index]));
        }
    }
    Ok(groups.into_iter().map(|(_, rows)| rows).chain(singles).collect())
}

pub(super) fn fallback(key: BatchAttentionKey, paged: bool) -> super::BatchAttentionExecution {
    if paged
        && key.fragmented
        && key.sequence == 1
        && key.context_bucket >= 1_024
        && key.head_dim <= 256
        && key.head_dim.is_multiple_of(32)
        && key.kv_heads > 0
        && key.query_heads.is_multiple_of(key.kv_heads)
        && key.query_heads / key.kv_heads <= 32
    {
        super::BatchAttentionExecution::PagedRows
    } else {
        super::BatchAttentionExecution::Rows
    }
}

pub(super) const fn prefer_paged_batched(key: BatchAttentionKey, batchable: bool) -> bool {
    batchable && key.fragmented && key.sequence == 1 && key.context_bucket >= 8_192
}

fn row_key(query: &Array, context: &KvContext, causal: bool) -> Result<Option<RowKey>> {
    if context
        .paged
        .as_ref()
        .is_some_and(|paged| paged.key_scales.is_some() || paged.value_scales.is_some())
    {
        return Ok(None);
    }
    let dtype = dtype_key(query.native().dtype()?);
    let query = query.native().shape()?;
    let keys = context.keys.native().shape()?;
    let values = context.values.native().shape()?;
    let query = query.dimensions();
    let keys = keys.dimensions();
    let values = values.dimensions();
    let compatible = query.len() == 4
        && keys.len() == 4
        && values == keys
        && query[0] == 1
        && query[2] == 1
        && keys[0] == 1
        && query[2] > 0
        && keys[2] > 0
        && keys[1] > 0
        && query[1].is_multiple_of(keys[1])
        && query[3] == keys[3];
    Ok(compatible.then(|| RowKey {
        sequence: query[2],
        context: keys[2],
        query_heads: query[1],
        kv_heads: keys[1],
        head_dim: query[3],
        dtype,
        causal,
        fragmented: context.paged.as_ref().is_some_and(|paged| paged.fragmented),
    }))
}

fn context_bucket(tokens: usize) -> usize {
    tokens.max(1_024).checked_next_power_of_two().unwrap_or(usize::MAX)
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
