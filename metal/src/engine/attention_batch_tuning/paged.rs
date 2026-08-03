use super::{Array, KvContext, Result, Stream};

pub(super) fn batchable(contexts: &[&KvContext]) -> bool {
    let Some(first) = contexts.first().and_then(|context| context.paged.as_ref()) else {
        return false;
    };
    if first.key_scales.is_some() || first.value_scales.is_some() {
        return false;
    }
    let Ok(key_dtype) = first.key_pages.native().dtype() else {
        return false;
    };
    contexts.iter().skip(1).all(|context| {
        context.paged.as_ref().is_some_and(|paged| {
            paged.key_scales.is_none()
                && paged.value_scales.is_none()
                && paged.page_size == first.page_size
                && paged.key_pages.native().dtype().is_ok_and(|dtype| dtype == key_dtype)
                && paged.value_pages.native().dtype().is_ok_and(|dtype| dtype == key_dtype)
        })
    })
}

pub(super) fn rows(
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    stream: &Stream,
) -> Result<Vec<Array>> {
    queries
        .iter()
        .zip(contexts)
        .map(|(query, context)| {
            let paged = context.paged.as_ref().ok_or_else(|| {
                super::super::Error::InvalidModel("paged context is missing".into())
            })?;
            query.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                scale,
                stream,
            )
        })
        .collect()
}

pub(super) fn batched(
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    stream: &Stream,
) -> Result<Vec<Array>> {
    if !batchable(contexts) {
        return Err(super::super::Error::InvalidModel(
            "paged contexts are not batch compatible".into(),
        ));
    }
    let mut chunks = Vec::new();
    for (queries, contexts) in queries
        .chunks(super::super::kernels::BATCHED_PAGED_ROWS)
        .zip(contexts.chunks(super::super::kernels::BATCHED_PAGED_ROWS))
    {
        chunks.push(chunk(queries, contexts, scale, stream)?);
    }
    let output = if chunks.len() == 1 {
        chunks.pop().ok_or(super::super::Error::ShapeOverflow)?
    } else {
        let refs = chunks.iter().collect::<Vec<_>>();
        Array::concatenate(&refs, 0, stream)?
    };
    let shape = output.shape()?;
    (0..shape[0])
        .map(|row| {
            output.slice(
                &[usize::try_from(row)?, 0, 0, 0],
                &[
                    usize::try_from(row + 1)?,
                    usize::try_from(shape[1])?,
                    usize::try_from(shape[2])?,
                    usize::try_from(shape[3])?,
                ],
                stream,
            )
        })
        .collect()
}

fn chunk(
    queries: &[&Array],
    contexts: &[&KvContext],
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let first = paged(contexts[0])?;
    let queries = Array::concatenate(queries, 0, stream)?;
    let context_tokens = contexts
        .iter()
        .filter_map(|context| context.paged.as_ref().map(|paged| paged.context_tokens))
        .max()
        .ok_or(super::super::Error::ShapeOverflow)?;
    let pages = context_tokens.div_ceil(first.page_size);
    let mut tables = Vec::with_capacity(contexts.len());
    let mut dependencies = Vec::with_capacity(contexts.len());
    let mut capacities = Vec::with_capacity(contexts.len());
    let mut keys = [first.key_pages.native(); 8];
    let mut values = [first.value_pages.native(); 8];
    for (row, context) in contexts.iter().enumerate() {
        let context = paged(context)?;
        tables.push(context.page_table.slice(&[0], &[pages], stream)?);
        dependencies.push(&context.page_dependency);
        capacities.push(u32::try_from(context.key_pages.native().shape()?.dimensions()[1])?);
        keys[row] = context.key_pages.native();
        values[row] = context.value_pages.native();
    }
    let table_refs = tables.iter().collect::<Vec<_>>();
    let tables = Array::concatenate(&table_refs, 0, stream)?;
    let dependencies = Array::concatenate(&dependencies, 0, stream)?;
    let capacities = Array::from_u32(&capacities, &[i32::try_from(capacities.len())?])?;
    let output = stream.batched_paged_attention(
        [
            queries.native(),
            keys[0],
            keys[1],
            keys[2],
            keys[3],
            keys[4],
            keys[5],
            keys[6],
            keys[7],
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            tables.native(),
            dependencies.native(),
            capacities.native(),
        ],
        first.page_size,
        context_tokens,
        scale,
    )?;
    Array::from_native(output)
}

fn paged(context: &KvContext) -> Result<&super::super::PagedKvContext> {
    context
        .paged
        .as_ref()
        .ok_or_else(|| super::super::Error::InvalidModel("paged context is missing".into()))
}
