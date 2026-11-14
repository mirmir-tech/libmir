use super::{Array, KvContext, Result, Stream};

pub(super) fn attention(
    context: &KvContext,
    stream: &Stream,
    refresh_fragmented: bool,
) -> Result<(Array, Array)> {
    let Some(paged) = context.paged.as_ref().filter(|paged| refresh_fragmented && paged.fragmented)
    else {
        return Ok((
            Array::from_native(context.keys.native().clone())?,
            Array::from_native(context.values.native().clone())?,
        ));
    };
    let graph = stream.native().graph();
    let logical_pages = paged.context_tokens.div_ceil(paged.page_size);
    let ids = graph.slice(paged.page_table.native(), &[0], &[logical_pages])?;
    let keys = graph.take(paged.key_pages.native(), &ids, 1)?;
    let values = graph.take(paged.value_pages.native(), &ids, 1)?;
    let dimensions = keys.shape()?.dimensions().to_vec();
    let shape =
        mirtal::Shape::new([1, dimensions[0], logical_pages * paged.page_size, dimensions[3]])?;
    let stop = [1, dimensions[0], paged.context_tokens, dimensions[3]];
    Ok((
        Array::from_native(graph.slice(&graph.reshape(&keys, &shape)?, &[0, 0, 0, 0], &stop)?)?,
        Array::from_native(graph.slice(&graph.reshape(&values, &shape)?, &[0, 0, 0, 0], &stop)?)?,
    ))
}
