use std::sync::Arc;

use super::{Arena, PagedStore, Storage, lock};
use crate::engine::{Array, Error, PagedKvContext, Result, Stream};

impl PagedStore {
    pub(in crate::engine::kv) fn context(
        &self,
        tokens: usize,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        if self.format.quantized() {
            return Err(Error::InvalidModel(
                "quantized K/V must be consumed by native paged attention".into(),
            ));
        }
        let storage = self.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let arena = lock(&storage.arena)?;
        if tokens > storage.page_ids.len() * self.page_size {
            return Err(Error::InvalidModel("paged context exceeds initialized storage".into()));
        }
        let graph = stream.native().graph();
        let pages = tokens.div_ceil(self.page_size);
        let (keys, values) = gather_runs(storage, &arena, pages, graph)?;
        let shape =
            mirtal::Shape::new([1, arena.kv_heads, pages * self.page_size, arena.head_dim])?;
        let keys = graph.reshape(&keys, &shape)?;
        let values = graph.reshape(&values, &shape)?;
        let stop = [1, arena.kv_heads, tokens, arena.head_dim];
        let keys = graph.slice(&keys, &[0, 0, 0, 0], &stop)?;
        let values = graph.slice(&values, &[0, 0, 0, 0], &stop)?;
        drop(arena);
        Ok((Array::from_native(keys)?, Array::from_native(values)?))
    }

    pub(in crate::engine::kv) fn context_for_attention(
        &self,
        tokens: usize,
        stream: &Stream,
    ) -> Result<PagedKvContext> {
        let storage = self.storage.as_ref().ok_or(Error::NullHandle("paged storage"))?;
        let arena = lock(&storage.arena)?;
        let offset = mirtal::Array::from_slice(&[u32::try_from(tokens)?], [1])?;
        let mut dependencies = vec![arena.keys.native(), arena.values.native()];
        dependencies.extend(arena.key_scales.as_ref().map(Array::native));
        dependencies.extend(arena.value_scales.as_ref().map(Array::native));
        let dependency = stream.native().graph().depends(&offset, &dependencies)?;
        let key_pages = arena.keys.native().clone();
        let value_pages = arena.values.native().clone();
        let key_scales = clone_optional(arena.key_scales.as_ref())?;
        let value_scales = clone_optional(arena.value_scales.as_ref())?;
        drop(arena);
        Ok(PagedKvContext {
            key_pages: Array::from_native(key_pages)?,
            value_pages: Array::from_native(value_pages)?,
            key_scales,
            value_scales,
            page_table: Array::from_native(storage.table.native().clone())?,
            page_dependency: Array::from_native(dependency)?,
            scratch: Arc::clone(&self.attention_scratch),
            page_size: self.page_size,
            context_tokens: tokens,
            fragmented: !storage.identity,
        })
    }
}

fn gather_runs(
    storage: &Storage,
    arena: &Arena,
    pages: usize,
    graph: mirtal::Graph<'_>,
) -> Result<(mirtal::Array, mirtal::Array)> {
    let runs = physical_runs(&storage.page_ids[..pages])?;
    if let [(0, stop)] = runs.as_slice() {
        let end = [arena.kv_heads, *stop, arena.page_size, arena.head_dim];
        return Ok((
            graph.slice(arena.keys.native(), &[0, 0, 0, 0], &end)?,
            graph.slice(arena.values.native(), &[0, 0, 0, 0], &end)?,
        ));
    }
    if runs.len() == 1 {
        let (start, stop) = runs[0];
        let begin = [0, start, 0, 0];
        let end = [arena.kv_heads, stop, arena.page_size, arena.head_dim];
        return Ok((
            graph.slice(arena.keys.native(), &begin, &end)?,
            graph.slice(arena.values.native(), &begin, &end)?,
        ));
    }
    let ids = graph.slice(storage.table.native(), &[0], &[pages])?;
    Ok((
        graph.take(arena.keys.native(), &ids, 1)?,
        graph.take(arena.values.native(), &ids, 1)?,
    ))
}

fn physical_runs(page_ids: &[u32]) -> Result<Vec<(usize, usize)>> {
    let mut runs = Vec::new();
    for page in page_ids {
        let page = usize::try_from(*page)?;
        match runs.last_mut() {
            Some((_, stop)) if *stop == page => *stop += 1,
            _ => runs.push((page, page + 1)),
        }
    }
    Ok(runs)
}

fn clone_optional(array: Option<&Array>) -> Result<Option<Array>> {
    array.map(|array| Array::from_native(array.native().clone())).transpose()
}

#[cfg(test)]
mod tests {
    use super::physical_runs;

    #[test]
    fn preserves_logical_order_while_coalescing_physical_pages() -> crate::engine::Result<()> {
        assert_eq!(physical_runs(&[32, 33, 34, 2, 3, 19])?, [(32, 35), (2, 4), (19, 20)]);
        Ok(())
    }
}
