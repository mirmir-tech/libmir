use super::{KvCache, paged};
use crate::engine::{Array, Error, Result, Stream};

impl KvCache {
    /// Appends one token to every row with a single page write when all rows
    /// are native pages in one arena. Returns `false`, untouched, otherwise.
    /// Callers attend elsewhere: no view of the pages is built.
    pub(crate) fn write_rows(
        caches: &mut [&mut Self],
        [keys, values]: [&Array; 2],
        stream: &Stream,
    ) -> Result<bool> {
        let eligible = caches.len() > 1
            && caches.iter().all(|cache| {
                cache.max_context.is_none()
                    && cache
                        .pages
                        .as_ref()
                        .is_some_and(|pages| pages.active() && !pages.quantized())
            });
        if !eligible {
            return Ok(false);
        }
        let shape = keys.shape()?;
        let &[_, heads, 1, dim] = shape.as_slice() else {
            return Ok(false);
        };
        let [heads, dim] = [usize::try_from(heads)?, usize::try_from(dim)?];
        for (index, cache) in caches.iter().enumerate() {
            let key = keys.slice(&[index, 0, 0, 0], &[index + 1, heads, 1, dim], stream)?;
            let value = values.slice(&[index, 0, 0, 0], &[index + 1, heads, 1, dim], stream)?;
            cache.validate_update(&key, &value)?;
        }
        let offsets = caches.iter().map(|cache| cache.offset).collect::<Vec<_>>();
        let mut stores = caches
            .iter_mut()
            .map(|cache| cache.pages.as_mut().ok_or(Error::NullHandle("paged K/V store")))
            .collect::<Result<Vec<_>>>()?;
        paged::write_rows(&mut stores, &offsets, [keys, values], stream)?;
        for cache in caches.iter_mut() {
            cache.offset += 1;
            cache.history = None;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
