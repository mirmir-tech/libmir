//! Arenas sized to the blocks in use. A layer arena is indexed directly by
//! runtime `BlockId`s, but holds pages only up to the highest one bound so
//! far, growing geometrically up to the configured cache: allocated whole
//! up front, an arena made every head's rows the whole cache apart and
//! reserved the cache's memory before any request.

use super::{Arena, ArenaPages};
use crate::engine::{Array, Error, Result, Stream};

/// The fewest pages an arena grows by.
const GROWTH_PAGES: usize = 64;

/// The pages an arena holding `current` grows to so that it holds
/// `required`, at most `maximum`.
pub(in crate::engine::kv::paged) fn target(
    current: usize,
    required: usize,
    maximum: usize,
) -> Result<usize> {
    if required > maximum {
        return Err(Error::InvalidModel(format!(
            "paged arena requires {required} pages but the configured K/V capacity is {maximum}"
        )));
    }
    if required <= current {
        return Ok(current);
    }
    let geometric = current.saturating_mul(2).max(required);
    Ok(geometric.div_ceil(GROWTH_PAGES).saturating_mul(GROWTH_PAGES).min(maximum))
}

impl Arena {
    /// Grows the arena to hold `required` pages, keeping every page written.
    pub(in crate::engine::kv::paged) fn grow(
        &mut self,
        required: usize,
        maximum: usize,
        stream: &Stream,
    ) -> Result<()> {
        let capacity = target(self.capacity, required, maximum)?;
        if capacity == self.capacity {
            return Ok(());
        }
        let extra = capacity - self.capacity;
        let graph = stream.native().graph();
        let grow = |array: &Array, fill: f32| -> Result<Array> {
            let mut shape = array.native().shape()?.dimensions().to_vec();
            shape[1] = extra;
            let padding = graph.full(&mirtal::Shape::new(shape)?, fill, array.native().dtype()?)?;
            Array::from_native(graph.concatenate(&[array.native(), &padding], 1)?)
        };
        self.pages = match &self.pages {
            ArenaPages::Native { keys, values } => ArenaPages::Native {
                keys: grow(keys, 0.0)?,
                values: grow(values, 0.0)?,
            },
            ArenaPages::Quantized { keys, values, key_scales, value_scales } => {
                ArenaPages::Quantized {
                    keys: grow(keys, 0.0)?,
                    values: grow(values, 0.0)?,
                    key_scales: grow(key_scales, 1.0)?,
                    value_scales: grow(value_scales, 1.0)?,
                }
            },
        };
        #[cfg(test)]
        if self.unbound_refs.len() < capacity {
            self.unbound_refs.resize(capacity, 0);
        }
        self.capacity = capacity;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::target;

    /// An arena grows to what its blocks need, at least doubling and in
    /// steps of 64 pages, never past the configured cache.
    #[test]
    fn an_arena_grows_to_the_blocks_in_use_within_the_cache() -> crate::engine::Result<()> {
        assert_eq!(target(0, 13, 12_133)?, 64);
        assert_eq!(target(64, 64, 12_133)?, 64);
        assert_eq!(target(64, 65, 12_133)?, 128);
        assert_eq!(target(128, 1000, 12_133)?, 1024);
        assert_eq!(target(8192, 9000, 12_133)?, 12_133);
        assert!(target(0, 12_134, 12_133).is_err());
        Ok(())
    }
}
