use runtime::kv::CacheConfig;

use crate::{Error, Result};

pub(super) fn max_sequence_blocks(context_tokens: usize, cache: CacheConfig) -> Result<usize> {
    if context_tokens == 0 || cache.block_size == 0 || cache.block_count == 0 {
        return Err(Error::InvalidPagedKv("invalid model sequence capacity"));
    }
    Ok(context_tokens
        .div_ceil(cache.block_size)
        .min(usize::try_from(cache.block_count)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_is_bounded_by_one_model_context() -> Result<()> {
        let cache = CacheConfig {
            block_count: 159_419,
            ..CacheConfig::new(4_096)
        };
        assert_eq!(max_sequence_blocks(262_144, cache)?, 16_384);
        Ok(())
    }

    #[test]
    fn physical_cache_can_be_the_tighter_limit() -> Result<()> {
        let cache = CacheConfig::new(4_096);
        assert_eq!(max_sequence_blocks(262_144, cache)?, 4_096);
        Ok(())
    }
}
