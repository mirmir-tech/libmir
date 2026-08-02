use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable, CacheConfig},
};
use uuid::Uuid;

use super::CudaMoeModelSession;
use crate::{Error, Result};

impl CudaMoeModelSession {
    pub(crate) fn warmup(
        &mut self,
        session_id: Uuid,
        cache: CacheConfig,
        context_len: usize,
    ) -> Result<()> {
        let capacity = warmup_capacity(self.prefill_tokens.capacity(), cache, context_len)?;
        for tokens in canonical_geometries(capacity) {
            let prompt = vec![0; tokens];
            let table = block_table(tokens, cache)?;
            self.prefill_from_for_sampling(session_id, &prompt, 0, &table, SamplingLogits::None)?;
        }
        let mut table = block_table(capacity + 2, cache)?;
        table.set_token_len(capacity + 1);
        self.sample(SamplingLogits::None)?;
        self.decode_sampled_for_sampling(session_id, &table, SamplingLogits::None)?;
        table.set_token_len(capacity + 2);
        self.sample(SamplingLogits::None)?;
        self.decode_sampled_for_sampling(session_id, &table, SamplingLogits::None)?;
        self.sample(SamplingLogits::None)?;
        Ok(())
    }
}

fn warmup_capacity(chunk: usize, cache: CacheConfig, context: usize) -> Result<usize> {
    let cache_tokens = usize::try_from(cache.block_count)?
        .checked_mul(cache.block_size)
        .ok_or(Error::InvalidPagedKv("CUDA warmup cache capacity overflow"))?;
    let capacity = chunk.min(context.saturating_sub(2)).min(cache_tokens.saturating_sub(2));
    if capacity == 0 {
        Err(Error::InvalidPagedKv("CUDA warmup requires at least three context slots"))
    } else {
        Ok(capacity)
    }
}

fn canonical_geometries(capacity: usize) -> Vec<usize> {
    let mut geometries = Vec::new();
    let mut tokens = 1;
    while tokens < capacity {
        geometries.push(tokens);
        tokens = tokens.saturating_mul(2);
    }
    if geometries.last().copied() != Some(capacity) {
        geometries.push(capacity);
    }
    geometries
}

fn block_table(tokens: usize, cache: CacheConfig) -> Result<BlockTable> {
    let blocks = tokens.div_ceil(cache.block_size);
    if blocks > usize::try_from(cache.block_count)? {
        return Err(Error::InvalidPagedKv("CUDA warmup exceeds KV block arena"));
    }
    let mut table = BlockTable::with_block_size(cache.block_size);
    for block in 0..blocks {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(tokens);
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::canonical_geometries;

    #[test]
    fn covers_every_tail_with_prepared_geometries() {
        assert_eq!(canonical_geometries(256), [1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(canonical_geometries(192), [1, 2, 4, 8, 16, 32, 64, 128, 192]);
    }
}
