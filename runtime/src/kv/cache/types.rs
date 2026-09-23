use serde::{Deserialize, Serialize};

use crate::kv::{BlockTable, KvCacheDType, KvQuantMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheConfig {
    pub block_size: usize,
    pub block_count: u32,
    pub dtype: KvCacheDType,
}

/// Divisor of the page bytes that bounds retained prefix checkpoints.
///
/// A checkpoint holds recurrent state only; the pages of its prefix already
/// live in the cache, and the prefixes the cache can hold bound how many
/// checkpoints are useful.
pub const PREFIX_CHECKPOINT_PAGE_DIVISOR: usize = 4;

impl CacheConfig {
    /// Prefix checkpoint budget for `page_bytes` of K/V pages.
    #[must_use]
    pub const fn prefix_checkpoint_bytes(page_bytes: usize) -> usize {
        page_bytes / PREFIX_CHECKPOINT_PAGE_DIVISOR
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheCounters {
    pub probes: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_tokens: usize,
    pub miss_tokens: usize,
    pub evictions: usize,
    pub protected_prefix_skips: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub block_size: usize,
    pub dtype: KvCacheDType,
    pub quant_mode: KvQuantMode,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub cached_prefixes: usize,
    pub counters: CacheCounters,
}

#[derive(Debug, Clone)]
pub struct BlockAllocation {
    pub table: BlockTable,
    pub token_capacity: usize,
}

impl CacheConfig {
    #[must_use]
    pub const fn new(block_count: u32) -> Self {
        Self {
            block_size: 16,
            block_count,
            dtype: KvCacheDType::Auto,
        }
    }
}

impl CacheCounters {
    pub fn record_prefix_probe(&mut self, cached_tokens: usize, missing_tokens: usize) {
        self.probes += 1;
        self.hit_tokens += cached_tokens;
        self.miss_tokens += missing_tokens;
        if cached_tokens > 0 {
            self.hits += 1;
        }
        if missing_tokens > 0 {
            self.misses += 1;
        }
    }
}
