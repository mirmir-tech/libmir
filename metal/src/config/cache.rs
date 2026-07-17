#[derive(Debug, Clone)]
pub struct MetalCacheConfig {
    pub prefix_cache_entries: usize,
    pub prefill_step: Option<usize>,
    pub kv_reserve_tokens: usize,
    pub paged_attention_min_context: usize,
    pub force_native_paged_attention: bool,
}

impl Default for MetalCacheConfig {
    fn default() -> Self {
        Self {
            prefix_cache_entries: 16,
            prefill_step: None,
            kv_reserve_tokens: 256,
            paged_attention_min_context: 128,
            force_native_paged_attention: false,
        }
    }
}
