use models::execution::TaskExecutionPlan;
use runtime::{kv::CacheConfig, trace::TraceKvCache};

use super::LoadedModel;

pub(super) fn build(model: &LoadedModel, cache: CacheConfig, sessions: usize) -> TraceKvCache {
    if !matches!(&model.task_plan, TaskExecutionPlan::Generation { .. }) {
        return TraceKvCache {
            dtype: cache.dtype,
            quant_mode: cache.dtype.quant_mode(),
            scale_granularity: cache.dtype.scale_granularity(),
            decode_attention: "not applicable to non-generative task execution".into(),
            block_size: None,
            physical_page_key: "not applicable".into(),
            prefix_cache: false,
            paged_attention: false,
            paged_attention_min_context: None,
            entry_count: 0,
            cached_tokens: 0,
            resident_token_slots: 0,
        };
    }
    TraceKvCache {
        dtype: cache.dtype,
        quant_mode: cache.dtype.quant_mode(),
        scale_granularity: cache.dtype.scale_granularity(),
        decode_attention: "native split-KV paged CUDA attention".into(),
        block_size: Some(cache.block_size),
        physical_page_key: "runtime BlockId".into(),
        prefix_cache: true,
        paged_attention: true,
        paged_attention_min_context: Some(1),
        entry_count: sessions,
        cached_tokens: 0,
        resident_token_slots: cache.block_size * cache.block_count as usize,
    }
}
