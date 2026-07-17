use runtime::{kv::KvCacheDType, trace::TraceKvCache};

use super::super::model::LoadedModel;
use crate::engine::NATIVE_PAGED_ATTENTION_MIN_CONTEXT;

pub(super) fn build(
    model: &LoadedModel,
    paged_attention: bool,
    paged_attention_min_context: Option<usize>,
) -> TraceKvCache {
    let decoder = &model.info.decoder;
    let decode_attention = paged_attention_min_context.map_or_else(
        || "MLX fast scaled dot-product attention over contiguous full K/V and rotating sliding K/V".into(),
        |minimum| format!(
            "canonical head-major pages start at {minimum} tokens; identity maps use a zero-copy MLX SDPA view, supported fragmented COW maps use native paged SDPA, and benchmarked identity shapes switch to native paged SDPA with persistent scratch from {NATIVE_PAGED_ATTENTION_MIN_CONTEXT} tokens"
        ),
    );
    let physical_page_key = if paged_attention {
        "layer + session; full-attention K/V uses persistent device pages with prepared aliasing writes, sliding layers use a bounded ring"
    } else {
        "layer + session; full layers grow contiguously, sliding layers use a bounded ring"
    };
    let cached_tokens = model.resident_cached_tokens();
    TraceKvCache {
        dtype: KvCacheDType::BFloat16,
        quant_mode: KvCacheDType::BFloat16.quant_mode(),
        scale_granularity: KvCacheDType::BFloat16.scale_granularity(),
        decode_attention,
        block_size: paged_attention.then_some(16),
        physical_page_key: physical_page_key.into(),
        prefix_cache: model.prefix_cache_enabled(),
        paged_attention,
        paged_attention_min_context,
        entry_count: decoder.num_hidden_layers,
        cached_tokens,
        resident_token_slots: cached_tokens,
    }
}
