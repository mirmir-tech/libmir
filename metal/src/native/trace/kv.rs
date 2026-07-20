use runtime::{kv::KvCacheDType, trace::TraceKvCache};

use super::super::model::{LoadedExecution, LoadedModel};
use crate::engine::NATIVE_PAGED_ATTENTION_MIN_CONTEXT;

pub(super) fn build(
    model: &LoadedModel,
    paged_attention: bool,
    paged_attention_min_context: Option<usize>,
) -> TraceKvCache {
    let LoadedExecution::Generation(_) = &model.execution else {
        return TraceKvCache {
            dtype: KvCacheDType::BFloat16,
            quant_mode: KvCacheDType::BFloat16.quant_mode(),
            scale_granularity: KvCacheDType::BFloat16.scale_granularity(),
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
    };
    let decoder = model.info.decoder.as_ref();
    let configured_dtype = model.stream().config().kv_cache.dtype;
    let dtype = match configured_dtype {
        KvCacheDType::Auto => KvCacheDType::BFloat16,
        configured => configured,
    };
    let quantized = dtype == KvCacheDType::Int8PerTokenHead;
    let decode_attention = paged_attention_min_context.map_or_else(
        || "MLX fast scaled dot-product attention over contiguous full K/V and rotating sliding K/V".into(),
        |minimum| if quantized {
            "INT8 per-token/head pages use fused dequantization in native paged SDPA for prefill and decode".into()
        } else {
            format!(
                "canonical head-major pages start at {minimum} tokens; identity maps use a zero-copy MLX SDPA view, supported fragmented COW maps use native paged SDPA, and benchmarked identity shapes switch to native paged SDPA with persistent scratch from {NATIVE_PAGED_ATTENTION_MIN_CONTEXT} tokens"
            )
        },
    );
    let physical_page_key = if quantized {
        "layer + session; full-attention K/V uses packed INT8 pages and per-token/head FP32 scales"
    } else if paged_attention {
        "layer + session; full-attention K/V uses persistent device pages with prepared aliasing writes, sliding layers use a bounded ring"
    } else {
        "layer + session; full layers grow contiguously, sliding layers use a bounded ring"
    };
    let cached_tokens = model.resident_cached_tokens();
    TraceKvCache {
        dtype,
        quant_mode: dtype.quant_mode(),
        scale_granularity: dtype.scale_granularity(),
        decode_attention,
        block_size: paged_attention.then_some(model.stream().config().kv_cache.block_size),
        physical_page_key: physical_page_key.into(),
        prefix_cache: model.prefix_cache_enabled(),
        paged_attention,
        paged_attention_min_context,
        entry_count: decoder.map_or(0, |decoder| decoder.num_hidden_layers),
        cached_tokens,
        resident_token_slots: cached_tokens,
    }
}
