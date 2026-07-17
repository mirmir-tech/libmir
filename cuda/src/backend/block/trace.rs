use super::DecodeMoeBlockConfig;

pub(super) fn prepared(config: DecodeMoeBlockConfig) {
    tracing::debug!(
        backend = "cuda",
        layer = config.attention.layer,
        hidden = config.attention.hidden_size,
        attention_heads = config.attention.query_heads,
        kv_heads = config.attention.cache.kv_heads,
        dense_intermediate = config.dense_intermediate,
        expert_intermediate = config.expert_intermediate,
        experts = config.experts,
        top_k = config.top_k,
        cache_dtype = %config.attention.cache.cache.dtype,
        cache_blocks = config.attention.cache.cache.block_count,
        "prepared routed MoE decode block"
    );
}
