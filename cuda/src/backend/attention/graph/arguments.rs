use super::{AttentionArguments, KvArguments, QkvPostprocessArguments, Resources};
use crate::{
    Result,
    kernels::{MergeAttentionArguments, SplitAttentionArguments},
};

pub(super) fn qkv_arguments(resources: &mut Resources) -> QkvPostprocessArguments<'_> {
    let geometry = resources.geometry;
    let dynamic = resources.dynamic;
    let weights = resources.weights.borrow();
    let scratch = &mut resources.attention.scratch;
    let (query, key, value) = if geometry.separate_qkv {
        (&scratch.qkv_separate[0], &scratch.qkv_separate[1], &scratch.qkv_separate[2])
    } else {
        (&scratch.qkv, &scratch.qkv, &scratch.qkv)
    };
    (
        query,
        key,
        value,
        weights.query_norm,
        weights.key_norm,
        &mut scratch.query_rope,
        &mut scratch.key_rope,
        &mut scratch.value_norm,
        1,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.value_head_dim,
        geometry.rotary_dim,
        geometry.pairing_dim,
        dynamic.position,
        geometry.theta,
        geometry.epsilon,
        u32::from(geometry.separate_qkv),
        u32::from(geometry.normalization.query),
        u32::from(geometry.normalization.key),
        u32::from(geometry.normalization.value),
    )
}

pub(super) fn kv_arguments(resources: &mut Resources) -> KvArguments<'_> {
    let geometry = resources.geometry;
    let dynamic = resources.dynamic;
    let attention = &mut resources.attention;
    let (key_pages, value_pages) = attention.cache.pages_mut();
    (
        &attention.scratch.key_rope,
        &attention.scratch.value_norm,
        key_pages,
        value_pages,
        dynamic.local_start,
        1,
        dynamic.physical_block,
        dynamic.page_start,
        geometry.block_size_abi,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.value_head_dim,
    )
}

pub(super) fn attention_arguments(resources: &mut Resources) -> AttentionArguments<'_> {
    let geometry = resources.geometry;
    let dynamic = resources.dynamic;
    let attention = &mut resources.attention;
    (
        &attention.scratch.query_rope,
        attention.cache.key_pages(),
        attention.cache.value_pages(),
        attention.attention.table_device(),
        &mut attention.scratch.attention,
        dynamic.token_count,
        dynamic.block_count,
        geometry.block_size_abi,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.value_head_dim,
        geometry.window,
        geometry.scale,
        geometry.split_threshold,
    )
}

pub(super) fn split_attention_arguments(
    resources: &mut Resources,
) -> Result<SplitAttentionArguments<'_>> {
    let geometry = resources.geometry;
    let dynamic = resources.dynamic;
    let attention = &mut resources.attention;
    attention.attention.captured_split_arguments(
        &attention.scratch.query_rope,
        &attention.cache,
        dynamic.token_count,
        dynamic.block_count,
        geometry.window,
        geometry.scale,
    )
}

pub(super) fn merge_attention_arguments(
    resources: &mut Resources,
) -> Result<MergeAttentionArguments<'_>> {
    let geometry = resources.geometry;
    let dynamic = resources.dynamic;
    let attention = &mut resources.attention;
    attention.attention.captured_merge_arguments(
        &mut attention.scratch.attention,
        dynamic.token_count,
        geometry.window,
    )
}
