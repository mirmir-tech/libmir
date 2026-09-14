use super::LoadedModel;
use crate::{
    engine::Array,
    native::{
        error::Result,
        prefill::diagnostics::{Stage, measure},
        prefix::PrefixCache,
        session::SessionState,
    },
};

pub(in crate::native) fn cache_prefix_snapshot(
    prefixes: &mut PrefixCache,
    model: &str,
    tokens: &[u32],
    state: &SessionState,
    logits: &Array,
    block_size: Option<usize>,
    bytes: usize,
) -> Result<bool> {
    if !prefixes.enabled() {
        return Ok(false);
    }
    let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
    measure(Stage::Snapshot, || {
        prefixes.insert(model, tokens, state, logits, block_size, bytes)
    })?;
    Ok(true)
}

pub(in crate::native) fn cache_prefix_checkpoint(
    prefixes: &mut PrefixCache,
    model: &str,
    tokens: &[u32],
    state: &SessionState,
    block_size: usize,
    bytes: usize,
) -> Result<bool> {
    if !prefixes.enabled() {
        return Ok(false);
    }
    let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
    measure(Stage::Checkpoint, || {
        prefixes.insert_checkpoint(model, tokens, state, block_size, bytes)
    })?;
    Ok(true)
}
