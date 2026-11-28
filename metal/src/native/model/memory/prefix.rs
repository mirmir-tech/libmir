use foundation::model::ModelId;

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

#[allow(clippy::too_many_arguments)]
pub(in crate::native) fn cache_prefix_snapshot(
    prefixes: &mut PrefixCache,
    model: &ModelId,
    tokens: &[u32],
    state: &SessionState,
    logits: &Array,
    block_size: Option<usize>,
    bytes: usize,
    stream: &crate::engine::Stream,
) -> Result<bool> {
    if !prefixes.enabled() {
        return Ok(false);
    }
    let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
    measure(Stage::Snapshot, || {
        let mut snapshot = state.cache.snapshot_at(state.position)?;
        snapshot.capture_tails(stream)?;
        prefixes.insert_snapshot(model, tokens, state, snapshot, logits, block_size, bytes)
    })?;
    Ok(true)
}

pub(in crate::native) fn cache_prefix_checkpoint(
    prefixes: &mut PrefixCache,
    model: &ModelId,
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
