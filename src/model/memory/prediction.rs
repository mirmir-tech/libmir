//! Residency of the multi-token prediction modules a speculative load adds.

use architecture::LayerCacheEntry;
use foundation::model::BackendTarget;
use models::execution::DecoderExecutionContract;

use crate::RuntimeConfig;

/// Drafts a backend verifies per step of `requested`: the accelerators all
/// of them.
pub const fn backend_drafts(target: BackendTarget, requested: usize) -> usize {
    match target {
        BackendTarget::Cuda | BackendTarget::Metal => requested,
    }
}

/// Recurrent-state snapshots a drafting session keeps: one per draft, to
/// roll back to the accepted prefix.
pub(super) fn rollback_snapshots(
    execution: Option<&DecoderExecutionContract>,
    config: &RuntimeConfig,
    target: BackendTarget,
) -> usize {
    let drafts =
        execution.is_some_and(|execution| !execution.semantic.decoder.prediction_layers.is_empty());
    config
        .scheduler
        .speculation
        .filter(|_| drafts)
        .map_or(0, |speculation| backend_drafts(target, speculation.draft_tokens.get()))
}

/// K/V bytes per token and in total of the prediction modules' attention,
/// driven from the admitted cache layout.
pub(super) fn kv_budget(
    prediction_layers: &[LayerCacheEntry],
    config: &RuntimeConfig,
    target: BackendTarget,
) -> (u64, u64) {
    let Some(speculation) = config.scheduler.speculation else {
        return (0, 0);
    };
    if backend_drafts(target, speculation.draft_tokens.get()) == 0 {
        return (0, 0);
    }
    super::kv_budget(prediction_layers, config)
}
