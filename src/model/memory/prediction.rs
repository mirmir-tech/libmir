//! Residency of the multi-token prediction modules a speculative load adds.

use foundation::model::BackendTarget;
use models::execution::DecoderExecutionContract;

use crate::RuntimeConfig;

/// Drafts a backend verifies per step of `requested`: the accelerators all
/// of them; other backends do not speculate.
pub fn backend_drafts(target: &BackendTarget, requested: usize) -> Option<usize> {
    match target {
        BackendTarget::Cuda | BackendTarget::Metal => Some(requested),
        BackendTarget::CpuReference => None,
    }
}

/// Recurrent-state snapshots a drafting session keeps: one per draft, to
/// roll back to the accepted prefix.
pub(super) fn rollback_snapshots(
    execution: Option<&DecoderExecutionContract>,
    config: &RuntimeConfig,
    target: &BackendTarget,
) -> usize {
    let drafts =
        execution.is_some_and(|execution| !execution.semantic.decoder.prediction_layers.is_empty());
    config
        .scheduler
        .speculation
        .filter(|_| drafts)
        .and_then(|speculation| backend_drafts(target, speculation.draft_tokens.get()))
        .unwrap_or(0)
}

/// K/V bytes per token and in total of the prediction modules' attention,
/// which a speculative load keeps over the whole sequence like a decoder
/// layer's.
pub(super) fn kv_budget(
    execution: Option<&DecoderExecutionContract>,
    config: &RuntimeConfig,
    target: &BackendTarget,
) -> (u64, u64) {
    if config
        .scheduler
        .speculation
        .and_then(|speculation| backend_drafts(target, speculation.draft_tokens.get()))
        .is_none()
    {
        return (0, 0);
    }
    execution
        .into_iter()
        .flat_map(|execution| &execution.semantic.decoder.prediction_layers)
        .filter_map(|layer| {
            let shape = layer.cache_shape()?;
            Some(super::layer_storage(shape, config.kv_cache).memory_budget())
        })
        .fold((0_u64, 0_u64), |(per_token, total), budget| {
            let bytes = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
            (
                per_token
                    .saturating_add(bytes(budget.data_bytes_per_token))
                    .saturating_add(bytes(budget.scale_bytes_per_token)),
                total.saturating_add(bytes(budget.total_bytes)),
            )
        })
}
