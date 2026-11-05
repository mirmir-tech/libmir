//! Residency of the multi-token prediction modules a speculative load adds.

use foundation::model::BackendTarget;
use models::{execution::DecoderExecutionContract, semantic::MixerSpec};
use runtime::kv::KvStorageSpec;

use crate::RuntimeConfig;

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
        .filter(|_| drafts && *target == BackendTarget::Cuda)
        .map_or(0, |speculation| speculation.draft_tokens.get())
}

/// K/V bytes per token and in total of the prediction modules' attention,
/// which a speculative load keeps over the whole sequence like a decoder
/// layer's.
pub(super) fn kv_budget(
    execution: Option<&DecoderExecutionContract>,
    config: &RuntimeConfig,
    target: &BackendTarget,
) -> (u64, u64) {
    if *target != BackendTarget::Cuda || config.scheduler.speculation.is_none() {
        return (0, 0);
    }
    execution
        .into_iter()
        .flat_map(|execution| &execution.semantic.decoder.prediction_layers)
        .filter_map(|layer| match &layer.mixer {
            MixerSpec::SoftmaxAttention(attention) => Some(
                KvStorageSpec::new(config.kv_cache, attention.key_value_heads, attention.head_dim)
                    .memory_budget(),
            ),
            MixerSpec::LinearAttention(_) | MixerSpec::StateSpace(_) => None,
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
