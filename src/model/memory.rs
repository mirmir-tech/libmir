use models::layout::AttentionLayerType;
use runtime::kv::KvStorageSpec;

use super::ModelDescriptor;
use crate::{ModelMemoryEstimate, RuntimeConfig};

const MIB: u64 = 1024 * 1024;

impl ModelDescriptor {
    #[must_use]
    /// Estimates weights, K/V cache, and workspace bytes without loading the
    /// model.
    pub fn memory_estimate(&self, config: &RuntimeConfig) -> ModelMemoryEstimate {
        let weight_bytes = self.layout.weights.iter().map(|weight| weight.bytes).sum::<u64>();
        let cache_capacity_tokens = u64::from(config.kv_cache.block_count)
            .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX));
        let (kv_bytes_per_token, kv_cache_bytes) = self
            .decoder
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, layer)| **layer != AttentionLayerType::Linear)
            .map(|(index, _)| {
                KvStorageSpec::new(
                    config.kv_cache,
                    self.decoder.layer_key_value_heads(index),
                    self.decoder.layer_head_dim(index),
                )
                .memory_budget()
            })
            .fold((0_u64, 0_u64), |(per_token, total), budget| {
                let data = u64::try_from(budget.data_bytes_per_token).unwrap_or(u64::MAX);
                let scales = u64::try_from(budget.scale_bytes_per_token).unwrap_or(u64::MAX);
                (
                    per_token.saturating_add(data.saturating_add(scales)),
                    total.saturating_add(u64::try_from(budget.total_bytes).unwrap_or(u64::MAX)),
                )
            });
        let workspace_bytes = workspace(weight_bytes);
        let required_bytes =
            weight_bytes.saturating_add(kv_cache_bytes).saturating_add(workspace_bytes);
        ModelMemoryEstimate {
            weight_bytes,
            kv_cache_bytes,
            workspace_bytes,
            required_bytes,
            kv_bytes_per_token,
            cache_capacity_tokens,
            model_context_tokens: u64::try_from(self.metadata.context_len).unwrap_or(u64::MAX),
        }
    }
}

const fn workspace(weights: u64) -> u64 {
    let proportional = weights / 10;
    if proportional > 512 * MIB {
        proportional
    } else {
        512 * MIB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_at_least_half_a_gibibyte_for_runtime_workspace() {
        assert_eq!(workspace(128 * MIB), 512 * MIB);
        assert_eq!(workspace(10 * 1024 * MIB), 1024 * MIB);
    }
}
