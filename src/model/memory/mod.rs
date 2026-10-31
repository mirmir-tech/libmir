use foundation::model::BackendTarget;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{AttentionLayerType, DecoderConfig},
    weights::{
        BlockActivationMode, BlockFormat, LayerTensorRole, LogicalTensorRole, TensorStorage,
    },
};
use runtime::kv::{CacheConfig, KvStorageSpec};

use super::ModelDescriptor;
use crate::{ModelMemoryEstimate, RuntimeConfig};

mod resident;
mod session;

pub use resident::UnloadedTensors;
pub(super) use resident::unloaded;

const CUDA_SLIDING_PHYSICAL_SLOTS_PER_SESSION: usize = 4;

const MIB: u64 = 1024 * 1024;

impl ModelDescriptor {
    #[must_use]
    /// Estimates weights, K/V cache, and workspace bytes without loading the
    /// model using backend-neutral physical K/V residency.
    pub fn memory_estimate(&self, config: &RuntimeConfig) -> ModelMemoryEstimate {
        self.memory_estimate_for(config, &BackendTarget::Metal)
    }

    #[must_use]
    /// Estimates residency including physical storage specific to `target`.
    pub fn memory_estimate_for(
        &self,
        config: &RuntimeConfig,
        target: &BackendTarget,
    ) -> ModelMemoryEstimate {
        let weight_bytes = resident::weight_bytes(&self.layout, self.unloaded.as_ref());
        let generation_decoder = match &self.task_plan {
            TaskExecutionPlan::Generation { decoder }
            | TaskExecutionPlan::CausalScoring { decoder, .. } => Some(decoder),
            TaskExecutionPlan::Embedding { .. } | TaskExecutionPlan::SequenceScoring { .. } => None,
        };
        let cache_capacity_tokens = generation_decoder.map_or(0, |_| {
            u64::from(config.kv_cache.block_count)
                .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX))
        });
        let (kv_bytes_per_token, kv_cache_bytes) =
            generation_decoder.map_or((0, 0), |decoder| kv_budget(decoder, config, target));
        let workspace_bytes =
            workspace(weight_bytes).saturating_add(if *target == BackendTarget::Cuda {
                cuda_persistent_workspace(self.execution.as_ref())
            } else {
                0
            });
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
            session_state_bytes: generation_decoder.map_or(0, session::state_bytes),
        }
    }
}

fn kv_budget(
    decoder: &DecoderConfig,
    config: &RuntimeConfig,
    target: &BackendTarget,
) -> (u64, u64) {
    decoder
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, layer)| {
            matches!(layer, AttentionLayerType::Full | AttentionLayerType::Sliding)
        })
        .fold((0_u64, 0_u64), |(per_token, total), (index, layer)| {
            let cache = if *layer == AttentionLayerType::Sliding && bounded_sliding(decoder) {
                sliding_cache(decoder, config, target)
            } else {
                config.kv_cache
            };
            let budget = KvStorageSpec::new(
                cache,
                decoder.layer_key_value_heads(index),
                decoder.layer_head_dim(index),
            )
            .memory_budget();
            let data = u64::try_from(budget.data_bytes_per_token).unwrap_or(u64::MAX);
            let scales = u64::try_from(budget.scale_bytes_per_token).unwrap_or(u64::MAX);
            let growing = if *layer != AttentionLayerType::Sliding || !bounded_sliding(decoder) {
                data.saturating_add(scales)
            } else {
                0
            };
            (
                per_token.saturating_add(growing),
                total.saturating_add(u64::try_from(budget.total_bytes).unwrap_or(u64::MAX)),
            )
        })
}

fn sliding_cache(
    decoder: &DecoderConfig,
    config: &RuntimeConfig,
    target: &BackendTarget,
) -> CacheConfig {
    let window = decoder.sliding_window.unwrap_or(config.kv_cache.block_size);
    let ring_blocks = window
        .saturating_add(config.kv_cache.block_size.saturating_sub(1))
        .div_ceil(config.kv_cache.block_size);
    let physical_slots = if *target == BackendTarget::Cuda {
        CUDA_SLIDING_PHYSICAL_SLOTS_PER_SESSION
    } else {
        1
    };
    let sessions = config.scheduler.max_batch_requests.max(1).saturating_mul(physical_slots);
    let blocks = ring_blocks.saturating_mul(sessions).min(config.kv_cache.block_count as usize);
    CacheConfig {
        block_count: u32::try_from(blocks).unwrap_or(u32::MAX),
        ..config.kv_cache
    }
}

const fn bounded_sliding(decoder: &DecoderConfig) -> bool {
    decoder.swiglu_limit.is_some() && decoder.num_experts.is_some()
}

const fn workspace(weights: u64) -> u64 {
    let proportional = weights / 10;
    if proportional > 512 * MIB {
        proportional
    } else {
        512 * MIB
    }
}

fn cuda_persistent_workspace(execution: Option<&DecoderExecutionContract>) -> u64 {
    execution.map_or(0, |execution| {
        execution
            .bindings
            .tensors
            .iter()
            .filter(|binding| {
                matches!(
                    binding.role,
                    LogicalTensorRole::Layer {
                        tensor: LayerTensorRole::ExpertProjection { .. },
                        ..
                    }
                ) && matches!(
                    binding.storage,
                    TensorStorage::BlockQuantized { format, .. }
                        if format.format == BlockFormat::NvFp4
                            && format.activation_mode == BlockActivationMode::WeightOnly
                )
            })
            .filter_map(|binding| binding.logical_shape.as_deref())
            .map(nvfp4_marlin_bytes)
            .fold(0_u64, u64::saturating_add)
    })
}

fn nvfp4_marlin_bytes(shape: &[usize]) -> u64 {
    let elements = shape.iter().fold(1_u64, |elements, dimension| {
        elements.saturating_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
    });
    let experts = shape.first().copied().filter(|_| shape.len() == 3).unwrap_or(1);
    elements
        .div_ceil(2)
        .saturating_add(elements.div_ceil(16))
        .saturating_add(u64::try_from(experts).unwrap_or(u64::MAX).saturating_mul(4))
}

#[cfg(test)]
mod tests;
