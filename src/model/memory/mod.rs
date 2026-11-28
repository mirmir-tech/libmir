pub(super) mod admission;
pub(super) mod policy;
use architecture::{CacheLayout, LayerCacheEntry, LayerCacheKind, cuda::CudaDecoderRuntime};
use foundation::model::BackendTarget;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::{LayerCacheShape, SemanticModelSpec},
    weights::{
        BlockActivationMode, BlockFormat, LayerTensorRole, LogicalTensorRole, TensorStorage,
    },
};
use runtime::kv::{CacheConfig, KvStorageSpec, ring_blocks};

use super::ModelDescriptor;
use crate::{ModelMemoryEstimate, RuntimeConfig};

mod conversion;
pub(super) mod prediction;
mod resident;
mod session;

pub use resident::UnloadedTensors;
pub(super) use resident::unloaded;

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
        let speculation = config.scheduler.speculation.is_some_and(|speculation| {
            prediction::backend_drafts(*target, speculation.draft_tokens.get()) > 0
        });
        let weight_bytes =
            resident::weight_bytes(&self.inspected.layout, self.unloaded.as_ref(), speculation)
                .saturating_add(if *target == BackendTarget::Metal {
                    conversion::metal_affine_growth(self.inspected.contract.as_ref(), speculation)
                } else {
                    0
                });
        let generation_decoder = match &self.inspected.task_plan {
            TaskExecutionPlan::Generation { decoder }
            | TaskExecutionPlan::CausalScoring { decoder, .. } => Some(decoder),
            TaskExecutionPlan::Embedding { .. } | TaskExecutionPlan::SequenceScoring { .. } => None,
        };
        let cache_capacity_tokens = generation_decoder.map_or(0, |_| {
            u64::from(config.kv_cache.block_count)
                .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX))
        });
        let semantic = self.inspected.contract.as_ref().map(|execution| &execution.semantic);
        let (kv_bytes_per_token, kv_cache_bytes) =
            generation_decoder.zip(semantic).map_or((0, 0), |(_, semantic)| {
                let layout = cache_layout(&self.inspected.task_plan, semantic, *target);
                let (per_token, total) = kv_budget(&layout.layers, config);
                let (prediction_per_token, prediction) =
                    prediction::kv_budget(&layout.prediction_layers, config, *target);
                (per_token.saturating_add(prediction_per_token), total.saturating_add(prediction))
            });
        let workspace_bytes =
            workspace(weight_bytes).saturating_add(if *target == BackendTarget::Cuda {
                cuda_persistent_workspace(self.inspected.contract.as_ref())
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
            model_context_tokens: u64::try_from(self.inspected.metadata.context_len)
                .unwrap_or(u64::MAX),
            session_state_bytes: generation_decoder.map_or(0, |decoder| {
                session::state_bytes(
                    decoder,
                    prediction::rollback_snapshots(
                        self.inspected.contract.as_ref(),
                        config,
                        *target,
                    ),
                )
            }),
        }
    }
}

pub(super) fn cache_layout(
    task: &TaskExecutionPlan,
    semantic: &SemanticModelSpec,
    target: BackendTarget,
) -> CacheLayout {
    match target {
        BackendTarget::Metal => match architecture::metal::admit(task, Some(semantic)) {
            Ok(architecture::metal::MetalArchitecture::Generation(plan)) => {
                plan.cache_layout().clone()
            },
            _ => CacheLayout::for_metal(&semantic.decoder),
        },
        BackendTarget::Cuda => match architecture::cuda::admit(task, Some(semantic)) {
            Ok(architecture::cuda::CudaArchitecture::Generation(admitted)) => admitted.cache_layout,
            _ => CacheLayout::for_cuda(&semantic.decoder, CudaDecoderRuntime::Dense),
        },
    }
}

pub(super) fn kv_budget(entries: &[LayerCacheEntry], config: &RuntimeConfig) -> (u64, u64) {
    entries.iter().fold((0_u64, 0_u64), |(per_token, total), entry| {
        let (layer_per_token, layer_total) = layer_kv_budget(entry.kind, config);
        (per_token.saturating_add(layer_per_token), total.saturating_add(layer_total))
    })
}

fn layer_kv_budget(kind: LayerCacheKind, config: &RuntimeConfig) -> (u64, u64) {
    let bytes = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
    match kind {
        LayerCacheKind::Paged { shape } => {
            let budget = layer_storage(shape, config.kv_cache).memory_budget();
            let growing = bytes(budget.data_bytes_per_token)
                .saturating_add(bytes(budget.scale_bytes_per_token));
            (growing, bytes(budget.total_bytes))
        },
        LayerCacheKind::Ring { shape, window, slots_per_session } => {
            let cache = ring_cache(window.get(), slots_per_session, config);
            let budget = layer_storage(shape, cache).memory_budget();
            (0, bytes(budget.total_bytes))
        },
        LayerCacheKind::SharedFrom { .. } | LayerCacheKind::Recurrent(_) => (0, 0),
    }
}

/// A layer's K/V storage of `shape` under `cache`.
fn layer_storage(shape: LayerCacheShape, cache: CacheConfig) -> KvStorageSpec {
    KvStorageSpec {
        key_head_dim: shape.key_width,
        value_head_dim: shape.value_width,
        ..KvStorageSpec::new(cache, shape.heads, shape.key_width)
    }
}

/// The ring a bounded sliding layer of `window` recent tokens keeps across
/// concurrent sessions.
fn ring_cache(window: usize, slots_per_session: usize, config: &RuntimeConfig) -> CacheConfig {
    let per_slot = ring_blocks(window, config.kv_cache.block_size);
    let sessions = config.scheduler.max_batch_requests.max(1).saturating_mul(slots_per_session);
    let blocks = per_slot.saturating_mul(sessions).min(config.kv_cache.block_count as usize);
    CacheConfig {
        block_count: u32::try_from(blocks).unwrap_or(u32::MAX),
        ..config.kv_cache
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
