use std::sync::Arc;

use architecture::{InspectedModel, metal::MetalArchitecture};
use foundation::model::ModelManifest;
use models::{
    execution::TaskExecutionPlan,
    weights::{TensorBinding, TensorReadiness},
};

use super::{
    KV_CACHE_STEP, LoadedExecution, LoadedModel, ModelInfo, memory::prefix_cache_budget,
    prefill_step,
};

mod backend;
mod tokenizer;

use backend::{load_decoder_model, load_vision_model, validate_kv_storage};
use tokenizer::tokenizer_info;

use crate::{
    MetalConfig, MetalProgressEvent,
    engine::{ModelTensors, Stream, configure_recommended_wired_limit, memory_stats},
    native::{
        error::{Error, Result},
        prefix::PrefixCache,
    },
};

impl LoadedModel {
    #[cfg(test)]
    pub fn load(
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        Self::load_with_config(manifest, Arc::default(), progress)
    }

    #[cfg(test)]
    pub fn load_with_config(
        manifest: &ModelManifest,
        config: Arc<MetalConfig>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        Self::load_with_config_and_pool(manifest, config, Arc::default(), progress)
    }

    pub(crate) fn load_with_config_and_pool(
        manifest: &ModelManifest,
        config: Arc<MetalConfig>,
        paged_arenas: Arc<crate::engine::PagedArenaPool>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        let inspected = InspectedModel::inspect_directory(&manifest.path)?;
        let admitted = inspected
            .admit_metal()
            .map_err(|error| Error::UnsupportedModel(error.to_string()))?;
        Self::load_admitted(manifest, inspected, admitted, config, paged_arenas, progress)
    }

    /// Loads an already-inspected, already-admitted checkpoint onto Metal
    /// without re-reading layout files or re-running admission.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn load_admitted(
        manifest: &ModelManifest,
        inspected: InspectedModel,
        admitted: MetalArchitecture,
        config: Arc<MetalConfig>,
        paged_arenas: Arc<crate::engine::PagedArenaPool>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        let InspectedModel {
            layout,
            metadata,
            catalog,
            task_plan,
            decoder,
            encoder,
            contract,
            vision,
            vision_readiness,
            ..
        } = inspected;
        let decoder_lowering = match admitted {
            MetalArchitecture::Generation(plan) => Some(plan),
            MetalArchitecture::Embedding | MetalArchitecture::SequenceScoring => None,
        };
        if let Some(decoder) = decoder.as_ref()
            && matches!(
                task_plan,
                TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
            )
        {
            let lowering = decoder_lowering
                .as_ref()
                .ok_or_else(|| Error::UnsupportedModel("generation lowering is missing".into()))?;
            validate_kv_storage(decoder, lowering, vision.as_ref(), config.kv_cache.dtype)?;
        }
        let load_stream = Stream::new_cpu()?;
        let full_precision = contract
            .iter()
            .flat_map(|contract| &contract.bindings.tensors)
            .flat_map(TensorBinding::full_precision_sources)
            .map(str::to_owned)
            .collect();
        let tensors = ModelTensors::load_catalog_materialized_keeping(
            &layout,
            &catalog,
            &full_precision,
            &load_stream,
            |loaded, total, detail| {
                progress(MetalProgressEvent::load_weights(loaded, total, detail));
            },
        )?;
        let tensor_count = tensors.len();
        let stream = Stream::new_gpu_with_config_and_pool(config, paged_arenas)?;
        let _configured_wired_limit = configure_recommended_wired_limit()?;
        let execution = match &task_plan {
            TaskExecutionPlan::Embedding { decoder, task, tensors: tensor_layout } => {
                if task.pooling != models::execution::PoolingMode::LastToken || !task.normalize {
                    return Err(Error::UnsupportedModel(
                        "Metal embedding currently requires normalized last-token pooling".into(),
                    ));
                }
                LoadedExecution::Embedding(crate::engine::TextEmbeddingModel::load(
                    &tensors, decoder, tensor_layout, &stream,
                )?)
            },
            TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. } => {
                let decoder = decoder.as_ref().ok_or_else(|| {
                    Error::UnsupportedModel("generation decoder config is missing".into())
                })?;
                let contract = contract.as_ref().ok_or_else(|| {
                    Error::UnsupportedModel("generation contract is missing".into())
                })?;
                let lowering = decoder_lowering.as_ref().ok_or_else(|| {
                    Error::UnsupportedModel("generation lowering is missing".into())
                })?;
                LoadedExecution::Generation(load_decoder_model(
                    lowering,
                    &tensors,
                    decoder,
                    &contract.semantic.decoder,
                    &contract.bindings,
                    &stream,
                )?)
            },
            TaskExecutionPlan::SequenceScoring { encoder, bindings, .. } => {
                LoadedExecution::SequenceScoring(Box::new(
                    crate::engine::SequenceScoringModel::load(
                        &tensors, encoder, bindings, &stream,
                    )?,
                ))
            },
        };
        let vision_model = load_vision_model(
            vision.as_ref(),
            vision_readiness.as_ref().is_some_and(TensorReadiness::is_ready),
            contract.as_ref().map(|contract| &contract.bindings),
            &tensors,
            &stream,
        )?;
        let metal_memory = memory_stats()?;
        let (tokenizer, tokenizer_error) = tokenizer_info(&layout);
        let weight_bytes = layout.weights.iter().map(|weight| weight.bytes).sum();
        let cache_layout = decoder_lowering.as_ref().map(|plan| plan.cache_layout().clone());
        let non_paged_prefix = cache_layout
            .as_ref()
            .is_some_and(architecture::CacheLayout::has_non_paged_state);
        let prefix_cache_entries = if non_paged_prefix {
            stream.config().cache.prefix_cache_entries
        } else {
            0
        };
        let prefix_cache_bytes = if non_paged_prefix {
            prefix_cache_budget(metal_memory, stream.config().cache.prefix_cache_bytes)
        } else {
            0
        };
        let model_prefill_step = contract.as_ref().map_or(0, |contract| {
            prefill_step(&contract.semantic, &contract.bindings, stream.config().cache.prefill_step)
        });
        Ok(Self {
            info: ModelInfo {
                manifest: manifest.clone(),
                layout,
                metadata,
                decoder,
                encoder,
                vision,
                vision_readiness,
                contract,
                cache_layout,
                task_plan,
                tensor_count,
                weight_bytes,
                cache_step: KV_CACHE_STEP,
                prefill_step: model_prefill_step,
                tokenizer,
                tokenizer_error,
                metal_memory,
            },
            stream,
            execution,
            vision_model,
            prefixes: PrefixCache::new(prefix_cache_entries, prefix_cache_bytes),
            sessions: std::collections::HashMap::new(),
            recovery: super::recovery::ExecutionRecovery::Ready,
            active_prefills: crate::native::prefill::ActivePrefills::default(),
            history_prefill_reserved: false,
        })
    }
}
