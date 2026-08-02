use std::sync::Arc;

use foundation::model::ModelManifest;
use models::{
    execution::TaskExecutionPlan,
    layout::{ModelLayout, ModelMetadata, VisionConfig},
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};

use super::{
    KV_CACHE_STEP, LoadedExecution, LoadedModel, ModelInfo, memory::prefix_cache_budget,
    prefill_step,
};

mod backend;
mod execution;
mod tokenizer;

use backend::{load_decoder_model, load_vision_model, validate_kv_storage};
use execution::execution_metadata;
use tokenizer::tokenizer_info;

use crate::{
    MetalConfig, MetalProgressEvent,
    engine::{ModelTensors, Stream, configure_recommended_wired_limit, lowering, memory_stats},
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

    #[allow(clippy::too_many_lines)]
    pub fn load_with_config(
        manifest: &ModelManifest,
        config: Arc<MetalConfig>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        let layout = ModelLayout::inspect(&manifest.path)?;
        let metadata = ModelMetadata::from_layout(&layout)?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let task_plan = TaskExecutionPlan::discover(&layout, &catalog)?;
        let (decoder, encoder, contract) = execution_metadata(&task_plan, &layout, &catalog)?;
        let decoder_lowering = matches!(task_plan, TaskExecutionPlan::Generation { .. })
            .then(|| {
                contract
                    .as_ref()
                    .ok_or_else(|| Error::UnsupportedModel("generation contract is missing".into()))
                    .and_then(|contract| Ok(lowering::plan(&contract.semantic)?))
            })
            .transpose()?;
        let vision = VisionConfig::from_layout(&layout)?;
        if let Some(decoder) = decoder.as_ref()
            && matches!(task_plan, TaskExecutionPlan::Generation { .. })
        {
            let lowering = decoder_lowering
                .as_ref()
                .ok_or_else(|| Error::UnsupportedModel("generation lowering is missing".into()))?;
            validate_kv_storage(decoder, lowering, vision.as_ref(), config.kv_cache.dtype)?;
        }
        let vision_readiness = vision
            .as_ref()
            .map(|config| VisionTensorSchema::discover(config).readiness(&catalog));
        let load_stream = Stream::new_cpu()?;
        let tensors = ModelTensors::load_layout_materialized_with_progress(
            &layout,
            &load_stream,
            |loaded, total, detail| {
                progress(MetalProgressEvent::load_weights(loaded, total, detail));
            },
        )?;
        let tensor_count = tensors.len();
        let stream = Stream::new_gpu_with_config(config)?;
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
            TaskExecutionPlan::Generation { .. } => {
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
        let prefix_cache_entries = stream.config().cache.prefix_cache_entries;
        let prefix_cache_bytes =
            prefix_cache_budget(metal_memory, stream.config().cache.prefix_cache_bytes);
        let model_prefill_step = contract.as_ref().map_or(0, |contract| {
            prefill_step(&contract.semantic, stream.config().cache.prefill_step)
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
        })
    }
}
