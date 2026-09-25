use std::{collections::HashSet, path::Path, sync::Mutex};

use foundation::model::ModelManifest;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{ModelLayout, ModelMetadata, VisionConfig},
    weights::{TensorCatalog, VisionTensorSchema},
};
use runtime::{backend::ModelHandle, progress::ProgressEvent};

use super::{LoadedModel, capacity::max_sequence_blocks};
use crate::{
    CudaArchitecture, CudaDecoderRuntime, Result,
    engine::{CudaEngine, runner::RunnerQueue, vision::model::load_vision_model},
};

mod runner;

impl CudaEngine {
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<ModelHandle> {
        self.load_model_with_sequence_capacity(manifest, None, progress)
    }

    /// Whether the runtime this checkpoint lowers to can reallocate its K/V
    /// pages after loading (`resize_kv_cache`), so a provisional cache and a
    /// measured resize are possible. Inspects the layout only.
    pub(crate) fn kv_cache_resizable_from_layout(manifest: &ModelManifest) -> Result<bool> {
        let layout = ModelLayout::inspect(Path::new(&manifest.path))?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let task_plan = TaskExecutionPlan::discover(&layout, &catalog)?;
        let TaskExecutionPlan::Generation { decoder } = &task_plan else {
            return Ok(false);
        };
        let contract = DecoderExecutionContract::discover(&layout, decoder, &catalog)?;
        Ok(matches!(
            crate::admit_architecture(&task_plan, Some(&contract.semantic))?,
            CudaArchitecture::Generation(
                CudaDecoderRuntime::SharedRouted | CudaDecoderRuntime::DenseMixed
            )
        ))
    }

    /// Loads with one sequence's block table sized for `capacity_blocks`
    /// rather than the cache configured now, which may be provisional.
    pub fn load_model_with_sequence_capacity(
        &self,
        manifest: &ModelManifest,
        capacity_blocks: Option<u32>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<ModelHandle> {
        let layout = ModelLayout::inspect(Path::new(&manifest.path))?;
        let total = layout.weights.iter().map(|weight| weight.bytes).sum();
        progress(ProgressEvent::load_weights(0, total, "inspecting checkpoint"));
        let metadata = ModelMetadata::from_layout(&layout)?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let task_plan = TaskExecutionPlan::discover(&layout, &catalog)?;
        let (decoder, encoder) = match &task_plan {
            TaskExecutionPlan::Generation { decoder }
            | TaskExecutionPlan::CausalScoring { decoder, .. }
            | TaskExecutionPlan::Embedding { decoder, .. } => (Some(decoder.clone()), None),
            TaskExecutionPlan::SequenceScoring { encoder, .. } => (None, Some(encoder.clone())),
        };
        let contract = decoder
            .as_ref()
            .map(|decoder| DecoderExecutionContract::discover(&layout, decoder, &catalog))
            .transpose()?;
        let vision = VisionConfig::from_layout(&layout)?;
        let vision_readiness = vision
            .as_ref()
            .map(|config| VisionTensorSchema::discover(config).readiness(&catalog));
        let capacity = ::runtime::kv::CacheConfig {
            block_count: capacity_blocks.unwrap_or(self.cache.block_count),
            ..self.cache
        };
        let blocks = max_sequence_blocks(manifest.context_len, capacity)?;
        let mut report = |current: u64, detail: String| {
            progress(ProgressEvent::load_weights(current.min(total), total, detail));
        };
        let runner = self.load_task_runner(
            manifest,
            &task_plan,
            decoder.as_ref(),
            encoder.as_ref(),
            contract.as_ref(),
            &catalog,
            blocks,
            &mut report,
        )?;
        let vision_model =
            load_vision_model(&self.backend, vision.as_ref(), vision_readiness.as_ref(), &catalog)?;
        self.backend.synchronize()?;
        let loaded = LoadedModel {
            manifest: manifest.clone(),
            layout,
            metadata,
            decoder,
            encoder,
            catalog,
            contract,
            task_plan,
            vision,
            vision_readiness,
            vision_model,
            sessions: Mutex::new(HashSet::new()),
            runner: RunnerQueue::new(runner, self.scheduler.decode_priority_burst),
        };
        self.models()?.insert(manifest.id.clone(), std::sync::Arc::new(loaded));
        progress(ProgressEvent::load_weights(total, total, "checkpoint resident on CUDA"));
        Ok(ModelHandle {
            id: manifest.id.clone(),
            backend: "cuda-native".into(),
        })
    }
}
