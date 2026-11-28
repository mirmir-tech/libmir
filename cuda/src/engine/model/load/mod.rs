use std::{collections::HashSet, sync::Mutex};

use architecture::{InspectedModel, cuda::CudaArchitecture};
use foundation::model::ModelManifest;
use runtime::{backend::ModelHandle, progress::ProgressEvent};

use super::{LoadedModel, capacity::max_sequence_blocks};
use crate::{
    Result,
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

    /// Loads with one sequence's block table sized for `capacity_blocks`
    /// rather than the cache configured now, which may be provisional.
    pub fn load_model_with_sequence_capacity(
        &self,
        manifest: &ModelManifest,
        capacity_blocks: Option<u32>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<ModelHandle> {
        let inspected = InspectedModel::inspect_directory(&manifest.path)?;
        let admitted = inspected.admit_cuda()?;
        self.load_admitted(manifest, &inspected, &admitted, capacity_blocks, progress)
    }

    /// Loads an already-inspected, already-admitted checkpoint onto CUDA
    /// without re-reading layout files or re-running admission.
    pub fn load_admitted(
        &self,
        manifest: &ModelManifest,
        inspected: &InspectedModel,
        admitted: &CudaArchitecture,
        capacity_blocks: Option<u32>,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<ModelHandle> {
        let total = inspected.layout.weights.iter().map(|weight| weight.bytes).sum();
        progress(ProgressEvent::load_weights(0, total, "loading admitted checkpoint"));
        let capacity = ::runtime::kv::CacheConfig {
            block_count: capacity_blocks.unwrap_or(self.cache.block_count),
            ..self.cache
        };
        let blocks = max_sequence_blocks(manifest.context_len, capacity)?;
        let mut report = |current: u64, detail: String| {
            progress(ProgressEvent::load_weights(current.min(total), total, detail));
        };
        let _execution = self.execution.acquire_prefill()?;
        let (runner, generation_admitted) = self.load_task_runner(
            manifest,
            &inspected.task_plan,
            admitted,
            inspected.decoder.as_ref(),
            inspected.encoder.as_ref(),
            inspected.contract.as_ref(),
            &inspected.catalog,
            blocks,
            &mut report,
        )?;
        let inline = inspected
            .layout
            .derived
            .as_ref()
            .and_then(|derived| derived.assembled_safetensors.as_ref());
        let vision_model = load_vision_model(
            &self.backend,
            inspected.vision.as_ref(),
            inspected.vision_readiness.as_ref(),
            &inspected.catalog,
            inline,
        )?;
        self.backend.synchronize()?;
        let loaded = LoadedModel {
            manifest: manifest.clone(),
            layout: inspected.layout.clone(),
            metadata: inspected.metadata.clone(),
            decoder: inspected.decoder.clone(),
            encoder: inspected.encoder.clone(),
            catalog: inspected.catalog.clone(),
            contract: inspected.contract.clone(),
            task_plan: inspected.task_plan.clone(),
            admitted: generation_admitted,
            vision: inspected.vision.clone(),
            vision_readiness: inspected.vision_readiness.clone(),
            vision_model,
            sessions: Mutex::new(HashSet::new()),
            runner: RunnerQueue::shared(runner, &self.execution),
        };
        self.models()?.insert(manifest.id.clone(), std::sync::Arc::new(loaded));
        progress(ProgressEvent::load_weights(total, total, "checkpoint resident on CUDA"));
        Ok(ModelHandle {
            id: manifest.id.clone(),
            backend: "cuda-native".into(),
        })
    }
}
