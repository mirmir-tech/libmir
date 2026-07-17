use std::{
    collections::HashSet,
    path::Path,
    sync::{Mutex, MutexGuard},
};

use foundation::model::{ModelManifest, Quantization};
use models::{
    execution::{AttentionFeature, DecoderArchetype, ExecutionPlan},
    layout::{DecoderConfig, ModelLayout, ModelMetadata},
    weights::TensorCatalog,
};
use runtime::{backend::ModelHandle, progress::ProgressEvent};
use uuid::Uuid;

use super::{
    CudaEngine,
    batch::DecodeBuckets,
    runner::{RunnerGuard, RunnerQueue},
};
use crate::{
    CudaMoeModelSession, DenseSwiGluLayerLoadConfig, Error, NvFp4MoeLayerLoadConfig, Result,
    kernels::QkvNormalization,
};

pub(super) struct LoadedModel {
    pub manifest: ModelManifest,
    pub layout: ModelLayout,
    pub metadata: ModelMetadata,
    pub decoder: DecoderConfig,
    pub catalog: TensorCatalog,
    pub plan: ExecutionPlan,
    sessions: Mutex<HashSet<Uuid>>,
    runner: RunnerQueue<ModelRunner>,
}

pub(super) struct ModelRunner {
    pub model: CudaMoeModelSession,
    pub batches: DecodeBuckets,
    pub selected: Option<DeviceToken>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DeviceToken {
    pub session: Uuid,
    pub token: u32,
}

impl CudaEngine {
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<ModelHandle> {
        let layout = ModelLayout::inspect(Path::new(&manifest.path))?;
        let total = layout.weights.iter().map(|weight| weight.bytes).sum();
        progress(ProgressEvent::load_weights(0, total, "inspecting checkpoint"));
        let metadata = ModelMetadata::from_layout(&layout)?;
        let decoder = DecoderConfig::from_layout(&layout)?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let plan = ExecutionPlan::discover(&decoder, &catalog)?;
        let blocks = usize::try_from(self.cache.block_count)?;
        let mut report = |current: u64, detail: String| {
            progress(ProgressEvent::load_weights(current.min(total), total, detail));
        };
        let template = match (plan.decoder, &manifest.quantization) {
            (DecoderArchetype::HybridMoe, Quantization::NvFp4) => {
                self.backend.load_nvfp4_moe_model_template_with_progress(
                    &decoder,
                    &catalog,
                    NvFp4MoeLayerLoadConfig {
                        cache: self.cache,
                        max_sequence_blocks: blocks,
                    },
                    &mut report,
                )?
            },
            (
                DecoderArchetype::DenseSwiGlu,
                Quantization::Bf16 | Quantization::None | Quantization::NvFp4,
            ) => self.backend.load_dense_swiglu_model_template_with_progress(
                &decoder,
                &catalog,
                DenseSwiGluLayerLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                    qkv_normalization: normalization(plan.attention)?,
                    projection_format: match manifest.quantization {
                        Quantization::NvFp4 => crate::ProjectionFormat::NvFp4,
                        _ => crate::ProjectionFormat::Bf16,
                    },
                },
                &mut report,
            )?,
            (decoder, quantization) => {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "CUDA does not implement {decoder:?} with {quantization:?} weights"
                )));
            },
        };
        progress(ProgressEvent::load_weights(total, total, "preparing CUDA execution runner"));
        let caches = template.allocate_shared_kv()?;
        let mut runner =
            template.instantiate_with_config_and_caches(self.session_config, &caches)?;
        runner.warmup(Uuid::nil(), self.cache, manifest.context_len)?;
        let selected = runner.sample(runtime::backend::SamplingLogits::None)?;
        let _token = self.backend.read_token(selected)?;
        progress(ProgressEvent::load_weights(total, total, "warming CUDA decode buckets"));
        let batches = DecodeBuckets::prepare(
            &template,
            &caches,
            self.scheduler.max_batch_requests,
            self.cache,
        )?;
        self.backend.synchronize()?;
        let loaded = LoadedModel {
            manifest: manifest.clone(),
            layout,
            metadata,
            decoder,
            catalog,
            plan,
            sessions: Mutex::new(HashSet::new()),
            runner: RunnerQueue::new(
                ModelRunner { model: runner, batches, selected: None },
                self.scheduler.decode_priority_burst,
            ),
        };
        self.models()?.insert(manifest.id.clone(), std::sync::Arc::new(loaded));
        progress(ProgressEvent::load_weights(total, total, "checkpoint resident on CUDA"));
        Ok(ModelHandle {
            id: manifest.id.clone(),
            backend: "cuda-native".into(),
        })
    }
}

fn normalization(attention: AttentionFeature) -> Result<QkvNormalization> {
    match attention {
        AttentionFeature::RmsNormalizedSharedKv => Ok(QkvNormalization::ALL),
        AttentionFeature::RmsNormalizedGroupedQuery => Ok(QkvNormalization::QUERY_KEY),
        AttentionFeature::GroupedQuery => Ok(QkvNormalization::NONE),
        AttentionFeature::GatedDeltaAndRmsNormalizedGroupedQuery => Err(
            Error::UnsupportedDecoderLayer("CUDA gated-delta attention is not implemented".into()),
        ),
    }
}

impl LoadedModel {
    pub(super) fn sessions(&self) -> Result<MutexGuard<'_, HashSet<Uuid>>> {
        let Ok(sessions) = self.sessions.lock() else {
            return Err(Error::State("session registry lock is poisoned".into()));
        };
        Ok(sessions)
    }

    pub fn clear_sessions(&self) -> Result<()> {
        let mut runner = self.prefill_runner()?;
        self.sessions()?.clear();
        runner.selected = None;
        drop(runner);
        Ok(())
    }

    pub(super) fn register_session(&self, session: Uuid) -> Result<()> {
        self.sessions()?.insert(session);
        Ok(())
    }

    pub fn release_session(&self, session: Uuid) -> Result<()> {
        let mut runner = self.prefill_runner()?;
        self.sessions()?.remove(&session);
        if runner.selected.is_some_and(|selected| selected.session == session) {
            runner.selected = None;
        }
        drop(runner);
        Ok(())
    }

    pub(super) fn require_session(&self, session: Uuid) -> Result<()> {
        if !self.sessions()?.contains(&session) {
            return Err(Error::State("decode session is not initialized".into()));
        }
        Ok(())
    }

    pub(super) fn decode_runner(&self) -> Result<RunnerGuard<'_, ModelRunner>> {
        self.runner.acquire_decode()
    }

    pub(super) fn prefill_runner(&self) -> Result<RunnerGuard<'_, ModelRunner>> {
        self.runner.acquire_prefill()
    }
}
