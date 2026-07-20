use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Mutex,
};

use foundation::model::{ModelManifest, Quantization};
use models::{
    execution::{AttentionFeature, DecoderArchetype, ExecutionPlan, TaskExecutionPlan},
    layout::{ModelLayout, ModelMetadata, VisionConfig},
    weights::{TensorCatalog, VisionTensorSchema},
};
use runtime::{backend::ModelHandle, progress::ProgressEvent};
use uuid::Uuid;

use super::{HybridExecution, LoadedModel, ModelExecution, ModelRunner};
use crate::{
    DenseSwiGluLayerLoadConfig, Error, HybridLinearModelLoadConfig, NvFp4MoeLayerLoadConfig,
    Result,
    backend::{CudaSequenceScoringModel, CudaTextEmbeddingModel},
    engine::{
        CudaEngine, batch::DecodeBuckets, runner::RunnerQueue, vision::model::load_vision_model,
    },
    kernels::QkvNormalization,
};

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
        let catalog = TensorCatalog::from_layout(&layout)?;
        let task_plan = TaskExecutionPlan::discover(&layout, &catalog)?;
        let (decoder, encoder) = match &task_plan {
            TaskExecutionPlan::Generation { decoder }
            | TaskExecutionPlan::Embedding { decoder, .. } => (Some(decoder.clone()), None),
            TaskExecutionPlan::SequenceScoring { encoder, .. } => (None, Some(encoder.clone())),
        };
        let plan = decoder
            .as_ref()
            .map(|value| ExecutionPlan::discover(value, &catalog))
            .transpose()?;
        let vision = VisionConfig::from_layout(&layout)?;
        let vision_readiness = vision
            .as_ref()
            .map(|config| VisionTensorSchema::discover(config).readiness(&catalog));
        let blocks = usize::try_from(self.cache.block_count)?;
        let mut report = |current: u64, detail: String| {
            progress(ProgressEvent::load_weights(current.min(total), total, detail));
        };
        let runner = self.load_task_runner(
            manifest,
            &task_plan,
            decoder.as_ref(),
            encoder.as_ref(),
            plan.as_ref(),
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
            plan,
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

    #[allow(clippy::too_many_arguments)]
    fn load_task_runner(
        &self,
        manifest: &ModelManifest,
        task: &TaskExecutionPlan,
        decoder: Option<&models::layout::DecoderConfig>,
        encoder: Option<&models::layout::EncoderConfig>,
        plan: Option<&ExecutionPlan>,
        catalog: &TensorCatalog,
        blocks: usize,
        report: &mut dyn FnMut(u64, String),
    ) -> Result<ModelRunner> {
        if let Some(encoder) = encoder {
            return Ok(runner(ModelExecution::SequenceScoring(Box::new(
                CudaSequenceScoringModel::load(&self.backend, encoder, catalog)?,
            ))));
        }
        let decoder =
            decoder.ok_or_else(|| Error::State("CUDA decoder config is missing".into()))?;
        if let TaskExecutionPlan::Embedding { tensors, .. } = task {
            return Ok(runner(ModelExecution::Embedding(Box::new(CudaTextEmbeddingModel::load(
                &self.backend, decoder, catalog, tensors,
            )?))));
        }
        let plan = plan.ok_or_else(|| Error::State("CUDA execution plan is missing".into()))?;
        if plan.decoder == DecoderArchetype::HybridLinearMoe {
            let template = self.backend.load_hybrid_linear_model_template_with_progress(
                decoder,
                catalog,
                HybridLinearModelLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                },
                report,
            )?;
            return Ok(runner(ModelExecution::Hybrid(Box::new(HybridExecution {
                template,
                sessions: HashMap::new(),
            }))));
        }
        self.load_standard_runner(manifest, decoder, *plan, catalog, blocks, report)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_standard_runner(
        &self,
        manifest: &ModelManifest,
        decoder: &models::layout::DecoderConfig,
        plan: ExecutionPlan,
        catalog: &TensorCatalog,
        blocks: usize,
        report: &mut dyn FnMut(u64, String),
    ) -> Result<ModelRunner> {
        let template = match (plan.decoder, &manifest.quantization) {
            (DecoderArchetype::HybridMoe, Quantization::NvFp4) => {
                self.backend.load_nvfp4_moe_model_template_with_progress(
                    decoder,
                    catalog,
                    NvFp4MoeLayerLoadConfig {
                        cache: self.cache,
                        max_sequence_blocks: blocks,
                    },
                    report,
                )?
            },
            (
                DecoderArchetype::DenseSwiGlu,
                Quantization::Bf16 | Quantization::None | Quantization::NvFp4,
            ) => self.backend.load_dense_swiglu_model_template_with_progress(
                decoder,
                catalog,
                DenseSwiGluLayerLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                    qkv_normalization: normalization(plan.attention)?,
                    projection_format: if manifest.quantization == Quantization::NvFp4 {
                        crate::ProjectionFormat::NvFp4
                    } else {
                        crate::ProjectionFormat::Bf16
                    },
                },
                report,
            )?,
            (kind, quantization) => {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "CUDA does not implement {kind:?} with {quantization:?} weights"
                )));
            },
        };
        report(u64::MAX, "preparing CUDA execution runner".into());
        let caches = template.allocate_shared_kv()?;
        let mut session =
            template.instantiate_with_config_and_caches(self.session_config, &caches)?;
        session.warmup(Uuid::nil(), self.cache, manifest.context_len)?;
        let selected = session.sample(runtime::backend::SamplingLogits::None)?;
        let _token = self.backend.read_token(selected)?;
        report(u64::MAX, "warming CUDA decode buckets".into());
        let batches = DecodeBuckets::prepare(
            &template,
            &caches,
            self.scheduler.max_batch_requests,
            self.cache,
        )?;
        Ok(ModelRunner {
            execution: ModelExecution::Standard(Box::new(session)),
            batches: Some(batches),
            selected: None,
        })
    }
}

fn runner(execution: ModelExecution) -> ModelRunner {
    ModelRunner { execution, batches: None, selected: None }
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
