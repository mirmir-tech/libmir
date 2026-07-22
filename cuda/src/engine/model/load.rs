use std::{collections::HashSet, path::Path, sync::Mutex};

use foundation::model::ModelManifest;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{ModelLayout, ModelMetadata, VisionConfig},
    weights::{BlockFormat, TensorCatalog, VisionTensorSchema},
};
use runtime::{backend::ModelHandle, progress::ProgressEvent};
use uuid::Uuid;

use super::{
    LoadedModel, ModelExecution, ModelRunner,
    generation::{
        GenerationExecution, GraphExecution, MixedMixerExecution, SinkAttentionExecution,
    },
};
use crate::{
    DenseSwiGluLayerLoadConfig, Error, NvFp4MoeLayerLoadConfig, Result,
    SharedRoutedModelLoadConfig,
    backend::{CudaSequenceScoringModel, CudaTextEmbeddingModel},
    engine::{
        CudaEngine, batch::DecodeBuckets, lowering::CudaDecoderPlan, runner::RunnerQueue,
        vision::model::load_vision_model,
    },
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
        let contract = decoder
            .as_ref()
            .map(|decoder| DecoderExecutionContract::discover(&layout, decoder, &catalog))
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

    #[allow(clippy::too_many_arguments)]
    fn load_task_runner(
        &self,
        manifest: &ModelManifest,
        task: &TaskExecutionPlan,
        decoder: Option<&models::layout::DecoderConfig>,
        encoder: Option<&models::layout::EncoderConfig>,
        contract: Option<&DecoderExecutionContract>,
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
        let contract =
            contract.ok_or_else(|| Error::State("CUDA decoder contract is missing".into()))?;
        let plan = CudaDecoderPlan::lower(&contract.semantic);
        if plan.all_shared_routed() && plan.has_linear_mixer() && plan.has_softmax_mixer() {
            let template = self.backend.load_shared_routed_model_template_with_progress(
                decoder,
                &contract.semantic,
                catalog,
                &contract.bindings,
                SharedRoutedModelLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                },
                report,
            )?;
            return Ok(generation(MixedMixerExecution::new(template)));
        }
        if plan.all_unshared_clamped_routed() {
            let template = self.backend.load_clamped_routed_model_template_with_progress(
                decoder, contract, catalog, self.cache, blocks, report,
            )?;
            return Ok(generation(SinkAttentionExecution::new(template)));
        }
        self.load_standard_runner(manifest, decoder, &plan, contract, catalog, blocks, report)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_standard_runner(
        &self,
        manifest: &ModelManifest,
        decoder: &models::layout::DecoderConfig,
        plan: &CudaDecoderPlan,
        contract: &DecoderExecutionContract,
        catalog: &TensorCatalog,
        blocks: usize,
        report: &mut dyn FnMut(u64, String),
    ) -> Result<ModelRunner> {
        let bindings = &contract.bindings;
        let nvfp4 = bindings.uses_block_format(BlockFormat::NvFp4);
        let int8 = bindings.uses_packed_int8();
        let template = if plan.all_dense_and_routed() && nvfp4 {
            self.backend.load_nvfp4_moe_model_template_with_bindings(
                decoder,
                bindings,
                catalog,
                NvFp4MoeLayerLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                },
                report,
            )?
        } else if plan.all_dense() {
            self.backend.load_dense_swiglu_model_template_with_bindings(
                decoder,
                bindings,
                catalog,
                DenseSwiGluLayerLoadConfig {
                    cache: self.cache,
                    max_sequence_blocks: blocks,
                    qkv_normalization: plan.graph_normalization()?,
                    projection_format: if nvfp4 {
                        crate::ProjectionFormat::NvFp4
                    } else if int8 {
                        crate::ProjectionFormat::Int8
                    } else {
                        crate::ProjectionFormat::Bf16
                    },
                },
                report,
            )?
        } else {
            return Err(Error::MissingCapability {
                operation: "decoder layer composition",
                storage: if nvfp4 {
                    "NVFP4 bindings"
                } else {
                    "non-NVFP4 bindings"
                }
                .into(),
                geometry: format!("layers={}", plan.layers().len()),
                requirement: "the available graph decoder admits uniform dense layers or dense-plus-routed NVFP4 layers",
            });
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
            execution: ModelExecution::Generation(Box::new(GraphExecution::new(session))),
            batches: Some(batches),
            selected: None,
        })
    }
}

fn runner(execution: ModelExecution) -> ModelRunner {
    ModelRunner { execution, batches: None, selected: None }
}

fn generation(execution: impl GenerationExecution + 'static) -> ModelRunner {
    runner(ModelExecution::Generation(Box::new(execution)))
}
