//! Single-pass local checkpoint inspection shared across `libmir`, Metal, and
//! CUDA so backend loaders consume an inspected contract and admitted plan
//! without re-reading checkpoint files or re-running admission.

use std::path::Path;

use foundation::model::BackendTarget;
use models::{
    ModelsError,
    execution::{ArchitectureRequirements, DecoderExecutionContract, TaskExecutionPlan},
    layout::{
        DecoderConfig, EncoderConfig, ImageProcessorConfig, ModelLayout, ModelMetadata,
        VisionConfig,
    },
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};

use crate::{
    Result,
    cuda::{self, CudaArchitecture},
    metal::{self, MetalArchitecture},
    physical::{self, BackendAdmissionReport, CheckpointEncoding},
};

/// Inspected checkpoint layout, tensor catalog, task plan, and semantic/weight
/// contracts produced once before backend admission and weight loading.
#[derive(Debug, Clone)]
pub struct InspectedModel {
    pub layout: ModelLayout,
    pub metadata: ModelMetadata,
    pub catalog: TensorCatalog,
    pub task_plan: TaskExecutionPlan,
    pub decoder: Option<DecoderConfig>,
    pub encoder: Option<EncoderConfig>,
    pub contract: Option<DecoderExecutionContract>,
    pub vision: Option<VisionConfig>,
    pub vision_readiness: Option<TensorReadiness>,
    pub image_processor: Option<ImageProcessorConfig>,
}

impl InspectedModel {
    /// Inspects a Hugging Face-style checkpoint directory.
    pub fn inspect_directory(path: impl AsRef<Path>) -> std::result::Result<Self, ModelsError> {
        Self::from_layout(ModelLayout::inspect(path)?)
    }

    /// Inspects a local GGUF text model and optional `mmproj` vision projector
    /// directly in memory without creating an on-disk import directory.
    pub fn inspect_gguf(
        text: impl AsRef<Path>,
        vision: Option<&Path>,
    ) -> std::result::Result<Self, ModelsError> {
        Self::from_layout(ModelLayout::from_gguf(text, vision)?)
    }

    /// Inspects an already-constructed [`ModelLayout`].
    pub fn from_layout(layout: ModelLayout) -> std::result::Result<Self, ModelsError> {
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
        let image_processor = vision
            .as_ref()
            .map(|vision| ImageProcessorConfig::from_layout(&layout, vision.pipeline()))
            .transpose()?
            .flatten();
        Ok(Self {
            layout,
            metadata,
            catalog,
            task_plan,
            decoder,
            encoder,
            contract,
            vision,
            vision_readiness,
            image_processor,
        })
    }

    /// Runs Metal architecture admission once for this inspected model.
    pub fn admit_metal(&self) -> Result<MetalArchitecture> {
        metal::admit(&self.task_plan, self.contract.as_ref().map(|contract| &contract.semantic))
    }

    /// Runs CUDA architecture admission once for this inspected model.
    pub fn admit_cuda(&self) -> Result<CudaArchitecture> {
        cuda::admit(&self.task_plan, self.contract.as_ref().map(|contract| &contract.semantic))
    }

    /// Reports conservative static compatibility with `backend`.
    #[must_use]
    pub fn admission_report(&self, backend: BackendTarget) -> BackendAdmissionReport {
        physical::inspect(
            self.contract.as_ref(),
            &self.task_plan,
            self.vision.as_ref(),
            self.vision_readiness.as_ref(),
            self.image_processor.as_ref(),
            backend,
        )
    }

    /// Describes the deduplicated physical weight encodings of this checkpoint.
    #[must_use]
    pub fn checkpoint_encoding(&self) -> CheckpointEncoding {
        let mut encoding = self.contract.as_ref().map_or_else(
            || match &self.task_plan {
                TaskExecutionPlan::SequenceScoring { bindings, .. } => {
                    CheckpointEncoding::from_encoder_bindings(bindings)
                },
                TaskExecutionPlan::Generation { .. }
                | TaskExecutionPlan::CausalScoring { .. }
                | TaskExecutionPlan::Embedding { .. } => CheckpointEncoding::default(),
            },
            |contract| CheckpointEncoding::from_bindings(&contract.bindings),
        );
        if let Some(readiness) = &self.vision_readiness {
            encoding.include_dense_dtypes(&readiness.dtypes);
        }
        encoding
    }

    /// Returns normalized task and decoder capabilities required by this model.
    #[must_use]
    pub fn architecture_requirements(&self) -> ArchitectureRequirements {
        ArchitectureRequirements::discover(
            &self.task_plan,
            self.contract.as_ref().map(|contract| &contract.semantic),
        )
    }
}
