#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

mod cancellation;
mod config;
mod embedding;
mod engine;
mod error;
mod generation;
mod memory;
mod model;
mod rerank;
mod scheduler;
mod session;
mod telemetry;

pub use cancellation::CancellationToken;
pub use config::{MemoryRuntimeConfig, RuntimeConfig, VisionRuntimeConfig};
#[cfg(feature = "cuda")]
pub use cuda::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedPairTensors, AffineQuantizedTensors, Bf16Embedding, Bf16Linear,
    BucketedNvFp4MoeBf16, CudaAttentionPolicy, CudaBackend, CudaConfig, CudaDenseVectorPolicy,
    CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaExecutionPlanner, CudaHardwareProfile,
    CudaKernelAdmission, CudaMemoryArchitecture, CudaModelSessionConfig, CudaMoeBatchPolicy,
    CudaMoeFusionPolicy, CudaMoeModelSession, CudaMoeModelTemplate, CudaNumericalPolicy,
    CudaOutputHeadPolicy, CudaPlanningPolicy, CudaTensor, CudaTensorDType, CudaTensorSet,
    CudaTuningConfig, CudaTuningMode, DecodeMoeLayerTemplate, DenseExecution, DensePlan,
    DensePlanRequest, DenseRole, DeviceSamplerBf16, Error as CudaError, ExecutionPhase,
    GatedActivation, GroupedNvFp4MoeBf16, NvFp4MoeLayerLoadConfig, PlanSource,
    PrefillAttentionBf16, PrefillMoeBlockBf16, SelectedAffineGatedBf16Linear,
    SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear, TensorUploadBatch,
};
pub use embedding::{EmbeddingOutput, EmbeddingRequest};
pub use engine::Engine;
pub use error::{Error, Result};
pub use foundation::{
    self,
    protocol::{
        ChatCompletionRequest, ChatFunctionCall, ChatFunctionDefinition, ChatMessage, ChatTool,
        ChatToolCall,
    },
};
pub use generation::GenerationOutput;
pub use memory::{MemorySnapshot, ModelMemoryEstimate};
#[cfg(feature = "metal")]
pub use metal::{
    DenseBatchMode, FeatureToggle, FusionMode, MetalBatchConfig, MetalCacheConfig, MetalConfig,
    MetalDiagnosticsConfig, MetalFusionConfig,
};
pub use model::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, BackendAdmissionReport,
    CheckpointEncoding, IMAGE_PLACEHOLDER, Library, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION, Model,
    ModelDescriptor, ModelLoadOptions, PreparedPrompt, PreparedVisionPrompt, RemoteModelContract,
    RemoteTaskMetadata, RemoteVisionContract, WeightEncoding,
};
pub use models::{
    self,
    execution::{ArchitectureCapability, ArchitectureRequirements},
    generation::{GenerationChannel, GenerationOverrides, GenerationSettings, GenerationToken},
};
pub use rerank::{RerankOutput, RerankRequest, RerankResult};
pub use runtime::{
    self,
    backend::{
        BackendInfo, DecodeOutput, DecodeTimings, ModelHandle, PrefillOutput, SamplingLogits,
    },
    kv::{CacheStats, KvCacheDType},
    metrics::GenerationMetrics,
    progress::{ProgressEvent, ProgressStage, ProgressUnit},
};
pub use session::Session;
pub use telemetry::DeviceTelemetrySnapshot;
