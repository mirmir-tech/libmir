#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

mod cancellation;
mod config;
mod engine;
mod error;
mod generation;
mod memory;
mod model;
mod scheduler;
mod session;

pub use cancellation::CancellationToken;
pub use config::RuntimeConfig;
#[cfg(feature = "cuda")]
pub use cuda::{
    AffineQuantizedBf16Linear, AffineQuantizedBf16Qmm, AffineQuantizedConfig,
    AffineQuantizedPairTensors, AffineQuantizedTensors, Bf16Embedding, Bf16Linear,
    BucketedNvFp4MoeBf16, CudaAttentionPolicy, CudaBackend, CudaConfig, CudaDenseVectorPolicy,
    CudaDenseWeightPolicy, CudaExecutionPlanner, CudaHardwareProfile, CudaKernelAdmission,
    CudaMemoryArchitecture, CudaModelSessionConfig, CudaMoeBatchPolicy, CudaMoeFusionPolicy,
    CudaMoeModelSession, CudaMoeModelTemplate, CudaNumericalPolicy, CudaOutputHeadPolicy,
    CudaPlanningPolicy, CudaTensor, CudaTensorDType, CudaTensorSet, DecodeMoeLayerTemplate,
    DenseExecution, DensePlan, DensePlanRequest, DenseRole, DeviceSamplerBf16, Error as CudaError,
    ExecutionPhase, GatedActivation, GroupedNvFp4MoeBf16, NvFp4MoeLayerLoadConfig, PlanSource,
    PrefillAttentionBf16, PrefillMoeBlockBf16, SelectedAffineGatedBf16Linear,
    SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear, TensorUploadBatch,
};
pub use engine::Engine;
pub use error::{Error, Result};
pub use foundation::{
    self,
    protocol::{ChatCompletionRequest, ChatMessage},
};
pub use generation::GenerationOutput;
pub use memory::{MemorySnapshot, ModelMemoryEstimate};
#[cfg(feature = "metal")]
pub use metal::{
    DenseBatchMode, FeatureToggle, FusionMode, MetalBatchConfig, MetalCacheConfig, MetalConfig,
    MetalDiagnosticsConfig, MetalFusionConfig,
};
pub use model::{Library, Model, ModelDescriptor, PreparedPrompt};
pub use models::{
    self,
    generation::{GenerationChannel, GenerationOverrides, GenerationSettings, GenerationToken},
};
pub use runtime::{
    self,
    backend::{BackendInfo, DecodeOutput, ModelHandle, PrefillOutput, SamplingLogits},
    kv::{CacheStats, KvCacheDType},
    metrics::GenerationMetrics,
    progress::{ProgressEvent, ProgressStage, ProgressUnit},
};
pub use session::Session;
