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
pub use embedding::{EmbeddingOutput, EmbeddingRequest};
pub use engine::Engine;
pub use error::{Error, Result};
pub use foundation::{
    conversation::{
        Conversation, FunctionCall, FunctionDefinition, Message, Tool, ToolCall, ToolChoice,
    },
    model::BackendTarget,
};
pub use generation::{GenerationOutput, GenerationRequest};
pub use memory::{MemorySnapshot, ModelMemoryEstimate};
pub use model::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, BackendAdmissionReport,
    CheckpointEncoding, IMAGE_PLACEHOLDER, Library, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION, Model,
    ModelDescriptor, ModelLoadOptions, PreparedPrompt, PreparedVisionPrompt, RemoteModelContract,
    RemoteTaskMetadata, RemoteVisionContract, WeightEncoding,
};
pub use models::{
    chat::TemplateKind,
    execution::{
        ArchitectureCapability, ArchitectureRequirements, EmbeddingTask, ModelTask, PoolingMode,
        TaskExecutionPlan,
    },
    generation::{GenerationChannel, GenerationOverrides, GenerationSettings, GenerationToken},
    tokenizer::{TokenizerAssets, TokenizerKind},
    weights::{TensorCatalog, TensorInfo, safetensors_header_len},
};
pub use rerank::{RerankOutput, RerankRequest, RerankResult};
pub use runtime::{
    RuntimeError,
    backend::{
        BackendInfo, DecodeOutput, DecodeTimings, ModelHandle, PrefillOutput, SamplingLogits,
    },
    kv::{CacheStats, KvCacheDType},
    metrics::GenerationMetrics,
    progress::{ProgressEvent, ProgressStage, ProgressUnit},
};
pub use session::Session;
pub use telemetry::DeviceTelemetrySnapshot;
