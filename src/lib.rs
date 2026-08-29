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
pub use cuda;
pub use embedding::{EmbeddingOutput, EmbeddingRequest};
pub use engine::Engine;
pub use error::{Error, Result};
pub use foundation::{
    self,
    conversation::{
        Conversation, FunctionCall, FunctionDefinition, Message, Tool, ToolCall, ToolChoice,
    },
};
pub use generation::{GenerationOutput, GenerationRequest};
pub use memory::{MemorySnapshot, ModelMemoryEstimate};
#[cfg(feature = "metal")]
pub use metal;
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
