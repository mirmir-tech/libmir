mod admission;
mod config;
pub mod engine;
mod native;
mod progress;

pub use admission::{MetalArchitecture, admit_architecture};
pub use config::{
    DenseBatchMode, FeatureToggle, FusionMode, MetalBatchConfig, MetalCacheConfig, MetalConfig,
    MetalDiagnosticsConfig, MetalFusionConfig, MetalTuningConfig, MetalTuningMode,
};
pub use native::{
    MetalBackend, MetalGenerationStepOutput, MetalMemoryStats, MetalPrefillBatch,
    MetalPrefillSchedule,
};
pub use progress::{MetalProgressEvent, MetalProgressStage, MetalProgressUnit};
