//! Hardware-free model architecture admission shared by libmir backends.
//!
//! Each backend lowers a semantic spec into typed operations, derives what
//! the plan requires, and admits the highest-priority runtime whose declared
//! capabilities cover it; the loaders build from that plan.

mod set;

pub mod cache;
pub mod capability;
pub mod cuda;
pub mod decoder;
mod error;
pub mod ggml;
mod inspected;
pub mod lowering;
pub mod metal;
pub mod physical;

pub use cache::{
    CUDA_SLIDING_SLOTS_PER_SESSION, CacheLayout, LayerCacheEntry, LayerCacheKind,
    RecurrentCacheKind,
};
pub use error::{
    ArchitectureError, Constant, Rejection, Result, Runtime, TaskRefusal, Unlowerable,
};
pub use inspected::InspectedModel;
pub use physical::{
    AdmissionCheck, AdmissionCheckKind, AdmissionStatus, ArchitectureOutcome,
    BackendAdmissionReport, CheckSubject, CheckpointEncoding, CudaClampedExpertLayout,
    EncodingKind, MODEL_FORMAT_REGISTRY_SCHEMA_VERSION, WeightEncoding,
};
pub use set::{Flag, Set};

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;
