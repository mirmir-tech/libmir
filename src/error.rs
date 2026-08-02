use std::path::PathBuf;

use thiserror::Error;

/// Result type returned by the high-level libmir API.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
/// Error produced while inspecting, loading, or running a model.
pub enum Error {
    #[error("generation was cancelled")]
    /// Generation stopped after its cancellation token was signalled.
    Cancelled,
    #[cfg(feature = "cuda")]
    #[error("CUDA backend error: {0}")]
    /// The CUDA backend rejected an operation.
    Cuda(#[from] cuda::Error),
    #[error("model error: {0}")]
    /// Model metadata, tokenizer, or template processing failed.
    Model(#[from] models::ModelsError),
    #[error("runtime error: {0}")]
    /// Backend-neutral runtime or cache processing failed.
    Runtime(#[from] runtime::RuntimeError),
    #[error("model path has no usable identifier: {0}")]
    /// A stable model identifier could not be derived from this path.
    ModelId(PathBuf),
    #[error("required environment variable is not configured: {0}")]
    /// A required backend environment variable was not configured.
    MissingEnvironment(&'static str),
    #[error("tokenized prompt cannot be empty")]
    /// Prompt rendering produced no input tokens.
    EmptyPrompt,
    #[error("model does not support the requested {requested} task; discovered {actual}")]
    /// The caller selected an operation not exposed by the checkpoint contract.
    TaskMismatch {
        /// User-facing operation name.
        requested: &'static str,
        /// Discovered model task.
        actual: &'static str,
    },
    #[error("model is currently serving a request")]
    /// An unload was requested while sessions or model clones still exist.
    ModelInUse,
    #[error(
        "model `{model}` needs an estimated {required_bytes} bytes, but only {available_bytes} bytes remain in the accelerator load budget"
    )]
    /// A model load could not reserve enough accelerator memory atomically.
    MemoryAdmission {
        /// Stable model identifier.
        model: String,
        /// Estimated peak residency, including transient execution headroom.
        required_bytes: u64,
        /// Bytes still available after safety and in-flight load reservations.
        available_bytes: u64,
    },
    #[error(
        "requested {requested} tokens exceeds model context {context} (prompt {prompt}, max_tokens {max_tokens})"
    )]
    /// Prompt and requested output exceed the model context window.
    Context {
        /// Total number of tokens requested.
        requested: usize,
        /// Maximum context length supported by the model.
        context: usize,
        /// Number of tokens in the prepared prompt.
        prompt: usize,
        /// Maximum number of output tokens requested.
        max_tokens: usize,
    },
    #[error(
        "vision input needs an estimated {required_bytes} byte attention buffer for {patch_tokens} patches, exceeding the configured {budget_bytes} byte budget; reduce image dimensions or raise the vision limit"
    )]
    /// Vision preprocessing could not satisfy the configured resource budget.
    VisionResourceLimit {
        /// Number of patch tokens presented to the vision transformer.
        patch_tokens: usize,
        /// Conservative size of one full attention score matrix.
        required_bytes: u64,
        /// Effective runtime attention budget.
        budget_bytes: u64,
    },
}
