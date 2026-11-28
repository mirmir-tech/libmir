use std::path::PathBuf;

use foundation::model::ModelId;
use models::execution::TaskKind;
use thiserror::Error;

/// Result type returned by the high-level libmir API.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Typed reason a model's generated tool call output was rejected.
pub enum ToolCallError {
    #[error("required tool call was not emitted")]
    /// The caller required a tool call, but the model emitted none.
    MissingRequired,
    #[error("tool-call output is empty")]
    /// The parsed tool-call array contained no calls.
    Empty,
    #[error("tool calls were disabled")]
    /// Tool calling was disabled, yet the model emitted a tool call.
    Disabled,
    #[error("model called `{actual}` instead of requested tool `{expected}`")]
    /// The caller forced one function name, but the model called another.
    WrongFunction {
        /// Function name forced by the caller.
        expected: String,
        /// Function name emitted by the model.
        actual: String,
    },
    #[error("model called undeclared tool `{0}`")]
    /// The model emitted a function name not present in the request tools.
    UndeclaredFunction(String),
    #[error("schema-constrained tool generation ended before completion")]
    /// Generation stopped before closing the constrained tool grammar.
    Incomplete,
    #[error("tool arguments violate schema at {instance_path} (constraint {schema_path})")]
    /// Tool call arguments failed post-generation JSON Schema validation.
    SchemaViolation {
        /// JSON Pointer to the invalid argument value.
        instance_path: String,
        /// JSON Pointer to the violated schema keyword.
        schema_path: String,
    },
    #[error("{0}")]
    /// Tool call envelope or argument payload was syntactically malformed.
    Malformed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Typed reason generated content violated a requested JSON response format.
pub enum ResponseFormatError {
    #[error("JSON content ended before its schema was complete")]
    /// Generation stopped before closing the constrained JSON grammar.
    Incomplete,
    #[error("content violates schema at {instance_path} (constraint {schema_path})")]
    /// Content failed post-generation JSON Schema validation.
    SchemaViolation {
        /// JSON Pointer to the invalid value.
        instance_path: String,
        /// JSON Pointer to the violated schema keyword.
        schema_path: String,
    },
    #[error("{0}")]
    /// Content was not valid JSON for the requested response format.
    Malformed(String),
}

#[derive(Debug, Error)]
/// Error produced while inspecting, loading, or running a model.
pub enum Error {
    #[error("invalid model tool call: {0}")]
    /// Generated tool output was malformed or violated the requested tool
    /// choice.
    InvalidToolCall(#[from] ToolCallError),
    #[error("invalid model response format output: {0}")]
    /// Generated content violated the requested JSON response format.
    InvalidResponseFormat(#[from] ResponseFormatError),
    #[error("generation was cancelled")]
    /// Generation stopped after its cancellation token was signalled.
    Cancelled,
    #[error("model architecture admission failed: {0}")]
    /// The selected backend cannot lower or execute the inspected architecture.
    Admission(#[from] architecture::ArchitectureError),
    #[cfg(feature = "cuda")]
    #[error("CUDA backend error: {0}")]
    /// The CUDA backend rejected an operation.
    Cuda(#[from] cuda::Error),
    #[cfg(feature = "cpu")]
    #[error("CPU backend error: {0}")]
    /// The CPU backend rejected an operation.
    Cpu(#[from] cpu::Error),
    #[cfg(feature = "metal")]
    #[error("Metal decision backend error: {0}")]
    /// The Metal backend rejected a decision operation.
    Metal(#[from] metal::engine::Error),
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
        /// Requested operation task kind.
        requested: TaskKind,
        /// Discovered checkpoint task kind.
        actual: TaskKind,
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
        model: ModelId,
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
