/// Result returned by the native CUDA adapter.
pub type Result<T> = std::result::Result<T, Error>;

/// CUDA adapter initialization or execution failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Mircuda rejected the native operation.
    #[error(transparent)]
    Native(#[from] mircuda::Error),
    /// Model tensor metadata is invalid.
    #[error(transparent)]
    Model(#[from] models::ModelsError),
    /// Backend-neutral session or K/V planning failed.
    #[error(transparent)]
    Runtime(#[from] runtime::RuntimeError),
    /// Checkpoint payload I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A checkpoint dimension or offset cannot be represented by the target
    /// ABI.
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
    /// The configured CUDA device ordinal is unavailable.
    #[error("CUDA device ordinal {0} is unavailable")]
    DeviceUnavailable(usize),
    /// Safetensors storage type is not implemented by the CUDA loader.
    #[error("unsupported CUDA tensor dtype {0}")]
    UnsupportedDType(String),
    /// Tensor shape and payload byte count disagree.
    #[error("invalid tensor payload size for {name}: expected {expected}, got {actual}")]
    InvalidTensorSize {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// A completed upload contains the same tensor name more than once.
    #[error("duplicate CUDA tensor: {0}")]
    DuplicateTensor(String),
    /// A requested CUDA tensor is not present in the completed set.
    #[error("CUDA tensor is missing: {0}")]
    MissingTensor(String),
    /// A diagnostic host read requested the wrong tensor storage type.
    #[error("CUDA tensor {name} is not {expected}")]
    DTypeMismatch { name: String, expected: &'static str },
    /// A dense projection received a weight with a different checkpoint shape.
    #[error("invalid linear weight {name}: expected {expected:?}, got {actual:?}")]
    InvalidLinearWeight {
        name: String,
        expected: [usize; 2],
        actual: Vec<usize>,
    },
    /// A packed affine tensor has a shape incompatible with its linear plan.
    #[error("invalid quantized tensor {name}: expected {expected:?}, got {actual:?}")]
    InvalidQuantizedTensor {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// An expert or matrix index is outside the uploaded tensor bank.
    #[error("quantized matrix index {index} exceeds matrix count {matrices}")]
    InvalidMatrixIndex { index: usize, matrices: usize },
    /// Model metadata names a gated activation without a native CUDA
    /// implementation.
    #[error("unsupported gated activation: {0}")]
    UnsupportedGatedActivation(String),
    /// An affine quantized GEMV shape or format is not representable.
    #[error("invalid affine quantized GEMV: {0}")]
    InvalidQuantizedGemv(&'static str),
    /// An NVFP4 checkpoint matrix violates the native block layout.
    #[error("invalid NVFP4 matrix: {0}")]
    InvalidNvFp4(&'static str),
    /// A normalization or rotary operation has invalid fixed geometry.
    #[error("invalid CUDA decoder kernel: {0}")]
    InvalidDecoderKernel(&'static str),
    /// The model-level CUDA execution planner rejected a request.
    #[error("invalid CUDA execution plan: {0}")]
    InvalidExecutionPlan(&'static str),
    /// A token identifier is outside the model vocabulary.
    #[error("token {token} exceeds CUDA model vocabulary {vocab}")]
    InvalidToken { token: u32, vocab: usize },
    /// Sampling policy cannot execute entirely on CUDA.
    #[error("invalid CUDA sampling policy: {0}")]
    InvalidSampling(String),
    /// Checkpoint metadata describes a decoder layer outside the native CUDA
    /// capability set.
    #[error("unsupported CUDA decoder layer: {0}")]
    UnsupportedDecoderLayer(String),
    /// The model/session registry cannot be accessed consistently.
    #[error("CUDA inference state failed: {0}")]
    State(String),
    /// Paged KV storage or attention received an incompatible geometry.
    #[error("invalid CUDA paged KV operation: {0}")]
    InvalidPagedKv(&'static str),
    /// Routed-expert selection received incompatible geometry or tensors.
    #[error("invalid CUDA router: {0}")]
    InvalidRouter(&'static str),
    /// A quantized GEMV buffer does not match the fixed execution shape.
    #[error("quantized GEMV {operand} length mismatch: expected at least {expected}, got {actual}")]
    QuantizedGemvLengthMismatch {
        operand: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl From<Error> for runtime::RuntimeError {
    fn from(value: Error) -> Self {
        Self::Backend(value.to_string())
    }
}
