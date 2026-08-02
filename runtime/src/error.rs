use foundation::MirmirError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("backend is not available: {0}")]
    BackendUnavailable(String),
    #[error("backend failed: {0}")]
    Backend(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("integer conversion failed: {0}")]
    Integer(#[from] std::num::TryFromIntError),
    #[error("model is not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("scheduler rejected request: {0}")]
    Scheduler(String),
    #[error("kv cache error: {0}")]
    KvCache(String),
    #[error("kv cache is waiting for active sessions to release blocks")]
    KvCachePressure,
}

impl From<RuntimeError> for MirmirError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value.to_string())
    }
}
