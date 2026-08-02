use std::sync::{PoisonError, mpsc};

use thiserror::Error;

use crate::engine;

pub(super) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
#[error("{0}")]
pub(super) struct WorkerFailure(String);

impl From<Error> for WorkerFailure {
    fn from(value: Error) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Error)]
pub(super) enum Error {
    #[error("native MLX FFI error: {0}")]
    Engine(#[from] engine::Error),
    #[error("lock poisoned: {0}")]
    Lock(String),
    #[error("integer conversion failed: {0}")]
    Integer(#[from] std::num::TryFromIntError),
    #[cfg(test)]
    #[error("integer parsing failed: {0}")]
    ParseInteger(#[from] std::num::ParseIntError),
    #[cfg(test)]
    #[error("benchmark output failed: {0}")]
    BenchmarkOutput(#[from] std::io::Error),
    #[error("model is not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("model configuration error: {0}")]
    Models(#[from] models::ModelsError),
    #[error("native model cannot prefill an empty token sequence")]
    EmptyPrompt,
    #[error("model {model} has no active session {session}")]
    Session { model: String, session: uuid::Uuid },
    #[error("native MLX has no scheduled greedy decode")]
    NoPendingDecode,
    #[error("native MLX prefix snapshot is missing prompt logits")]
    NoPrefixLogits,
    #[error("scheduled MLX token {expected} does not match requested token {actual}")]
    PendingToken { expected: u32, actual: u32 },
    #[error("invalid native MLX decode batch: {0}")]
    InvalidDecodeBatch(String),
    #[error("invalid native MLX prefill batch: {0}")]
    InvalidPrefillBatch(String),
    #[error("native MLX worker failed: {0}")]
    Worker(#[from] WorkerFailure),
    #[error("native MLX worker channel closed")]
    WorkerClosed,
    #[error("native MLX worker channel receive failed: {0}")]
    WorkerReceive(#[from] mpsc::RecvError),
    #[error("native MLX worker thread failed to start: {0}")]
    WorkerSpawn(std::io::Error),
    #[error("native MLX worker thread panicked while unloading the model")]
    WorkerJoin,
    #[cfg(test)]
    #[error("native MLX benchmark configuration error: {0}")]
    Benchmark(String),
    #[error("native MLX does not support this model: {0}")]
    UnsupportedModel(String),
}

impl<T> From<PoisonError<T>> for Error {
    fn from(value: PoisonError<T>) -> Self {
        Self::Lock(value.to_string())
    }
}

impl From<Error> for runtime::RuntimeError {
    fn from(value: Error) -> Self {
        match value {
            Error::ModelNotLoaded(id) => Self::ModelNotLoaded(id),
            Error::Models(error) => Self::ModelLoad(error.to_string()),
            Error::UnsupportedModel(error) => Self::ModelLoad(error),
            Error::Engine(error) => Self::Backend(error.to_string()),
            Error::Lock(error) => Self::Backend(error),
            Error::Integer(error) => Self::Integer(error),
            #[cfg(test)]
            Error::ParseInteger(error) => Self::Backend(error.to_string()),
            #[cfg(test)]
            Error::BenchmarkOutput(error) => Self::Backend(error.to_string()),
            #[cfg(test)]
            Error::Benchmark(error) => Self::Backend(error),
            Error::EmptyPrompt
            | Error::Session { .. }
            | Error::NoPendingDecode
            | Error::NoPrefixLogits
            | Error::PendingToken { .. }
            | Error::InvalidDecodeBatch(_)
            | Error::InvalidPrefillBatch(_)
            | Error::Worker(_)
            | Error::WorkerClosed
            | Error::WorkerReceive(_)
            | Error::WorkerSpawn(_)
            | Error::WorkerJoin => Self::Backend(value.to_string()),
        }
    }
}
