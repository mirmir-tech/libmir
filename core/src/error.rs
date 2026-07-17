use thiserror::Error;

pub type Result<T> = std::result::Result<T, MirmirError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Config,
    Backend,
    Model,
    Protocol,
    Runtime,
}

#[derive(Debug, Error)]
pub enum MirmirError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl MirmirError {
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Config(_) => ErrorKind::Config,
            Self::Backend(_) => ErrorKind::Backend,
            Self::Model(_) => ErrorKind::Model,
            Self::Protocol(_) => ErrorKind::Protocol,
            Self::Runtime(_) => ErrorKind::Runtime,
        }
    }
}
