use std::fmt;

pub type Result<T> = std::result::Result<T, ArchitectureError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchitectureError {
    detail: String,
}

impl ArchitectureError {
    pub(crate) fn invalid(detail: impl Into<String>) -> Self {
        Self { detail: detail.into() }
    }
}

impl fmt::Display for ArchitectureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for ArchitectureError {}
