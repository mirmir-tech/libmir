use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Models(#[from] models::ModelsError),
    #[error(transparent)]
    Tensor(#[from] mircup::Error),
    #[error("cannot read checkpoint weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("tensor size does not fit this platform: {0}")]
    Size(#[from] std::num::TryFromIntError),
    #[error("tensor `{name}` has unsupported dtype {dtype}")]
    DType { name: String, dtype: String },
    #[error("tensor `{0}` is not a matrix")]
    NotMatrix(String),
    #[error("decision batch is empty")]
    EmptyBatch,
    #[error("decision row of {tokens} tokens exceeds the rotary table of {positions} positions")]
    RowTooLong { tokens: usize, positions: usize },
}
