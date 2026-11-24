//! Output-row selection on a projection, applied once at load: GGML and
//! affine rows are independent blocks of their own scales, and a dense weight
//! is stored transposed, so each reorders without dequantizing.

use super::BoundLinear;
use crate::engine::{Array, Error, Result, Stream};

impl BoundLinear {
    /// The projection whose output row `i` is this one's row `rows[i]`.
    pub(in crate::engine) fn take_rows(&self, rows: &Array, stream: &Stream) -> Result<Self> {
        match self {
            Self::Ggml(linear) => linear.take_rows(rows, stream).map(Self::Ggml),
            Self::Dense(linear) => linear.take_output_rows(rows, stream).map(Self::Dense),
            Self::Affine(linear) => linear.take_output_rows(rows, stream).map(Self::Affine),
            _ => Err(Error::InvalidModel(
                "output rows reorder only GGML, affine and dense projections".into(),
            )),
        }
    }
}
