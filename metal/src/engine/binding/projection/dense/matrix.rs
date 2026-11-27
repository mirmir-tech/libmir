//! A projection's weight as a dense `[output, input]` matrix, for layers
//! that fold weights into other products at load (latent attention's key
//! and value up-projections).

use super::super::BoundLinear;
use crate::engine::{Array, Error, Result, Stream};

impl BoundLinear {
    /// The dense `[output, input]` weight; a bank's matrices are stacked.
    pub(in crate::engine) fn dense_matrix(&self, stream: &Stream) -> Result<Array> {
        match self {
            Self::Dense(linear) => linear.weight(stream),
            Self::Ggml(linear) => linear.dense(stream),
            Self::Affine(linear) => linear.dense(stream),
            _ => Err(Error::InvalidQuantization(
                "dense weights come only from dense, GGML and affine projections".into(),
            )),
        }
    }
}
