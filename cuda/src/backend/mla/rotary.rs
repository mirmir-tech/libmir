//! Rotary frequencies of the latent attention's rotary part, computed once on
//! the host (`PositionEncodingSpec::inverse_frequencies`); the kernels read
//! them per pair.

use models::semantic::PositionEncodingSpec;

use crate::{Error, Result};

/// Inverse frequencies of the `rope_dim / 2` pairs and the factor on their
/// cosine and sine.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn frequencies(
    position: &PositionEncodingSpec,
    rope_dim: usize,
) -> Result<(Vec<f32>, f32)> {
    let (inverse, scale) = position.inverse_frequencies(rope_dim)?.ok_or_else(|| {
        Error::UnsupportedDecoderLayer("CUDA has no latent attention without a rotary part".into())
    })?;
    Ok((inverse.into_iter().map(|value| value as f32).collect(), scale as f32))
}
