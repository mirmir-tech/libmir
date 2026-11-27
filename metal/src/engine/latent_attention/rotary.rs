//! The rotary part of latent attention: explicit (`YaRN`) frequencies with
//! their attention factor, and the optional per-position query temperature.

use models::semantic::{LatentAttentionSpec, QueryTemperature, RotaryPairingSpec};

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug)]
pub(super) struct LatentRotary {
    /// MLX's `freqs`: the wavelength of every pair, `1 / inverse frequency`.
    pub(super) wavelengths: Array,
    pub(super) dimensions: i32,
    pub(super) traditional: bool,
    /// `YaRN`'s factor on the rotated values.
    pub(super) factor: f32,
    pub(super) temperature: Option<QueryTemperature>,
}

impl LatentRotary {
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn new(spec: &LatentAttentionSpec) -> Result<Self> {
        let (inverse, factor) = spec
            .position
            .inverse_frequencies(spec.rope_head_dim)?
            .ok_or_else(|| Error::InvalidModel("latent attention without a rotary part".into()))?;
        let wavelengths = inverse.iter().map(|value| value.recip() as f32).collect::<Vec<_>>();
        Ok(Self {
            wavelengths: Array::from_f32(&wavelengths, &[i32::try_from(wavelengths.len())?])?,
            dimensions: i32::try_from(spec.rope_head_dim)?,
            traditional: spec.rotary_pairing == RotaryPairingSpec::Interleaved,
            factor: factor as f32,
            temperature: spec.query_temperature,
        })
    }

    /// Rotates `[.., sequence, rope]` values starting at `position`.
    pub(super) fn apply(&self, input: &Array, position: i32, stream: &Stream) -> Result<Array> {
        let input = if (self.factor - 1.0).abs() > f32::EPSILON {
            input.multiply_scalar(self.factor, stream)?
        } else {
            input.astype_like(input, stream)?
        };
        input.rope_with_frequencies(
            self.dimensions,
            self.traditional,
            &self.wavelengths,
            position,
            stream,
        )
    }

    /// Queries `[.., heads, sequence, width]` at their temperature.
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn temper(
        &self,
        queries: &Array,
        position: i32,
        sequence: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let Some(temperature) = self.temperature else {
            return Array::from_native(queries.native().clone());
        };
        let first = usize::try_from(position)?;
        let factors = (0..usize::try_from(sequence)?)
            .map(|offset| temperature.factor(first + offset) as f32)
            .collect::<Vec<_>>();
        let factors =
            Array::from_f32(&factors, &[1, 1, sequence, 1])?.astype_like(queries, stream)?;
        queries.multiply(&factors, stream)
    }
}
