use mirtal::{DType, Graph, Shape};
use models::semantic::{AttentionSpec, PositionEncodingSpec, RopeScalingSpec, RotarySpec};

use super::super::{Array, Error, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub struct RopeOptions {
    pub dimensions: i32,
    pub traditional: bool,
    pub base: Option<f32>,
    pub scale: f32,
    pub offset: i32,
}

impl Array {
    pub(crate) fn ntk_rope_frequencies(
        rotary_dimensions: i32,
        base: f32,
        factor: f32,
        stream: &Stream,
    ) -> Result<Self> {
        dimensions(rotary_dimensions)?;
        positive(base, "base")?;
        positive(factor, "factor")?;
        let graph = stream.native().graph();
        let dimensions = as_f32(rotary_dimensions)?;
        let exponents = graph.arange(0.0, dimensions, 2.0, DType::Float32)?;
        let exponents = graph.divide(&exponents, &scalar(graph, dimensions)?)?;
        let adjusted_base = graph.multiply(&scalar(graph, base)?, &scalar(graph, factor)?)?;
        let frequencies = graph.power(&adjusted_base, &exponents)?;
        let correction = factor.powf(2.0 / dimensions);
        Self::from_native(graph.multiply(&frequencies, &scalar(graph, correction)?)?)
    }

    /// MLX `freqs` (periods, the reciprocal inverse frequencies) of the
    /// attention's pairing width, from the semantic spec's scaling; pairs past
    /// the rotated ones get an infinite period and pass through (proportional
    /// `RoPE`). `None` when the fused kernel generates them from its `base`
    /// and `rope_position_scale`, as MLX-LM's `nn.RoPE` does for unscaled and
    /// linearly scaled rotations. The kernel rotates
    /// `RotarySpec::pairing_dimensions` dimensions.
    pub(crate) fn rope_periods(attention: &AttentionSpec) -> Result<Option<Self>> {
        let PositionEncodingSpec::Rotary(rotary) = &attention.position else {
            return Ok(None);
        };
        let rotated = rotary.rotated_dimensions(attention.head_dim)?;
        let pairing = rotary.pairing_dimensions(attention.head_dim)?;
        if fused(rotary, attention.head_dim)? {
            return Ok(None);
        }
        let Some((inverse, _)) = attention.position.inverse_frequencies(pairing)? else {
            return Ok(None);
        };
        let periods = inverse
            .iter()
            .enumerate()
            .map(|(pair, frequency)| {
                if pair < rotated / 2 {
                    Ok(frequency.recip().to_string().parse::<f32>()?)
                } else {
                    Ok(f32::INFINITY)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_f32(&periods, &[i32::try_from(periods.len())?]).map(Some)
    }

    pub fn rope(&self, options: RopeOptions, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().rope(
            self.native(),
            mirtal::RopeOptions {
                dimensions: usize::try_from(options.dimensions)?,
                traditional: options.traditional,
                base: options.base,
                scale: options.scale,
                offset: usize::try_from(options.offset)?,
            },
        )?)
    }

    pub fn rope_with_frequencies(
        &self,
        dimensions: i32,
        traditional: bool,
        frequencies: &Self,
        offset: i32,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().rope_with_frequencies(
            self.native(),
            frequencies.native(),
            mirtal::FrequencyRopeOptions {
                dimensions: usize::try_from(dimensions)?,
                traditional,
                offset: usize::try_from(offset)?,
            },
        )?)?
        .astype_like(self, stream)
    }
}

/// The fused kernel's position scale: `1 / factor` under linear scaling when
/// it generates the frequencies itself, else 1.
pub fn rope_position_scale(attention: &AttentionSpec) -> Result<f32> {
    match &attention.position {
        PositionEncodingSpec::Rotary(rotary) if fused(rotary, attention.head_dim)? => {
            match rotary.scaling {
                Some(RopeScalingSpec::Linear { factor }) => {
                    Ok(factor.recip().to_string().parse()?)
                },
                _ => Ok(1.0),
            }
        },
        PositionEncodingSpec::Rotary(_) | PositionEncodingSpec::None => Ok(1.0),
    }
}

/// Whether the fused kernel's `base` and position scale express the rotation.
fn fused(rotary: &RotarySpec, head_dim: usize) -> Result<bool> {
    Ok(matches!(rotary.scaling, None | Some(RopeScalingSpec::Linear { .. }))
        && rotary.pairing_dimensions(head_dim)? == rotary.rotated_dimensions(head_dim)?)
}

fn scalar(graph: Graph<'_>, value: f32) -> Result<mirtal::Array> {
    Ok(graph.full(&Shape::new([])?, value, DType::Float32)?)
}

fn dimensions(value: i32) -> Result<()> {
    if value <= 0 || value % 2 != 0 {
        return Err(invalid("RoPE dimensions must be positive and even"));
    }
    Ok(())
}

fn positive(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{name} must be finite and positive")));
    }
    Ok(())
}

fn as_f32(value: i32) -> Result<f32> {
    Ok(f32::from(u16::try_from(value)?))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}
