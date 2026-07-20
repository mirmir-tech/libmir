use mirtal::{DType, Graph, Shape};

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

    pub(crate) fn proportional_rope_frequencies(
        head_dim: i32,
        rope_dimensions: i32,
        base: f32,
        stream: &Stream,
    ) -> Result<Self> {
        dimensions(head_dim)?;
        dimensions(rope_dimensions)?;
        if rope_dimensions > head_dim {
            return Err(invalid("RoPE dimensions exceed head dimensions"));
        }
        positive(base, "base")?;
        let graph = stream.native().graph();
        let exponents = graph.arange(0.0, as_f32(rope_dimensions)?, 2.0, DType::Float32)?;
        let exponents = graph.divide(&exponents, &scalar(graph, as_f32(head_dim)?)?)?;
        let frequencies = graph.power(&scalar(graph, base)?, &exponents)?;
        let remaining = usize::try_from(head_dim / 2 - rope_dimensions / 2)?;
        if remaining == 0 {
            return Self::from_native(frequencies);
        }
        let tail = graph.full(&Shape::new([remaining])?, f32::INFINITY, DType::Float32)?;
        Self::from_native(graph.concatenate(&[&frequencies, &tail], 0)?)
    }

    pub(crate) fn piecewise_rope_frequencies(
        rotary_dimensions: i32,
        base: f32,
        factor: f32,
        low_frequency_factor: f32,
        high_frequency_factor: f32,
        original_context_len: i32,
        stream: &Stream,
    ) -> Result<Self> {
        dimensions(rotary_dimensions)?;
        for (value, name) in [
            (base, "base"),
            (factor, "factor"),
            (low_frequency_factor, "low frequency factor"),
            (high_frequency_factor, "high frequency factor"),
        ] {
            positive(value, name)?;
        }
        if high_frequency_factor <= low_frequency_factor || original_context_len <= 0 {
            return Err(invalid("invalid piecewise RoPE interpolation range"));
        }
        let graph = stream.native().graph();
        let exponents = graph.arange(0.0, as_f32(rotary_dimensions)?, 2.0, DType::Float32)?;
        let exponents = graph.divide(&exponents, &scalar(graph, as_f32(rotary_dimensions)?)?)?;
        let frequencies = graph.power(&scalar(graph, base)?, &exponents)?;
        let inverse = graph.reciprocal(&frequencies)?;
        let wavelength = graph.multiply(&scalar(graph, std::f32::consts::TAU)?, &frequencies)?;
        let ratio = graph.divide(&scalar(graph, as_f32(original_context_len)?)?, &wavelength)?;
        let smooth = graph.subtract(&ratio, &scalar(graph, low_frequency_factor)?)?;
        let smooth =
            graph.divide(&smooth, &scalar(graph, high_frequency_factor - low_frequency_factor)?)?;
        let smooth = graph.maximum(&smooth, &scalar(graph, 0.0)?)?;
        let smooth = graph.minimum(&smooth, &scalar(graph, 1.0)?)?;
        let slowed = graph.divide(&inverse, &scalar(graph, factor)?)?;
        let inverse_smooth = graph.subtract(&scalar(graph, 1.0)?, &smooth)?;
        let adjusted = graph
            .add(&graph.multiply(&slowed, &inverse_smooth)?, &graph.multiply(&inverse, &smooth)?)?;
        Self::from_native(graph.reciprocal(&adjusted)?)
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
