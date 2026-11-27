//! Rotary position encoding of `[rows, sequence, heads, head_dim]`
//! projections into the `[rows, heads, sequence, head_dim]` attention layout.

use super::super::config::DenseSwiGluLayerConfig;
use crate::engine::{Array, Result, RopeOptions, Stream};

pub(in crate::engine::dense::swiglu) fn rope_layout(
    input: &Array,
    frequencies: Option<&Array>,
    config: DenseSwiGluLayerConfig,
    position: i32,
    stream: &Stream,
) -> Result<Array> {
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    if config.rope_dimensions == 0 {
        return Ok(input);
    }
    let rotated = frequencies.map_or_else(
        || {
            input.rope(
                RopeOptions {
                    dimensions: config.rope_dimensions,
                    traditional: config.rope_traditional,
                    base: Some(config.rope_base),
                    scale: config.rope_position_scale,
                    offset: position,
                },
                stream,
            )
        },
        |frequencies| {
            input.rope_with_frequencies(
                config.rope_dimensions,
                config.rope_traditional,
                frequencies,
                position,
                stream,
            )
        },
    )?;
    let factor = config.rope_attention_factor;
    if factor.to_bits() == 1.0_f32.to_bits() {
        return Ok(rotated);
    }
    let head = usize::try_from(config.head_dim)?;
    let dimensions = usize::try_from(config.rope_dimensions)?;
    if dimensions >= head {
        return rotated.multiply_scalar(factor, stream);
    }
    // LongRoPE's and YaRN's factor scales the rotated pairs only (MLX-LM
    // `SuScaledRoPE`, `YarnRoPE`), not a partial rotation's pass-through.
    let scales = (0..head)
        .map(|index| {
            if index < dimensions {
                factor
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let scales = Array::from_f32(&scales, &[config.head_dim])?.astype_like(&rotated, stream)?;
    rotated.multiply(&scales, stream)
}

#[cfg(test)]
mod tests;
