use super::GatedFullAttentionConfig;
use crate::engine::{Array, Result, RopeOptions, Stream, attention::apply_mrope};

pub(super) fn rope(
    input: &Array,
    config: &GatedFullAttentionConfig,
    position: i32,
    positions: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    if let Some(positions) = positions {
        return apply_mrope(
            &input,
            positions,
            usize::try_from(config.rope_dimensions)?,
            config.rope_base,
            &config.rope_layout,
            stream,
        );
    }
    input.rope(
        RopeOptions {
            dimensions: config.rope_dimensions,
            traditional: false,
            base: Some(config.rope_base),
            scale: 1.0,
            offset: position,
        },
        stream,
    )
}
