//! The rotary encoding a layer declares, with its frequency scaling.

use super::super::{PositionEncodingSpec, RopeScalingSpec, RotaryLayoutSpec, RotarySpec};
use crate::layout::{AttentionLayerType, DecoderConfig, RopeScaling, RotaryEmbeddingLayout};

pub(in crate::semantic) fn rotary(decoder: &DecoderConfig, index: usize) -> PositionEncodingSpec {
    PositionEncodingSpec::Rotary(RotarySpec {
        theta: decoder.rope_theta_for_layer(index).unwrap_or(10_000.0),
        partial_factor: decoder.partial_rotary_factor_for_layer(index).unwrap_or(1.0),
        layout: match &decoder.rope_layout {
            RotaryEmbeddingLayout::Standard => RotaryLayoutSpec::Standard,
            RotaryEmbeddingLayout::MultiSection(sections) => {
                RotaryLayoutSpec::MultiSection(sections.clone())
            },
            RotaryEmbeddingLayout::InterleavedMultiSection(sections) => {
                RotaryLayoutSpec::InterleavedMultiSection(sections.clone())
            },
        },
        algorithm: decoder.rope_type_for_layer(index).unwrap_or_default(),
        scaling: if decoder.sliding_attention_rope_unscaled
            && decoder.layer_type(index) == AttentionLayerType::Sliding
        {
            None
        } else {
            decoder.rope_scaling.map(rope_scaling)
        },
    })
}

fn rope_scaling(scaling: RopeScaling) -> RopeScalingSpec {
    match scaling {
        RopeScaling::Linear { factor } => RopeScalingSpec::Linear { factor },
        RopeScaling::PiecewiseFrequency {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context_len,
        } => RopeScalingSpec::PiecewiseFrequency {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context_len,
        },
        RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            attention_factor,
            truncate,
        } => RopeScalingSpec::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context_len,
            attention_factor,
            truncate,
        },
        RopeScaling::LongRope {
            factors,
            original_context_len,
            attention_factor,
        } => RopeScalingSpec::LongRope {
            factors,
            original_context_len,
            attention_factor,
        },
    }
}
