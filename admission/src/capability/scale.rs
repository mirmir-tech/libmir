//! The constant factors a decoder applies to its embeddings and attention
//! scores, told apart by what a kernel either hard-codes or reads.

use crate::set::{flag_enum, product_flag};

flag_enum! {
    /// The factor on the token embeddings.
    pub enum EmbeddingScale {
        Unit,
        /// `sqrt(hidden_size)` (Gemma's).
        SqrtHidden,
        Other,
    }
}

flag_enum! {
    /// The factor on one attention layer's scores.
    pub enum AttentionScaleValue {
        /// `1 / sqrt(head_dim)`.
        InverseSqrtHeadDim,
        Unit,
        Other,
    }
}

flag_enum! {
    /// Where an attention scale comes from.
    pub enum ScaleSource {
        /// The configuration's scale, or the default without one.
        Configured,
        /// A spec override that only a spec-driven kernel honors.
        Override,
    }
}

/// One attention layer's scale and its source.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttentionScale {
    pub value: AttentionScaleValue,
    pub source: ScaleSource,
}

product_flag!(AttentionScale {
    value: AttentionScaleValue,
    source: ScaleSource
});

impl EmbeddingScale {
    /// The kind of `multiplier` for a decoder `sqrt_hidden` wide.
    #[allow(clippy::float_cmp, reason = "the identity factors are exact constants")]
    #[must_use]
    pub fn of(multiplier: f64, sqrt_hidden: f64) -> Self {
        if multiplier == 1.0 {
            Self::Unit
        } else if multiplier == sqrt_hidden {
            Self::SqrtHidden
        } else {
            Self::Other
        }
    }
}

impl AttentionScale {
    /// The kind of `scale` against the head's `default` and the
    /// configuration's `declared` scale.
    #[allow(clippy::float_cmp, reason = "the identity factors are exact constants")]
    #[must_use]
    pub fn of(scale: f64, default: f64, declared: Option<f64>) -> Self {
        let value = if scale == default {
            AttentionScaleValue::InverseSqrtHeadDim
        } else if scale == 1.0 {
            AttentionScaleValue::Unit
        } else {
            AttentionScaleValue::Other
        };
        let source = if scale == declared.unwrap_or(default) {
            ScaleSource::Configured
        } else {
            ScaleSource::Override
        };
        Self { value, source }
    }
}
