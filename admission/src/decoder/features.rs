//! The decoder-wide properties a runtime either executes or refuses: the
//! dense-only list both backends used to check by hand, lowered once.

use models::semantic::{
    AttentionWindow, MixerSpec, NormalizationKind, PositionEncodingSpec, RopeScalingSpec,
    SemanticModelSpec,
};

use crate::{
    ArchitectureError, Result, Unlowerable,
    set::{Set, flag_enum},
};

flag_enum! {
    /// One decoder property a runtime must execute to run the model exactly.
    pub enum Feature {
        /// A residual multiplier or a logit divisor other than one.
        ResidualScaling,
        /// `LayerNorm` instead of RMS normalization.
        LayerNorm,
        /// Post-norm or parallel blocks.
        PostOrParallelNorm,
        /// A norm scaling by `1 + weight` (Gemma's).
        NormWeightOffset,
        /// A cap on attention scores.
        ScoreSoftcap,
        /// Chunked (not sliding) attention windows.
        ChunkedAttention,
        /// A position-dependent query scale.
        QueryTemperature,
        /// `LongRoPE` rotary scaling.
        LongRope,
        /// Keys that double as values.
        SharedKeyValue,
        /// Softmax attention that rotates positions.
        RotaryAttention,
        /// An embedding factor other than one.
        EmbeddingNotUnit,
        /// An embedding factor other than `sqrt(hidden_size)`.
        EmbeddingNotSqrtHidden,
        /// An attention scale other than one.
        AttentionScaleNotUnit,
        /// An attention scale other than `1 / sqrt(head_dim)`.
        AttentionScaleNotDefault,
        /// An attention scale other than the configuration's (or the
        /// default): an override that only a spec-driven kernel honors.
        AttentionScaleUndeclared,
        /// Experts selected from their best groups.
        ExpertGroups,
        /// A factor on the selected experts' output other than one.
        ExpertOutputScale,
        /// No attention layer at all, so nothing to cache per token.
        AttentionFree,
        /// Per-layer inputs beside the residual stream (Gemma 4's E models).
        PerLayerInput,
        /// Layers attending over an earlier layer's keys and values instead
        /// of projecting their own (Gemma 4's E models).
        SharedLayerKeyValues,
    }
}

pub type DecoderFeatures = Set<Feature>;

/// The features of `spec`; `declared_attention_scale` is the configuration's
/// attention scale, against which spec overrides are told apart.
#[allow(clippy::float_cmp, reason = "the identity factors are exact constants")]
pub fn lower_features(
    spec: &SemanticModelSpec,
    declared_attention_scale: Option<f64>,
) -> Result<DecoderFeatures> {
    let decoder = &spec.decoder;
    let layers = &decoder.layers;
    let mut features = DecoderFeatures::EMPTY;
    let mut set = |feature, present: bool| {
        if present {
            features.insert(feature);
        }
    };
    set(
        Feature::ResidualScaling,
        decoder.residual_multiplier != 1.0 || decoder.logit_divisor != 1.0,
    );
    set(
        Feature::LayerNorm,
        decoder.final_norm.kind == NormalizationKind::Layer
            || layers.iter().any(|layer| layer.input_norm.kind == NormalizationKind::Layer),
    );
    set(
        Feature::PostOrParallelNorm,
        layers.iter().any(|layer| !layer.norm_placement.is_pre()),
    );
    let offsets =
        std::iter::once(decoder.final_norm.weight_offset).chain(layers.iter().flat_map(|layer| {
            [layer.input_norm.weight_offset, layer.post_attention_norm.weight_offset]
        }));
    set(Feature::NormWeightOffset, offsets.into_iter().any(|offset| offset != 0.0));
    let sqrt_hidden = sqrt(decoder.hidden_size, "hidden size")?;
    set(Feature::EmbeddingNotUnit, decoder.embedding_multiplier != 1.0);
    set(Feature::EmbeddingNotSqrtHidden, decoder.embedding_multiplier != sqrt_hidden);
    for layer in layers {
        if let Some(routed) = layer.feed_forward.routed() {
            set(Feature::ExpertGroups, routed.group_limit.is_some());
            set(Feature::ExpertOutputScale, routed.output_scale != 1.0);
        }
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        set(Feature::ScoreSoftcap, attention.logit_softcap.is_some());
        set(Feature::SharedLayerKeyValues, attention.key_value_source.is_some());
        set(
            Feature::ChunkedAttention,
            matches!(attention.window, Some(AttentionWindow::Chunked { .. })),
        );
        set(Feature::QueryTemperature, attention.query_temperature.is_some());
        set(
            Feature::LongRope,
            matches!(&attention.position, PositionEncodingSpec::Rotary(rotary)
                if matches!(rotary.scaling, Some(RopeScalingSpec::LongRope { .. }))),
        );
        set(
            Feature::RotaryAttention,
            matches!(attention.position, PositionEncodingSpec::Rotary(_)),
        );
        set(
            Feature::SharedKeyValue,
            attention.key_value_relation == models::semantic::KeyValueRelation::KeyEqualsValue,
        );
        let default = sqrt(attention.head_dim, "head dimension")?.recip();
        set(Feature::AttentionScaleNotUnit, attention.scale != 1.0);
        set(Feature::AttentionScaleNotDefault, attention.scale != default);
        set(
            Feature::AttentionScaleUndeclared,
            attention.scale != declared_attention_scale.unwrap_or(default),
        );
    }
    let attends = layers.iter().any(|layer| {
        matches!(layer.mixer, MixerSpec::SoftmaxAttention(_) | MixerSpec::LatentAttention(_))
    });
    set(Feature::AttentionFree, !attends);
    set(Feature::PerLayerInput, decoder.per_layer_input.is_some());
    Ok(features)
}

fn sqrt(value: usize, what: &'static str) -> Result<f64> {
    let Ok(narrow) = u32::try_from(value) else {
        return Err(ArchitectureError::Unlowerable {
            layer: None,
            feature: Unlowerable::Dimension { what, value },
        });
    };
    Ok(f64::from(narrow).sqrt())
}
