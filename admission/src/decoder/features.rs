//! The decoder-wide properties a runtime either executes or refuses: the
//! dense-only list both backends used to check by hand, lowered once.

use models::semantic::{
    AttentionWindow, MixerSpec, NormalizationKind, PositionEncodingSpec, RopeScalingSpec,
    SemanticModelSpec,
};

use crate::{
    ArchitectureError, Result, Unlowerable,
    capability::{AttentionScale, EmbeddingScale},
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
        /// Experts selected from their best groups.
        ExpertGroups,
        /// A factor on the selected experts' output other than one.
        ExpertOutputScale,
        /// No attention layer at all, so nothing to cache per token.
        AttentionFree,
        /// No state-space mixer, which the state-space hybrid runtimes are
        /// built around.
        StateSpaceFree,
        /// Per-layer inputs beside the residual stream (Gemma 4's E models).
        PerLayerInput,
        /// Layers attending over an earlier layer's keys and values instead
        /// of projecting their own (Gemma 4's E models).
        SharedLayerKeyValues,
        /// Softmax attention projecting without biases.
        UnbiasedAttention,
        /// Softmax attention not rotating the whole head with a scaled
        /// `RoPE` (`YaRN`, linear and the others): no rotation, a partial
        /// one, or an unscaled one.
        PartialOrUnscaledRope,
        /// Not one softmax-attention layer with routed experts and full
        /// `YaRN` rotation repeated in every layer (set by CUDA's plan,
        /// which lowers that layer for the runtimes built around it).
        NoRepeatedLayer,
    }
}

/// The decoder-wide properties of a plan: its feature flags, its embedding
/// factor and the scale of each attention layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderFeatures {
    pub flags: Set<Feature>,
    pub embedding: EmbeddingScale,
    pub attention: Set<AttentionScale>,
}

impl DecoderFeatures {
    #[must_use]
    pub fn contains(self, feature: Feature) -> bool {
        self.flags.contains(feature)
    }
}

/// The features of `spec`; `declared_attention_scale` is the configuration's
/// attention scale, against which spec overrides are told apart.
#[allow(clippy::float_cmp, reason = "the identity factors are exact constants")]
pub fn lower_features(
    spec: &SemanticModelSpec,
    declared_attention_scale: Option<f64>,
) -> Result<DecoderFeatures> {
    let decoder = &spec.decoder;
    let layers = &decoder.layers;
    let mut features = Set::<Feature>::EMPTY;
    let mut attention_scales = Set::EMPTY;
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
    let embedding = EmbeddingScale::of(decoder.embedding_multiplier, sqrt_hidden);
    for layer in layers {
        if let Some(routed) = layer.feed_forward.routed() {
            set(Feature::ExpertGroups, routed.group_limit.is_some());
            set(Feature::ExpertOutputScale, routed.output_scale != 1.0);
        }
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        set(Feature::ScoreSoftcap, attention.logit_softcap.is_some());
        set(Feature::UnbiasedAttention, !attention.projection_bias);
        let full_scaled = match &attention.position {
            PositionEncodingSpec::Rotary(rotary) => {
                rotary.scaling.is_some()
                    && rotary
                        .rotated_dimensions(attention.head_dim)
                        .is_ok_and(|dimensions| dimensions == attention.head_dim)
            },
            PositionEncodingSpec::None => false,
        };
        set(Feature::PartialOrUnscaledRope, !full_scaled);
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
        attention_scales.insert(AttentionScale::of(
            attention.scale,
            default,
            declared_attention_scale,
        ));
    }
    let attends = layers.iter().any(|layer| {
        matches!(layer.mixer, MixerSpec::SoftmaxAttention(_) | MixerSpec::LatentAttention(_))
    });
    set(Feature::AttentionFree, !attends);
    set(
        Feature::StateSpaceFree,
        !layers.iter().any(|layer| matches!(layer.mixer, MixerSpec::StateSpace(_))),
    );
    set(Feature::PerLayerInput, decoder.per_layer_input.is_some());
    Ok(DecoderFeatures {
        flags: features,
        embedding,
        attention: attention_scales,
    })
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
