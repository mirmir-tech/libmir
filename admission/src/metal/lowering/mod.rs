//! The Metal decoder plan: the shared lowering plus the norms Metal's kernels
//! compute.

use models::semantic::{
    DecoderLayerSpec, MixerSpec, NormPlacement, NormalizationKind, NormalizationSpec,
    PositionEncodingSpec, SemanticModelSpec,
};

use crate::{
    ArchitectureError, Constant, Result, Unlowerable,
    capability::Operations,
    decoder::{DecoderFeatures, lower_features},
    lowering::{FeedForwardLowering, MixerLowering, lower_feed_forward},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizationLowering {
    Rms,
    /// RMS norm scaling by `1 + weight` (Gemma's), declared as a weight offset
    /// of one.
    RmsUnitOffset,
    /// Mean-centred `LayerNorm` scaling by its weight, without bias (Cohere).
    Layer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerLowering {
    pub index: usize,
    pub input_norm: NormalizationLowering,
    pub post_attention_norm: NormalizationLowering,
    pub norm_placement: NormPlacement,
    pub mixer: MixerLowering,
    pub feed_forward: FeedForwardLowering,
}

pub fn lower_layer(layer: &DecoderLayerSpec) -> Result<LayerLowering> {
    // Metal's latent attention rotates the key part by any rotary scaling
    // (its loader reads the inverse frequencies), but there must be one.
    if let MixerSpec::LatentAttention(latent) = &layer.mixer
        && latent.position == PositionEncodingSpec::None
    {
        return Err(ArchitectureError::Unlowerable {
            layer: Some(layer.index),
            feature: Unlowerable::LatentUnrotated,
        });
    }
    Ok(LayerLowering {
        index: layer.index,
        input_norm: lower_normalization(Some(layer.index), layer.input_norm)?,
        post_attention_norm: lower_normalization(Some(layer.index), layer.post_attention_norm)?,
        norm_placement: layer.norm_placement,
        mixer: MixerLowering::lower(&layer.mixer),
        feed_forward: lower_feed_forward(&layer.feed_forward, layer.index)?,
    })
}

/// Every layer, the final norm and the decoder features, before a runtime is
/// chosen.
pub(super) struct Lowered {
    pub layers: Vec<LayerLowering>,
    pub final_norm: NormalizationLowering,
    pub features: DecoderFeatures,
}

impl Lowered {
    pub fn lower(spec: &SemanticModelSpec, declared_attention_scale: Option<f64>) -> Result<Self> {
        Ok(Self {
            layers: spec.decoder.layers.iter().map(lower_layer).collect::<Result<_>>()?,
            final_norm: lower_normalization(None, spec.decoder.final_norm)?,
            features: lower_features(spec, declared_attention_scale)?,
        })
    }

    pub fn requirements(&self) -> Operations {
        Operations::required(
            self.layers.iter().map(|layer| (&layer.mixer, &layer.feed_forward)),
            self.features,
        )
    }
}

fn lower_normalization(
    layer: Option<usize>,
    normalization: NormalizationSpec,
) -> Result<NormalizationLowering> {
    let offset = normalization.weight_offset;
    let refuse = |feature| Err(ArchitectureError::Unlowerable { layer, feature });
    match normalization.kind {
        NormalizationKind::Rms if offset.to_bits() == 0.0_f64.to_bits() => {
            Ok(NormalizationLowering::Rms)
        },
        NormalizationKind::Rms if offset.to_bits() == 1.0_f64.to_bits() => {
            Ok(NormalizationLowering::RmsUnitOffset)
        },
        NormalizationKind::Rms => {
            refuse(Unlowerable::NormWeightOffset { offset: Constant(offset) })
        },
        NormalizationKind::Layer if offset.to_bits() == 0.0_f64.to_bits() => {
            Ok(NormalizationLowering::Layer)
        },
        NormalizationKind::Layer => refuse(Unlowerable::LayerNormWeightOffset),
    }
}
