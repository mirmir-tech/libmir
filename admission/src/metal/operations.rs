use models::semantic::{
    ActivationSpec, DecoderLayerSpec, FeedForwardSpec, MixerSpec, NormalizationKind,
    NormalizationSpec, QkNormalization, RouterNormalization,
};

use crate::{ArchitectureError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizationLowering {
    Rms,
    /// RMS norm scaling by `1 + weight` (Gemma's), declared as a weight offset
    /// of one.
    RmsUnitOffset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixerLowering {
    Softmax {
        sinks: bool,
        window: Option<usize>,
        normalization: QkNormalization,
    },
    Linear,
    StateSpace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardLowering {
    Dense,
    /// Gemma 4-style gated GELU with output norms and a layer scale.
    DenseGelu,
    /// Routed `SwiGLU` experts with no shared expert and no clamp
    /// (`Qwen3-MoE`), run by the dense runtime in place of its gated MLP.
    Routed,
    DenseAndRouted,
    SharedRouted,
    ClampedRouted,
    /// Sigmoid-routed `relu²` experts with an optional projected latent space.
    LatentRouted,
    /// The layer has no feed-forward block.
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerLowering {
    pub index: usize,
    pub input_norm: NormalizationLowering,
    pub post_attention_norm: NormalizationLowering,
    pub mixer: MixerLowering,
    pub feed_forward: FeedForwardLowering,
}

pub fn lower_layer(layer: &DecoderLayerSpec) -> Result<LayerLowering> {
    Ok(LayerLowering {
        index: layer.index,
        input_norm: lower_normalization(layer.index, "input", layer.input_norm)?,
        post_attention_norm: lower_normalization(
            layer.index,
            "post-attention",
            layer.post_attention_norm,
        )?,
        mixer: lower_mixer(&layer.mixer),
        feed_forward: lower_feed_forward(layer.index, &layer.feed_forward, &layer.mixer)?,
    })
}

pub(super) fn lower_normalization(
    layer: usize,
    position: &str,
    normalization: NormalizationSpec,
) -> Result<NormalizationLowering> {
    match normalization.kind {
        NormalizationKind::Rms if normalization.weight_offset.to_bits() == 0.0_f64.to_bits() => {
            Ok(NormalizationLowering::Rms)
        },
        NormalizationKind::Rms if normalization.weight_offset.to_bits() == 1.0_f64.to_bits() => {
            Ok(NormalizationLowering::RmsUnitOffset)
        },
        NormalizationKind::Rms => Err(ArchitectureError::invalid(format!(
            "Metal cannot lower layer {layer} {position} weight offset {}",
            normalization.weight_offset
        ))),
        NormalizationKind::Layer => Err(ArchitectureError::invalid(format!(
            "Metal cannot lower layer {layer} {position} layer normalization"
        ))),
    }
}

fn lower_mixer(mixer: &MixerSpec) -> MixerLowering {
    match mixer {
        MixerSpec::SoftmaxAttention(attention) => MixerLowering::Softmax {
            sinks: attention.sinks,
            normalization: attention.qk_normalization,
            window: attention.window,
        },
        MixerSpec::LinearAttention(_) => MixerLowering::Linear,
        MixerSpec::StateSpace(_) => MixerLowering::StateSpace,
    }
}

fn lower_feed_forward(
    layer: usize,
    feed_forward: &FeedForwardSpec,
    mixer: &MixerSpec,
) -> Result<FeedForwardLowering> {
    // Gemma 4 normalizes values too and runs the windowed dense-and-routed
    // execution; Gemma 3's gated GELU runs in the dense runtime.
    let value_norm = matches!(
        mixer,
        MixerSpec::SoftmaxAttention(attention)
            if attention.qk_normalization == QkNormalization::QueryKeyValueRms
    );
    match feed_forward {
        FeedForwardSpec::Dense { activation: ActivationSpec::GeluTanh, .. } if value_norm => {
            Ok(FeedForwardLowering::DenseGelu)
        },
        FeedForwardSpec::Dense {
            activation:
                ActivationSpec::SwiGlu { clamp: None, up_shift: 0.0, .. } | ActivationSpec::GeluTanh,
            ..
        } => Ok(FeedForwardLowering::Dense),
        FeedForwardSpec::DenseAndRouted {
            dense_activation: ActivationSpec::GeluTanh,
            routed,
            ..
        } if routed.activation == ActivationSpec::GeluTanh => {
            Ok(FeedForwardLowering::DenseAndRouted)
        },
        FeedForwardSpec::Routed { shared: Some(_), .. } => Ok(FeedForwardLowering::SharedRouted),
        FeedForwardSpec::Routed { routed, shared: None }
            if matches!(
                routed.activation,
                ActivationSpec::SwiGlu { clamp: None, up_shift, .. } if up_shift == 0.0
            ) =>
        {
            Ok(FeedForwardLowering::Routed)
        },
        FeedForwardSpec::Routed { routed, shared: None }
            if matches!(
                routed.activation,
                ActivationSpec::SwiGlu { clamp: Some(_), up_shift, .. } if up_shift != 0.0
            ) =>
        {
            Ok(FeedForwardLowering::ClampedRouted)
        },
        FeedForwardSpec::LatentRouted { routed, .. }
            if routed.activation == ActivationSpec::ReluSquared
                && routed.router_normalization == RouterNormalization::SigmoidBiasedTopK =>
        {
            Ok(FeedForwardLowering::LatentRouted)
        },
        FeedForwardSpec::None => Ok(FeedForwardLowering::None),
        _ => Err(ArchitectureError::invalid(format!(
            "Metal cannot lower layer {layer} feed-forward activation composition"
        ))),
    }
}
