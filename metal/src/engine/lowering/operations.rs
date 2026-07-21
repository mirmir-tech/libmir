use models::semantic::{
    ActivationSpec, DecoderLayerSpec, FeedForwardSpec, MixerSpec, NormalizationKind,
    NormalizationSpec,
};

use super::super::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationLowering {
    Rms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerLowering {
    Softmax { sinks: bool, window: Option<usize> },
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardLowering {
    Dense,
    DenseAndRouted,
    SharedRouted,
    ClampedRouted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        feed_forward: lower_feed_forward(layer.index, &layer.feed_forward)?,
    })
}

fn lower_normalization(
    layer: usize,
    position: &str,
    normalization: NormalizationSpec,
) -> Result<NormalizationLowering> {
    match normalization.kind {
        NormalizationKind::Rms => Ok(NormalizationLowering::Rms),
        NormalizationKind::Layer => Err(Error::InvalidModel(format!(
            "Metal cannot lower layer {layer} {position} layer normalization"
        ))),
    }
}

const fn lower_mixer(mixer: &MixerSpec) -> MixerLowering {
    match mixer {
        MixerSpec::SoftmaxAttention(attention) => MixerLowering::Softmax {
            sinks: attention.sinks,
            window: attention.window,
        },
        MixerSpec::LinearAttention(_) => MixerLowering::Linear,
    }
}

fn lower_feed_forward(layer: usize, feed_forward: &FeedForwardSpec) -> Result<FeedForwardLowering> {
    match feed_forward {
        FeedForwardSpec::Dense {
            activation: ActivationSpec::SwiGlu { clamp: None, up_shift: 0.0, .. },
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
                ActivationSpec::SwiGlu { clamp: Some(_), up_shift, .. } if up_shift != 0.0
            ) =>
        {
            Ok(FeedForwardLowering::ClampedRouted)
        },
        _ => Err(Error::InvalidModel(format!(
            "Metal cannot lower layer {layer} feed-forward activation composition"
        ))),
    }
}
