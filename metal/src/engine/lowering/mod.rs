use models::semantic::SemanticModelSpec;

use super::{Error, Result};

mod operations;
#[cfg(test)]
mod tests;

pub use operations::{
    FeedForwardLowering, LayerLowering, MixerLowering, NormalizationLowering, lower_layer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderRuntime {
    Dense,
    DenseAndRouted,
    SharedRouted,
    ClampedRouted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderLowering {
    layers: Vec<LayerLowering>,
    runtime: DecoderRuntime,
}

impl DecoderLowering {
    #[must_use]
    pub fn layers(&self) -> &[LayerLowering] {
        &self.layers
    }

    #[must_use]
    pub const fn runtime(&self) -> DecoderRuntime {
        self.runtime
    }
}

pub fn plan(spec: &SemanticModelSpec) -> Result<DecoderLowering> {
    let layers = spec.decoder.layers.iter().map(lower_layer).collect::<Result<Vec<_>>>()?;
    let runtime = select_runtime(&layers)?;
    Ok(DecoderLowering { layers, runtime })
}

fn select_runtime(layers: &[LayerLowering]) -> Result<DecoderRuntime> {
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::Dense
            && matches!(layer.mixer, MixerLowering::Softmax { window: None, .. })
    }) {
        return Ok(DecoderRuntime::Dense);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::DenseAndRouted
            && matches!(layer.mixer, MixerLowering::Softmax { .. })
    }) {
        return Ok(DecoderRuntime::DenseAndRouted);
    }
    if layers
        .iter()
        .all(|layer| layer.feed_forward == FeedForwardLowering::SharedRouted)
        && layers.iter().any(|layer| layer.mixer == MixerLowering::Linear)
    {
        return Ok(DecoderRuntime::SharedRouted);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::ClampedRouted
            && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
    }) {
        return Ok(DecoderRuntime::ClampedRouted);
    }
    Err(Error::InvalidModel(
        "Metal has no runtime for the lowered decoder operation composition".into(),
    ))
}
