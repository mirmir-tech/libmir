use models::{
    layout::DecoderConfig,
    semantic::{DecoderLayerSpec, SemanticModelSpec},
};

use super::Result;

#[cfg(test)]
mod tests;

pub use architecture::metal::{
    Activation, Experts, FeedForwardLowering, LayerLowering, MetalDecoderPlan as DecoderLowering,
    MetalDecoderRuntime as DecoderRuntime, MixerLowering, NormalizationLowering,
};

/// Lowers `spec`, whose configuration is `decoder`, and selects its runtime.
pub fn plan(decoder: &DecoderConfig, spec: &SemanticModelSpec) -> Result<DecoderLowering> {
    Ok(architecture::metal::plan(decoder, spec)?)
}

pub fn lower_layer(layer: &DecoderLayerSpec) -> Result<LayerLowering> {
    Ok(architecture::metal::lower_layer(layer)?)
}
