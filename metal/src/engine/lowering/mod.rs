use models::semantic::{DecoderLayerSpec, SemanticModelSpec};

use super::{Error, Result};

#[cfg(test)]
mod tests;

pub use architecture::metal::{
    FeedForwardLowering, LayerLowering, MetalDecoderPlan as DecoderLowering,
    MetalDecoderRuntime as DecoderRuntime, MixerLowering, NormalizationLowering,
};

pub fn plan(spec: &SemanticModelSpec) -> Result<DecoderLowering> {
    match architecture::metal::plan(spec) {
        Ok(plan) => Ok(plan),
        Err(error) => Err(Error::InvalidModel(error.to_string())),
    }
}

pub fn lower_layer(layer: &DecoderLayerSpec) -> Result<LayerLowering> {
    match architecture::metal::lower_layer(layer) {
        Ok(layer) => Ok(layer),
        Err(error) => Err(Error::InvalidModel(error.to_string())),
    }
}
