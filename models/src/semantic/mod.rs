mod attention;
mod compile;
mod feed_forward;
mod model;
mod sidecar;
mod validation;

pub use attention::{
    AttentionOutputSpec, AttentionSpec, KeyValueRelation, LinearAttentionSpec, MixerSpec,
    PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec, RotarySpec,
};
pub use feed_forward::{
    ActivationSpec, FeedForwardSpec, RoutedExpertsSpec, RouterNormalization, SharedExpertSpec,
};
pub use model::{
    DecoderLayerSpec, DecoderSpec, NormalizationKind, NormalizationSpec, SemanticModelSpec,
};

use crate::{
    error::Result,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

impl SemanticModelSpec {
    pub fn discover(decoder: &DecoderConfig, tensors: &TensorCatalog) -> Result<Self> {
        let spec = compile::compile(decoder, tensors)?;
        validation::validate(&spec)?;
        Ok(spec)
    }

    pub fn from_layout(
        layout: &ModelLayout,
        decoder: &DecoderConfig,
        tensors: &TensorCatalog,
    ) -> Result<Self> {
        layout
            .model_spec_path
            .as_deref()
            .map_or_else(|| Self::discover(decoder, tensors), sidecar::read)
    }

    pub fn to_toml(&self) -> Result<String> {
        validation::validate(self)?;
        Ok(toml::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests;
