//! The vision half of a remote contract, from `SafeTensors` headers.

use models::{
    layout::{ImageProcessorConfig, VisionConfig},
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};

use crate::Result;

/// Vision configuration, processor, and tensor readiness discovered from a
/// remote checkpoint without fetching tensor payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteVisionContract {
    pub(super) config: VisionConfig,
    pub(super) readiness: TensorReadiness,
    pub(super) processor: Option<ImageProcessorConfig>,
}

impl RemoteVisionContract {
    /// Returns the normalized vision execution configuration.
    #[must_use]
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Returns vision tensor readiness derived from `SafeTensors` headers.
    #[must_use]
    pub const fn readiness(&self) -> &TensorReadiness {
        &self.readiness
    }

    /// Returns the parsed image processor, when the repository provides one.
    #[must_use]
    pub const fn processor(&self) -> Option<&ImageProcessorConfig> {
        self.processor.as_ref()
    }
}

pub(super) fn remote_vision(
    config: &serde_json::Value,
    catalog: &TensorCatalog,
    processor: Option<&serde_json::Value>,
) -> Result<Option<RemoteVisionContract>> {
    // A vision tower without a supported execution contract leaves a text-only
    // contract, matching local model inspection; its tensors stay unbound.
    let Some(config) = VisionConfig::from_value(config)? else {
        return Ok(None);
    };
    let readiness = VisionTensorSchema::discover(&config).readiness(catalog);
    let processor = processor
        .map(|value| ImageProcessorConfig::from_value(value, config.pipeline()))
        .transpose()?;
    Ok(Some(RemoteVisionContract { config, readiness, processor }))
}
