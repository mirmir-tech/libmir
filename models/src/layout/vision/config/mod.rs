use std::fs;

use serde_json::Value;

mod pixel_shuffle;
mod pooled;
mod spatial_merge;

pub(super) use self::pixel_shuffle::shuffle_factor;
pub use self::{
    pixel_shuffle::{ImageTags, PixelShuffleVisionConfig},
    pooled::PooledVisionConfig,
    spatial_merge::SpatialMergeVisionConfig,
};
use self::{
    pixel_shuffle::{is_pixel_shuffle_encoder, parse_pixel_shuffle},
    pooled::{is_pooled_encoder, parse_pooled},
    spatial_merge::{is_spatial_merge_encoder, parse_spatial_merge},
};
use super::parse::invalid;
use crate::{error::Result, layout::ModelLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionPipeline {
    PooledEncoder,
    SpatialMergeEncoder,
    PixelShuffleEncoder,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VisionConfig {
    PooledEncoder(PooledVisionConfig),
    SpatialMergeEncoder(SpatialMergeVisionConfig),
    PixelShuffleEncoder(PixelShuffleVisionConfig),
}

impl VisionConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Option<Self>> {
        let json = fs::read_to_string(&layout.config_path)?;
        let value: Value = serde_json::from_str(&json)?;
        Self::from_value(&value)
    }

    pub fn from_value(root: &Value) -> Result<Option<Self>> {
        let Some(vision) = root.get("vision_config").filter(|value| value.is_object()) else {
            return Ok(None);
        };
        let matched: Vec<VisionPipeline> = [
            is_pooled_encoder(root, vision).then_some(VisionPipeline::PooledEncoder),
            is_spatial_merge_encoder(root, vision).then_some(VisionPipeline::SpatialMergeEncoder),
            is_pixel_shuffle_encoder(root, vision).then_some(VisionPipeline::PixelShuffleEncoder),
        ]
        .into_iter()
        .flatten()
        .collect();
        let pipeline = match matched[..] {
            [] => return Ok(None),
            [pipeline] => pipeline,
            _ => {
                return Err(invalid(
                    "vision configuration matches multiple supported execution contracts",
                ));
            },
        };
        match pipeline {
            VisionPipeline::PooledEncoder => parse_pooled(root, vision).map(Self::PooledEncoder),
            VisionPipeline::SpatialMergeEncoder => {
                parse_spatial_merge(root, vision).map(Self::SpatialMergeEncoder)
            },
            VisionPipeline::PixelShuffleEncoder => {
                parse_pixel_shuffle(root, vision).map(Self::PixelShuffleEncoder)
            },
        }
        .map(Some)
    }

    #[must_use]
    pub const fn pipeline(&self) -> VisionPipeline {
        match self {
            Self::PooledEncoder(_) => VisionPipeline::PooledEncoder,
            Self::SpatialMergeEncoder(_) => VisionPipeline::SpatialMergeEncoder,
            Self::PixelShuffleEncoder(_) => VisionPipeline::PixelShuffleEncoder,
        }
    }

    #[must_use]
    pub const fn num_hidden_layers(&self) -> usize {
        match self {
            Self::PooledEncoder(config) => config.num_hidden_layers,
            Self::SpatialMergeEncoder(config) => config.num_hidden_layers,
            Self::PixelShuffleEncoder(config) => config.num_hidden_layers,
        }
    }
}
