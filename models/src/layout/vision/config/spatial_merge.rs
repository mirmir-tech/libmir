//! The spatial-merge encoder contract: temporal-spatial patches and a
//! learned spatial merger.

use serde_json::Value;

use super::super::parse::{
    bool_field, has_fields, object, scalar_usize_field, string_field, u32_field, usize_array_field,
    usize_field,
};
use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialMergeVisionConfig {
    pub hidden_size: usize,
    pub output_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_activation: String,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub mrope_interleaved: bool,
    pub mrope_sections: Vec<usize>,
}

pub(super) fn is_spatial_merge_encoder(root: &Value, vision: &Value) -> bool {
    has_fields(
        vision,
        &[
            "hidden_size",
            "out_hidden_size",
            "intermediate_size",
            "depth",
            "num_heads",
            "in_channels",
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "num_position_embeddings",
            "hidden_act",
        ],
    ) && has_fields(
        root,
        &["text_config", "image_token_id", "vision_start_token_id", "vision_end_token_id"],
    )
}

pub(super) fn parse_spatial_merge(
    root: &Value,
    vision: &Value,
) -> Result<SpatialMergeVisionConfig> {
    let text = object(root, "text_config")?;
    let rope = object(text, "rope_parameters")?;
    Ok(SpatialMergeVisionConfig {
        hidden_size: usize_field(vision, "hidden_size")?,
        output_hidden_size: usize_field(vision, "out_hidden_size")?,
        intermediate_size: usize_field(vision, "intermediate_size")?,
        num_hidden_layers: usize_field(vision, "depth")?,
        num_attention_heads: usize_field(vision, "num_heads")?,
        in_channels: usize_field(vision, "in_channels")?,
        patch_size: scalar_usize_field(vision, "patch_size")?,
        temporal_patch_size: scalar_usize_field(vision, "temporal_patch_size")?,
        spatial_merge_size: usize_field(vision, "spatial_merge_size")?,
        num_position_embeddings: usize_field(vision, "num_position_embeddings")?,
        hidden_activation: string_field(vision, "hidden_act")?,
        image_token_id: u32_field(root, "image_token_id")?,
        vision_start_token_id: u32_field(root, "vision_start_token_id")?,
        vision_end_token_id: u32_field(root, "vision_end_token_id")?,
        mrope_interleaved: bool_field(rope, "mrope_interleaved", false)?,
        mrope_sections: usize_array_field(rope, "mrope_section")?,
    })
}
