use std::fs;

use serde_json::Value;

use super::parse::{
    bool_field, float_field, has_fields, invalid, object, optional_usize_field, scalar_usize_field,
    string_field, u32_field, usize_array_field, usize_field,
};
use crate::{error::Result, layout::ModelLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionPipeline {
    PooledEncoder,
    SpatialMergeEncoder,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VisionConfig {
    PooledEncoder(PooledVisionConfig),
    SpatialMergeEncoder(SpatialMergeVisionConfig),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PooledVisionConfig {
    pub hidden_size: usize,
    pub output_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_activation: String,
    pub use_clipped_linears: bool,
    pub standardize: bool,
    pub image_token_id: u32,
    pub image_begin_token_id: u32,
    pub image_end_token_id: u32,
    pub soft_tokens_per_image: usize,
    pub bidirectional_image_attention: bool,
}

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

impl VisionConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Option<Self>> {
        let json = fs::read_to_string(&layout.config_path)?;
        let value: Value = serde_json::from_str(&json)?;
        Self::from_value(&value)
    }

    pub(crate) fn from_value(root: &Value) -> Result<Option<Self>> {
        let Some(vision) = root.get("vision_config").filter(|value| value.is_object()) else {
            return Ok(None);
        };
        let pooled = is_pooled_encoder(root, vision);
        let spatial_merge = is_spatial_merge_encoder(root, vision);
        match (pooled, spatial_merge) {
            (true, false) => parse_pooled(root, vision).map(Self::PooledEncoder).map(Some),
            (false, true) => {
                parse_spatial_merge(root, vision).map(Self::SpatialMergeEncoder).map(Some)
            },
            (false, false) => Ok(None),
            (true, true) => {
                Err(invalid("vision configuration matches multiple supported execution contracts"))
            },
        }
    }

    #[must_use]
    pub const fn pipeline(&self) -> VisionPipeline {
        match self {
            Self::PooledEncoder(_) => VisionPipeline::PooledEncoder,
            Self::SpatialMergeEncoder(_) => VisionPipeline::SpatialMergeEncoder,
        }
    }

    #[must_use]
    pub const fn num_hidden_layers(&self) -> usize {
        match self {
            Self::PooledEncoder(config) => config.num_hidden_layers,
            Self::SpatialMergeEncoder(config) => config.num_hidden_layers,
        }
    }
}

fn is_pooled_encoder(root: &Value, vision: &Value) -> bool {
    has_fields(
        vision,
        &[
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "patch_size",
            "pooling_kernel_size",
            "position_embedding_size",
            "rms_norm_eps",
            "hidden_activation",
        ],
    ) && has_fields(
        root,
        &[
            "text_config",
            "image_token_id",
            "boi_token_id",
            "eoi_token_id",
            "vision_soft_tokens_per_image",
        ],
    )
}

fn is_spatial_merge_encoder(root: &Value, vision: &Value) -> bool {
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

fn parse_pooled(root: &Value, vision: &Value) -> Result<PooledVisionConfig> {
    let text = object(root, "text_config")?;
    let num_attention_heads = usize_field(vision, "num_attention_heads")?;
    let hidden_size = usize_field(vision, "hidden_size")?;
    let head_dim = optional_usize_field(vision, "head_dim")?
        .unwrap_or_else(|| hidden_size / num_attention_heads);
    let rope_theta = vision
        .get("rope_parameters")
        .and_then(|value| value.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(100.0);
    Ok(PooledVisionConfig {
        hidden_size,
        output_hidden_size: usize_field(text, "hidden_size")?,
        intermediate_size: usize_field(vision, "intermediate_size")?,
        num_hidden_layers: usize_field(vision, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: optional_usize_field(vision, "num_key_value_heads")?
            .unwrap_or(num_attention_heads),
        head_dim,
        patch_size: usize_field(vision, "patch_size")?,
        pooling_kernel_size: usize_field(vision, "pooling_kernel_size")?,
        position_embedding_size: usize_field(vision, "position_embedding_size")?,
        rms_norm_eps: float_field(vision, "rms_norm_eps")?,
        rope_theta,
        hidden_activation: string_field(vision, "hidden_activation")?,
        use_clipped_linears: bool_field(vision, "use_clipped_linears", false)?,
        standardize: bool_field(vision, "standardize", false)?,
        image_token_id: u32_field(root, "image_token_id")?,
        image_begin_token_id: u32_field(root, "boi_token_id")?,
        image_end_token_id: u32_field(root, "eoi_token_id")?,
        soft_tokens_per_image: usize_field(root, "vision_soft_tokens_per_image")?,
        bidirectional_image_attention: text
            .get("use_bidirectional_attention")
            .and_then(Value::as_str)
            == Some("vision"),
    })
}

fn parse_spatial_merge(root: &Value, vision: &Value) -> Result<SpatialMergeVisionConfig> {
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
