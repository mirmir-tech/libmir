use std::fs;

use serde_json::Value;

use super::VisionPipeline;
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ImageProcessorConfig {
    Pooled(PooledImageProcessorConfig),
    SpatialMerge(SpatialMergeImageProcessorConfig),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PooledImageProcessorConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: usize,
    pub rescale_factor: f64,
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialMergeImageProcessorConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub rescale_factor: f64,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
}

impl ImageProcessorConfig {
    pub fn from_layout(layout: &ModelLayout, pipeline: VisionPipeline) -> Result<Option<Self>> {
        let Some(path) = layout
            .processor_config_path
            .as_ref()
            .or(layout.preprocessor_config_path.as_ref())
        else {
            return Ok(None);
        };
        let json = fs::read_to_string(path)?;
        let root: Value = serde_json::from_str(&json)?;
        let value = root.get("image_processor").filter(|value| value.is_object()).unwrap_or(&root);
        match pipeline {
            VisionPipeline::PooledEncoder => parse_pooled(value).map(Self::Pooled).map(Some),
            VisionPipeline::SpatialMergeEncoder => {
                parse_spatial_merge(value).map(Self::SpatialMerge).map(Some)
            },
        }
    }
}

fn parse_pooled(value: &Value) -> Result<PooledImageProcessorConfig> {
    Ok(PooledImageProcessorConfig {
        patch_size: usize_field(value, "patch_size")?,
        pooling_kernel_size: usize_field(value, "pooling_kernel_size")?,
        max_soft_tokens: usize_field(value, "max_soft_tokens")?,
        rescale_factor: float_field(value, "rescale_factor", 1.0 / 255.0)?,
        do_resize: bool_field(value, "do_resize", true)?,
        do_rescale: bool_field(value, "do_rescale", true)?,
        do_normalize: bool_field(value, "do_normalize", false)?,
    })
}

fn parse_spatial_merge(value: &Value) -> Result<SpatialMergeImageProcessorConfig> {
    let size = value.get("size").filter(|value| value.is_object()).unwrap_or(value);
    Ok(SpatialMergeImageProcessorConfig {
        patch_size: usize_field(value, "patch_size")?,
        temporal_patch_size: usize_field(value, "temporal_patch_size")?,
        spatial_merge_size: usize_field_alias(value, &["merge_size", "spatial_merge_size"])?,
        min_pixels: usize_field_alias(value, &["min_pixels"])
            .or_else(|_| usize_field(size, "shortest_edge"))?,
        max_pixels: usize_field_alias(value, &["max_pixels"])
            .or_else(|_| usize_field(size, "longest_edge"))?,
        rescale_factor: float_field(value, "rescale_factor", 1.0 / 255.0)?,
        image_mean: float_triplet(value, "image_mean")?,
        image_std: float_triplet(value, "image_std")?,
        do_resize: bool_field(value, "do_resize", true)?,
        do_rescale: bool_field(value, "do_rescale", true)?,
        do_normalize: bool_field(value, "do_normalize", true)?,
    })
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    usize_field_alias(value, &[field])
}

fn usize_field_alias(value: &Value, fields: &[&str]) -> Result<usize> {
    let raw = fields
        .iter()
        .find_map(|field| value.get(*field).and_then(Value::as_u64))
        .ok_or_else(|| invalid(format!("missing image processor integer {}", fields.join("/"))))?;
    Ok(usize::try_from(raw)?)
}

fn float_field(value: &Value, field: &str, default: f64) -> Result<f64> {
    value.get(field).map_or(Ok(default), |value| {
        value
            .as_f64()
            .ok_or_else(|| invalid(format!("invalid image processor float {field}")))
    })
}

fn bool_field(value: &Value, field: &str, default: bool) -> Result<bool> {
    value.get(field).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(format!("invalid image processor boolean {field}")))
    })
}

fn float_triplet(value: &Value, field: &str) -> Result<[f64; 3]> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("missing image processor triplet {field}")))?;
    let values: Vec<f64> = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| invalid(format!("invalid image processor triplet {field}")))
        })
        .collect::<Result<_>>()?;
    values
        .try_into()
        .map_err(|_values| invalid(format!("image processor {field} must contain three values")))
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
