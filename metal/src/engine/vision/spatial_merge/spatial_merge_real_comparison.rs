use std::path::PathBuf;

use models::layout::{ImageProcessorConfig, ModelLayout, VisionConfig};

use super::super::SpatialMergeVisionTower;
use crate::engine::{Error, ModelTensors, Result, Stream};

#[test]
#[ignore = "loads a real vision checkpoint; set MODEL and LIBMIR_VISION_TOWER_OUTPUT"]
fn records_a_real_spatial_merge_tower_output() -> Result<()> {
    let root = required_path("MODEL")?;
    let output_path = required_path("LIBMIR_VISION_TOWER_OUTPUT")?;
    let layout = ModelLayout::inspect(&root)?;
    let vision = VisionConfig::from_layout(&layout)?
        .ok_or_else(|| Error::InvalidModel("checkpoint has no vision config".into()))?;
    let processor = ImageProcessorConfig::from_layout(&layout, vision.pipeline())?
        .ok_or_else(|| Error::InvalidModel("checkpoint has no image processor".into()))?;
    let (VisionConfig::SpatialMergeEncoder(config), ImageProcessorConfig::SpatialMerge(processor)) =
        (vision, processor)
    else {
        return Err(Error::InvalidModel("checkpoint is not spatial-merge vision".into()));
    };
    let rgb = comparison_rgb()?;
    let image = processor.preprocess_rgb(&rgb, 64, 64)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let output = SpatialMergeVisionTower::load(&tensors, &config, &stream)?
        .forward_preprocessed(&image, &stream)?;
    let values = output.to_vec_f32_on_stream(&stream)?;
    assert_eq!(
        output.shape()?,
        [1, i32::try_from(image.soft_tokens)?, i32::try_from(config.output_hidden_size)?,]
    );
    assert!(values.iter().all(|value| value.is_finite()));
    std::fs::write(output_path, f32_bytes(&values))?;
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| Error::InvalidModel(format!("{name} is unset")))
}

fn comparison_rgb() -> Result<Vec<u8>> {
    (0..64 * 64 * 3)
        .map(|index| u8::try_from(index % 251).map_err(Error::from))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}
