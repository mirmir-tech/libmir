//! Records a real checkpoint's tower output for comparison with the CUDA
//! recorder (`records_a_real_pixel_shuffle_tower_output`) on the same image.

use std::path::PathBuf;

use models::layout::{ImageProcessorConfig, ModelLayout, VisionConfig};

use super::super::PixelShuffleVisionTower;
use crate::engine::{Error, ModelTensors, Result, Stream};

/// Runs the tower on `LIBMIR_VISION_IMAGE` and writes the projected soft tokens
/// as little-endian f32 to `LIBMIR_VISION_TOWER_OUTPUT`.
#[test]
#[ignore = "loads a real vision checkpoint; set MODEL, LIBMIR_VISION_IMAGE and LIBMIR_VISION_TOWER_OUTPUT"]
fn records_a_real_pixel_shuffle_tower_output() -> Result<()> {
    let root = required_path("MODEL")?;
    let image_path = required_path("LIBMIR_VISION_IMAGE")?;
    let output_path = required_path("LIBMIR_VISION_TOWER_OUTPUT")?;
    let layout = ModelLayout::inspect(&root)?;
    let vision = VisionConfig::from_layout(&layout)?
        .ok_or_else(|| Error::InvalidModel("checkpoint has no vision config".into()))?;
    let processor = ImageProcessorConfig::from_layout(&layout, vision.pipeline())?
        .ok_or_else(|| Error::InvalidModel("checkpoint has no image processor".into()))?;
    let (VisionConfig::PixelShuffleEncoder(config), ImageProcessorConfig::PixelShuffle(processor)) =
        (vision, processor)
    else {
        return Err(Error::InvalidModel("checkpoint is not pixel-shuffle vision".into()));
    };
    let image = processor.preprocess_encoded(&std::fs::read(image_path)?)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let output = PixelShuffleVisionTower::load(&tensors, &config, &stream)?
        .forward_preprocessed(&image, &stream)?;
    let values = output.to_vec_f32(&stream)?;
    assert!(values.iter().all(|value| value.is_finite()));
    std::fs::write(
        output_path,
        values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| Error::InvalidModel(format!("{name} is unset")))
}
