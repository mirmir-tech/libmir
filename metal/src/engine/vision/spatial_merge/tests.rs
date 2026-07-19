use std::{fs, path::Path};

use models::{
    layout::{
        ImageProcessorConfig, ModelLayout, SpatialMergeImageProcessorConfig,
        SpatialMergeVisionConfig, VisionConfig,
    },
    vision::SpatialMergePreprocessedImage,
};
use serde_json::{Map, Value, json};

use super::{SpatialMergeVisionTower, rope::VisionRope};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[test]
fn vision_rope_rotates_the_complete_head_across_both_spatial_axes() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Array::from_f32(&input_values, &[1, 1, 1, 8])?;
    let positions = Array::from_u32(&[1, 2], &[1, 2])?;
    let (actual, _) = VisionRope::new(8)?.apply(&input, &input, &positions, &stream)?;
    let angles = [1.0_f32, 0.01, 2.0, 0.02, 1.0, 0.01, 2.0, 0.02];
    let rotated = [-5.0_f32, -6.0, -7.0, -8.0, 1.0, 2.0, 3.0, 4.0];
    let expected = input_values
        .iter()
        .zip(rotated)
        .zip(angles)
        .map(|((&value, rotated), angle)| rotated.mul_add(angle.sin(), value * angle.cos()))
        .collect::<Vec<_>>();
    let actual = actual.to_vec_f32_on_stream(&stream)?;
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| { (actual - expected).abs() < 1.0e-5 })
    );
    Ok(())
}

#[test]
fn loads_and_executes_a_complete_synthetic_tower() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-spatial-merge-vision-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_weights(&root.join("model.safetensors"))?;
    let result = execute(&root);
    fs::remove_dir_all(root)?;
    result
}

#[test]
#[ignore = "loads a real spatial-merge vision checkpoint; set MIRMIR_QWEN36_MODEL"]
fn executes_a_real_spatial_merge_vision_tower() -> Result<()> {
    let root = std::env::var_os("MIRMIR_QWEN36_MODEL")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| Error::InvalidModel("MIRMIR_QWEN36_MODEL is unset".into()))?;
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
    let rgb = (0..64 * 64 * 3)
        .map(|index| u8::try_from(index % 251))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let image = processor.preprocess_rgb(&rgb, 64, 64)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let output = SpatialMergeVisionTower::load(&tensors, &config, &stream)?
        .forward_preprocessed(&image, &stream)?;
    let values = output.to_vec_f32_on_stream(&stream)?;
    assert_eq!(output.shape()?, [1, i32::try_from(image.soft_tokens)?, 2048]);
    assert!(values.iter().all(|value| value.is_finite()));
    Ok(())
}

fn execute(root: &Path) -> Result<()> {
    let tensors = ModelTensors::load(root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let tower = SpatialMergeVisionTower::load(&tensors, &config(), &stream)?;
    let image = SpatialMergeImageProcessorConfig {
        patch_size: 1,
        temporal_patch_size: 2,
        spatial_merge_size: 1,
        min_pixels: 1,
        max_pixels: 16,
        rescale_factor: 1.0,
        image_mean: [0.0; 3],
        image_std: [1.0; 3],
        do_resize: false,
        do_rescale: false,
        do_normalize: false,
    }
    .preprocess_rgb(&[1, 2, 3], 1, 1)?;
    assert_output(&tower, &image, &stream)
}

fn assert_output(
    tower: &SpatialMergeVisionTower,
    image: &SpatialMergePreprocessedImage,
    stream: &Stream,
) -> Result<()> {
    let output = tower.forward_preprocessed(image, stream)?;
    assert_eq!(output.shape()?, [1, 1, 8]);
    assert!(output.to_vec_f32_on_stream(stream)?.iter().all(|value| value.is_finite()));
    Ok(())
}

fn write_weights(path: &Path) -> Result<()> {
    let mut tensors = vec![
        tensor("model.visual.patch_embed.proj.weight", &[8, 3, 2, 1, 1], identity(8, 6)),
        tensor("model.visual.patch_embed.proj.bias", &[8], zeros(8)),
        tensor("model.visual.pos_embed.weight", &[4, 8], zeros(32)),
        tensor("model.visual.merger.norm.weight", &[8], ones(8)),
        tensor("model.visual.merger.norm.bias", &[8], zeros(8)),
        tensor("model.visual.merger.linear_fc1.weight", &[8, 8], identity(8, 8)),
        tensor("model.visual.merger.linear_fc1.bias", &[8], zeros(8)),
        tensor("model.visual.merger.linear_fc2.weight", &[8, 8], identity(8, 8)),
        tensor("model.visual.merger.linear_fc2.bias", &[8], zeros(8)),
    ];
    let prefix = "model.visual.blocks.0";
    for norm in ["norm1", "norm2"] {
        tensors.push(tensor(&format!("{prefix}.{norm}.weight"), &[8], ones(8)));
        tensors.push(tensor(&format!("{prefix}.{norm}.bias"), &[8], zeros(8)));
    }
    tensors.extend([
        tensor(&format!("{prefix}.attn.qkv.weight"), &[24, 8], zeros(192)),
        tensor(&format!("{prefix}.attn.qkv.bias"), &[24], zeros(24)),
        tensor(&format!("{prefix}.attn.proj.weight"), &[8, 8], identity(8, 8)),
        tensor(&format!("{prefix}.attn.proj.bias"), &[8], zeros(8)),
        tensor(&format!("{prefix}.mlp.linear_fc1.weight"), &[8, 8], zeros(64)),
        tensor(&format!("{prefix}.mlp.linear_fc1.bias"), &[8], zeros(8)),
        tensor(&format!("{prefix}.mlp.linear_fc2.weight"), &[8, 8], zeros(64)),
        tensor(&format!("{prefix}.mlp.linear_fc2.bias"), &[8], zeros(8)),
    ]);
    write_safetensors(path, &tensors)
}

struct TestTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TestTensor {
    TestTensor {
        name: name.into(),
        shape: shape.into(),
        values,
    }
}

fn write_safetensors(path: &Path, tensors: &[TestTensor]) -> Result<()> {
    let mut header = Map::new();
    let mut offset = 0;
    let mut payload = Vec::new();
    for tensor in tensors {
        let bytes = tensor.values.len() * size_of::<f32>();
        header.insert(
            tensor.name.clone(),
            json!({"dtype": "F32", "shape": &tensor.shape, "data_offsets": [offset, offset + bytes]}),
        );
        tensor
            .values
            .iter()
            .for_each(|value| payload.extend_from_slice(&value.to_le_bytes()));
        offset += bytes;
    }
    let mut header = serde_json::to_string(&Value::Object(header))?;
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = u64::try_from(header.len())?.to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend(payload);
    fs::write(path, file)?;
    Ok(())
}

fn identity(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| f32::from(index / columns == index % columns))
        .collect()
}

fn zeros(length: usize) -> Vec<f32> {
    vec![0.0; length]
}

fn ones(length: usize) -> Vec<f32> {
    vec![1.0; length]
}

fn config() -> SpatialMergeVisionConfig {
    SpatialMergeVisionConfig {
        hidden_size: 8,
        output_hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        in_channels: 3,
        patch_size: 1,
        temporal_patch_size: 2,
        spatial_merge_size: 1,
        num_position_embeddings: 4,
        hidden_activation: "gelu_pytorch_tanh".into(),
        image_token_id: 10,
        vision_start_token_id: 11,
        vision_end_token_id: 12,
        mrope_interleaved: true,
        mrope_sections: vec![1, 1, 2],
    }
}
