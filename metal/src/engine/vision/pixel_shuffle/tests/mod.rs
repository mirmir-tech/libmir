//! The Metal tower against a host reference of the same encoder on random
//! weights: a non-square grid, so the position table is both resized and read
//! off its square, two blocks, the final norm and a 2×2 shuffle.

mod real;
mod reference;

use std::{collections::BTreeMap, fs, path::Path};

use models::{
    layout::{ImageTags, PixelShuffleVisionConfig},
    vision::PixelShufflePreprocessedImage,
};
use serde_json::{Map, Value, json};

use super::{PixelShuffleVisionTower, encoder};
use crate::engine::{ModelTensors, Result, Stream};

pub(super) type Weights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

pub(super) fn config() -> PixelShuffleVisionConfig {
    PixelShuffleVisionConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_channels: 3,
        patch_size: 2,
        prefix_tokens: 2,
        position_grid: 3,
        layer_norm_eps: 1e-6,
        hidden_activation: models::layout::HiddenActivation::Gelu,
        shuffle_factor: 2,
        projector_hidden_size: 12,
        projector_norm_eps: 1e-5,
        output_hidden_size: 6,
        final_layer_norm: true,
        image_token_id: 18,
        image_tags: ImageTags::Img,
    }
}

fn values(count: usize, seed: &mut u32, spread: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f32::from(u16::try_from(*seed >> 16).unwrap_or(0)) / 32_768.0 - 1.0) * spread
        })
        .collect()
}

fn weights(config: &PixelShuffleVisionConfig) -> Weights {
    let (h, i, p) = (config.hidden_size, config.intermediate_size, 3 * 2 * 2);
    let mut seed = 7;
    let mut weights = Weights::new();
    let mut add = |name: String, shape: Vec<usize>, spread: f32, offset: f32| {
        let mut data = values(shape.iter().product(), &mut seed, spread);
        for value in &mut data {
            *value += offset;
        }
        weights.insert(name, (shape, data));
    };
    add(encoder("patch_generator.embedder.weight"), vec![h, p], 0.4, 0.0);
    add(encoder("patch_generator.pos_embed"), vec![1, 9, h], 0.5, 0.0);
    add(encoder("patch_generator.cls_token.token"), vec![2, h], 0.5, 0.0);
    for layer in 0..config.num_hidden_layers {
        let name = |suffix: &str| encoder(&format!("blocks.{layer}.{suffix}"));
        for norm in ["norm1", "norm2"] {
            add(name(&format!("{norm}.weight")), vec![h], 0.2, 1.0);
            add(name(&format!("{norm}.bias")), vec![h], 0.1, 0.0);
        }
        for (linear, out, input) in
            [("attn.qkv", 3 * h, h), ("attn.proj", h, h), ("mlp.fc1", i, h), ("mlp.fc2", h, i)]
        {
            add(name(&format!("{linear}.weight")), vec![out, input], 0.4, 0.0);
            add(name(&format!("{linear}.bias")), vec![out], 0.1, 0.0);
        }
    }
    add("vision_projector.vision_final_layernorm.weight".into(), vec![h], 0.2, 1.0);
    add("vision_projector.vision_final_layernorm.bias".into(), vec![h], 0.1, 0.0);
    add("mlp1.0.weight".into(), vec![4 * h], 0.2, 1.0);
    add("mlp1.1.weight".into(), vec![12, 4 * h], 0.3, 0.0);
    add("mlp1.3.weight".into(), vec![6, 12], 0.3, 0.0);
    weights
}

#[test]
fn tower_matches_a_host_reference_on_a_non_square_grid() -> Result<()> {
    let config = config();
    let weights = weights(&config);
    let image = PixelShufflePreprocessedImage {
        patches: values(4 * 2 * 12, &mut 11, 1.0),
        grid_height: 4,
        grid_width: 2,
        soft_tokens: 2,
    };
    let root = std::env::temp_dir().join(format!("libmir-pixel-shuffle-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"), &weights)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?);
    let stream = Stream::new_gpu()?;
    let output = tensors.and_then(|tensors| {
        PixelShuffleVisionTower::load(&tensors, &config, &stream)?
            .forward_preprocessed(&image, &stream)
    });
    fs::remove_dir_all(&root)?;
    let output = output?;

    assert_eq!(output.shape()?, vec![1, 2, 6]);
    let expected = reference::tower(&config, &weights, &image);
    let actual = output.to_vec_f32(&stream)?;
    for (expected, actual) in expected.iter().zip(&actual) {
        assert!((expected - actual).abs() < 2e-3, "{expected} != {actual}");
    }
    Ok(())
}

fn write_safetensors(path: &Path, weights: &Weights) -> Result<()> {
    let mut header = Map::new();
    let mut payload = Vec::new();
    for (name, (shape, data)) in weights {
        let start = payload.len();
        for value in data {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
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
