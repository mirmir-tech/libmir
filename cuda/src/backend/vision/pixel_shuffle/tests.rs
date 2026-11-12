use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::bf16;
use models::{
    layout::{
        ImageProcessorConfig, ImageTags, ModelLayout, PixelShuffleVisionConfig, VisionConfig,
    },
    vision::PixelShufflePreprocessedImage,
    weights::{TensorCatalog, TensorInfo, pixel_shuffle_encoder_tensor as encoder},
};

use super::CudaPixelShuffleVisionTower;
use crate::{CudaConfig, Result, backend::CudaBackend, checkpoint::load_vision_tensors};

#[test]
fn executes_a_complete_synthetic_pixel_shuffle_tower() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-pixel-shuffle-{}.bin", std::process::id()));
    let catalog = write_weights(&path)?;
    let result = execute(&config(), &catalog);
    let _removed = fs::remove_file(path);
    result
}

/// Runs a real checkpoint's tower on `LIBMIR_VISION_IMAGE` and writes the
/// projected soft tokens as little-endian f32 to `LIBMIR_VISION_TOWER_OUTPUT`.
#[test]
#[ignore = "loads a real vision checkpoint; set MODEL, LIBMIR_VISION_IMAGE and LIBMIR_VISION_TOWER_OUTPUT"]
fn records_a_real_pixel_shuffle_tower_output() -> Result<()> {
    let root = required_path("MODEL")?;
    let image_path = required_path("LIBMIR_VISION_IMAGE")?;
    let output_path = required_path("LIBMIR_VISION_TOWER_OUTPUT")?;
    let layout = ModelLayout::inspect(&root)?;
    let vision = VisionConfig::from_layout(&layout)?.ok_or_else(|| {
        crate::Error::UnsupportedVisionContract("checkpoint has no vision config".into())
    })?;
    let processor =
        ImageProcessorConfig::from_layout(&layout, vision.pipeline())?.ok_or_else(|| {
            crate::Error::UnsupportedVisionContract("checkpoint has no image processor".into())
        })?;
    let (VisionConfig::PixelShuffleEncoder(config), ImageProcessorConfig::PixelShuffle(processor)) =
        (vision, processor)
    else {
        return Err(crate::Error::UnsupportedVisionContract(
            "checkpoint is not pixel-shuffle vision".into(),
        ));
    };
    let image = processor.preprocess_encoded(&fs::read(image_path)?)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let tensors = load_vision_tensors(
        &backend,
        &VisionConfig::PixelShuffleEncoder(config.clone()),
        &catalog,
    )?;
    let output = CudaPixelShuffleVisionTower::new(&backend, config.clone(), tensors)?
        .forward_preprocessed(&image)?;
    assert_eq!((output.tokens, output.width), (image.soft_tokens, config.output_hidden_size));
    let values = read(&backend, &output.hidden)?;
    assert!(values.iter().all(|value| value.is_finite()));
    fs::write(
        output_path,
        values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| crate::Error::State(format!("{name} is unset")))
}

fn read(backend: &CudaBackend, source: &mircuda::DeviceBuffer<bf16>) -> Result<Vec<f32>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?.into_iter().map(bf16::to_f32).collect())
}

fn execute(config: &PixelShuffleVisionConfig, catalog: &TensorCatalog) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors =
        load_vision_tensors(&backend, &VisionConfig::PixelShuffleEncoder(config.clone()), catalog)?;
    let tower = CudaPixelShuffleVisionTower::new(&backend, config.clone(), tensors)?;
    // A 2×2 grid of 2×2-pixel patches folds into one soft token.
    let image = PixelShufflePreprocessedImage {
        patches: (0..48_u8).map(|index| f32::from(index % 7) / 7.0).collect(),
        grid_height: 2,
        grid_width: 2,
        soft_tokens: 1,
    };
    let mut steps = 0;
    let output = tower.forward_preprocessed_scheduled(&image, &mut |step| {
        steps += 1;
        step()
    })?;
    assert_eq!(steps, config.num_hidden_layers + 3);
    assert_eq!((output.tokens, output.width, output.hidden.len()), (1, 8, 8));
    assert!(read(&backend, &output.hidden)?.iter().all(|value| value.is_finite()));
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    let compiled = backend.inner.compiler.cache_stats();
    let output = tower.forward_preprocessed(&image)?;
    backend.synchronize()?;
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    assert_eq!(backend.inner.compiler.cache_stats().misses, compiled.misses);
    Ok(())
}

fn config() -> PixelShuffleVisionConfig {
    PixelShuffleVisionConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_channels: 3,
        patch_size: 2,
        prefix_tokens: 2,
        position_grid: 4,
        layer_norm_eps: 1e-6,
        hidden_activation: "gelu".into(),
        shuffle_factor: 2,
        projector_hidden_size: 16,
        projector_norm_eps: 1e-5,
        output_hidden_size: 8,
        final_layer_norm: true,
        image_token_id: 18,
        image_tags: ImageTags::Img,
    }
}

fn write_weights(path: &Path) -> Result<TensorCatalog> {
    let mut tensors = vec![
        tensor(&encoder("patch_generator.embedder.weight"), &[8, 12], 0.05),
        tensor(&encoder("patch_generator.pos_embed"), &[1, 16, 8], 0.01),
        tensor(&encoder("patch_generator.cls_token.token"), &[2, 8], 0.1),
        tensor("vision_projector.vision_final_layernorm.weight", &[8], 1.0),
        tensor("vision_projector.vision_final_layernorm.bias", &[8], 0.0),
        tensor("mlp1.0.weight", &[32], 1.0),
        tensor("mlp1.1.weight", &[16, 32], 0.03),
        tensor("mlp1.3.weight", &[8, 16], 0.03),
    ];
    for norm in ["norm1", "norm2"] {
        tensors.push(tensor(&encoder(&format!("blocks.0.{norm}.weight")), &[8], 1.0));
        tensors.push(tensor(&encoder(&format!("blocks.0.{norm}.bias")), &[8], 0.0));
    }
    for (suffix, shape) in
        [("attn.qkv", [24, 8]), ("attn.proj", [8, 8]), ("mlp.fc1", [16, 8]), ("mlp.fc2", [8, 16])]
    {
        tensors.push(tensor(&encoder(&format!("blocks.0.{suffix}.weight")), &shape, 0.02));
        tensors.push(tensor(&encoder(&format!("blocks.0.{suffix}.bias")), &shape[..1], 0.0));
    }
    let mut payload = Vec::new();
    let mut infos = Vec::with_capacity(tensors.len());
    for (name, shape, values) in tensors {
        let start = u64::try_from(payload.len())?;
        for value in values {
            payload.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        infos.push(TensorInfo {
            name,
            file: path.to_owned(),
            dtype: "BF16".into(),
            shape,
            data_start: 0,
            data_offsets: [start, u64::try_from(payload.len())?],
        });
    }
    fs::write(path, payload)?;
    Ok(TensorCatalog { tensors: infos })
}

fn tensor(name: &str, shape: &[usize], value: f32) -> (String, Vec<usize>, Vec<f32>) {
    (name.into(), shape.into(), vec![value; shape.iter().product()])
}
