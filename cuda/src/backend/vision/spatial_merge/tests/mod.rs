mod layout;

use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::bf16;
use models::{
    layout::{ImageProcessorConfig, ModelLayout, SpatialMergeVisionConfig, VisionConfig},
    vision::SpatialMergePreprocessedImage,
    weights::{TensorCatalog, TensorInfo},
};

use super::CudaSpatialMergeVisionTower;
use crate::{CudaConfig, Result, backend::CudaBackend, checkpoint::load_vision_tensors};

#[test]
fn executes_a_complete_synthetic_spatial_merge_tower() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-spatial-merge-{}.bin", std::process::id()));
    let config = config();
    let catalog = write_weights(&path)?;
    let result = execute(&config, &catalog);
    let _removed = fs::remove_file(path);
    result
}

#[test]
#[ignore = "loads a real vision checkpoint; set MODEL and LIBMIR_VISION_TOWER_OUTPUT"]
fn records_a_real_spatial_merge_tower_output() -> Result<()> {
    let root = required_path("MODEL")?;
    let output_path = required_path("LIBMIR_VISION_TOWER_OUTPUT")?;
    let layout = ModelLayout::inspect(&root)?;
    let vision = VisionConfig::from_layout(&layout)?.ok_or_else(|| {
        crate::Error::UnsupportedVisionContract("checkpoint has no vision config".into())
    })?;
    let processor =
        ImageProcessorConfig::from_layout(&layout, vision.pipeline())?.ok_or_else(|| {
            crate::Error::UnsupportedVisionContract("checkpoint has no image processor".into())
        })?;
    let (VisionConfig::SpatialMergeEncoder(config), ImageProcessorConfig::SpatialMerge(processor)) =
        (vision, processor)
    else {
        return Err(crate::Error::UnsupportedVisionContract(
            "checkpoint is not spatial-merge vision".into(),
        ));
    };
    let image = processor.preprocess_rgb(&comparison_rgb()?, 64, 64)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let tensors = load_vision_tensors(
        &backend,
        &VisionConfig::SpatialMergeEncoder(config.clone()),
        &catalog,
    )?;
    let output = CudaSpatialMergeVisionTower::new(&backend, config.clone(), tensors)?
        .forward_preprocessed(&image)?;
    assert_eq!((output.tokens, output.width), (image.soft_tokens, config.output_hidden_size));
    let values = read(&backend, &output.hidden)?;
    assert!(values.iter().all(|value| value.is_finite()));
    fs::write(output_path, f32_bytes(&values))?;
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| crate::Error::State(format!("{name} is unset")))
}

fn comparison_rgb() -> Result<Vec<u8>> {
    (0..64 * 64 * 3).map(|index| Ok(u8::try_from(index % 251)?)).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn read(backend: &CudaBackend, source: &mircuda::DeviceBuffer<bf16>) -> Result<Vec<f32>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?.into_iter().map(bf16::to_f32).collect())
}

fn execute(config: &SpatialMergeVisionConfig, catalog: &TensorCatalog) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors =
        load_vision_tensors(&backend, &VisionConfig::SpatialMergeEncoder(config.clone()), catalog)?;
    let tower = CudaSpatialMergeVisionTower::new(&backend, config.clone(), tensors)?;
    let image = SpatialMergePreprocessedImage {
        patches: vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        grid_t: 1,
        grid_height: 1,
        grid_width: 1,
        soft_tokens: 1,
    };
    let mut steps = 0;
    let output = tower.forward_preprocessed_scheduled(&image, &mut |step| {
        steps += 1;
        step()
    })?;
    assert_eq!(steps, config.num_hidden_layers + 3);
    assert_eq!((output.tokens, output.width, output.hidden.len()), (1, 8, 8));
    backend.synchronize()?;
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    let first_compile = backend.inner.compiler.cache_stats();
    let first_pool = backend.memory_pool_stats()?;
    let output = tower.forward_preprocessed(&image)?;
    assert_eq!((output.tokens, output.width, output.hidden.len()), (1, 8, 8));
    assert_eq!(tower.runner_pool_stats()?, (1, 0));
    backend.synchronize()?;
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    let second_compile = backend.inner.compiler.cache_stats();
    let second_pool = backend.memory_pool_stats()?;
    assert_eq!(first_compile.misses, second_compile.misses);
    assert_eq!(first_compile.hits, second_compile.hits);
    assert_eq!(first_pool.used, second_pool.used);
    assert_eq!(first_pool.reserved, second_pool.reserved);
    let first = tower.forward_preprocessed(&image)?;
    let second = tower.forward_preprocessed(&image)?;
    assert_eq!(tower.runner_pool_stats()?, (2, 0));
    backend.synchronize()?;
    drop((first, second));
    assert_eq!(tower.runner_pool_stats()?, (2, 1));
    Ok(())
}

fn write_weights(path: &Path) -> Result<TensorCatalog> {
    let mut tensors = vec![
        tensor("model.visual.patch_embed.proj.weight", &[8, 3, 2, 1, 1], 0.0),
        tensor("model.visual.patch_embed.proj.bias", &[8], 0.0),
        tensor("model.visual.pos_embed.weight", &[4, 8], 0.0),
        tensor("model.visual.merger.norm.weight", &[8], 1.0),
        tensor("model.visual.merger.norm.bias", &[8], 0.0),
        tensor("model.visual.merger.linear_fc1.weight", &[8, 8], 0.0),
        tensor("model.visual.merger.linear_fc1.bias", &[8], 0.0),
        tensor("model.visual.merger.linear_fc2.weight", &[8, 8], 0.0),
        tensor("model.visual.merger.linear_fc2.bias", &[8], 0.0),
    ];
    let layer = "model.visual.blocks.0";
    for norm in ["norm1", "norm2"] {
        tensors.push(tensor(&format!("{layer}.{norm}.weight"), &[8], 1.0));
        tensors.push(tensor(&format!("{layer}.{norm}.bias"), &[8], 0.0));
    }
    tensors.extend([
        tensor(&format!("{layer}.attn.qkv.weight"), &[24, 8], 0.0),
        tensor(&format!("{layer}.attn.qkv.bias"), &[24], 0.0),
        tensor(&format!("{layer}.attn.proj.weight"), &[8, 8], 0.0),
        tensor(&format!("{layer}.attn.proj.bias"), &[8], 0.0),
        tensor(&format!("{layer}.mlp.linear_fc1.weight"), &[8, 8], 0.0),
        tensor(&format!("{layer}.mlp.linear_fc1.bias"), &[8], 0.0),
        tensor(&format!("{layer}.mlp.linear_fc2.weight"), &[8, 8], 0.0),
        tensor(&format!("{layer}.mlp.linear_fc2.bias"), &[8], 0.0),
    ]);
    let mut payload = Vec::new();
    let mut infos = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let start = u64::try_from(payload.len())?;
        for value in tensor.values {
            payload.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        infos.push(TensorInfo {
            name: tensor.name,
            file: path.to_owned(),
            dtype: "BF16".into(),
            shape: tensor.shape,
            data_start: 0,
            data_offsets: [start, u64::try_from(payload.len())?],
        });
    }
    fs::write(path, payload)?;
    Ok(TensorCatalog { tensors: infos })
}

struct TestTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn tensor(name: &str, shape: &[usize], value: f32) -> TestTensor {
    TestTensor {
        name: name.into(),
        shape: shape.into(),
        values: vec![value; shape.iter().product()],
    }
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
