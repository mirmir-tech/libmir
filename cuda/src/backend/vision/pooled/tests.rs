use std::{fs, path::Path};

use mircuda::bf16;
use models::{
    layout::{PooledVisionConfig, VisionConfig},
    vision::PooledPreprocessedImage,
    weights::{TensorCatalog, TensorInfo},
};

use super::CudaPooledVisionTower;
use crate::{CudaConfig, Result, backend::CudaBackend, checkpoint::load_vision_tensors};

#[test]
fn executes_a_complete_synthetic_pooled_tower_without_a_host_barrier() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-pooled-vision-{}.bin", std::process::id()));
    let config = config();
    let catalog = write_weights(&path)?;
    let result = execute(&config, &catalog);
    let _removed = fs::remove_file(path);
    result
}

fn execute(config: &PooledVisionConfig, catalog: &TensorCatalog) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors =
        load_vision_tensors(&backend, &VisionConfig::PooledEncoder(config.clone()), catalog)?;
    let tower = CudaPooledVisionTower::new(&backend, config.clone(), tensors)?;
    let image = PooledPreprocessedImage {
        patches: vec![1.0, 0.5, 0.0],
        position_ids: vec![0, 0],
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
    assert_eq!((output.tokens, output.width, output.hidden.len()), (1, 4, 4));
    let first_values = read(&backend, &output.hidden)?;
    assert!(first_values.iter().copied().map(bf16::to_f32).all(f32::is_finite));
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    let first_compile = backend.inner.compiler.cache_stats();
    let first_pool = backend.memory_pool_stats()?;

    let output = tower.forward_preprocessed_scheduled(&image, &mut |step| step())?;
    assert_eq!(tower.runner_pool_stats()?, (1, 0));
    assert_eq!(read(&backend, &output.hidden)?, first_values);
    drop(output);
    assert_eq!(tower.runner_pool_stats()?, (1, 1));
    let second_compile = backend.inner.compiler.cache_stats();
    let second_pool = backend.memory_pool_stats()?;
    assert_eq!(first_compile.misses, second_compile.misses);
    assert_eq!(first_compile.hits, second_compile.hits);
    assert_eq!(first_pool.used, second_pool.used);
    assert_eq!(first_pool.reserved, second_pool.reserved);

    let first = tower.forward_preprocessed_scheduled(&image, &mut |step| step())?;
    let second = tower.forward_preprocessed_scheduled(&image, &mut |step| step())?;
    assert_eq!(tower.runner_pool_stats()?, (2, 0));
    backend.synchronize()?;
    drop((first, second));
    assert_eq!(tower.runner_pool_stats()?, (2, 1));
    Ok(())
}

fn read(backend: &CudaBackend, source: &mircuda::DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn write_weights(path: &Path) -> Result<TensorCatalog> {
    let mut tensors = vec![
        tensor("model.vision_tower.patch_embedder.input_proj.weight", &[4, 3], identity(4, 3)),
        tensor(
            "model.vision_tower.patch_embedder.position_embedding_table",
            &[2, 2, 4],
            zeros(16),
        ),
        tensor("model.embed_vision.embedding_projection.weight", &[4, 4], identity(4, 4)),
    ];
    let layer = "model.vision_tower.encoder.layers.0";
    for name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ] {
        tensors.push(tensor(&format!("{layer}.{name}.weight"), &[4], ones(4)));
    }
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.push(tensor(&format!("{layer}.self_attn.{name}.weight"), &[4, 4], identity(4, 4)));
    }
    tensors.extend([
        tensor(&format!("{layer}.mlp.gate_proj.weight"), &[8, 4], zeros(32)),
        tensor(&format!("{layer}.mlp.up_proj.weight"), &[8, 4], zeros(32)),
        tensor(&format!("{layer}.mlp.down_proj.weight"), &[4, 8], zeros(32)),
    ]);
    catalog(path, tensors)
}

fn catalog(path: &Path, tensors: Vec<TestTensor>) -> Result<TensorCatalog> {
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

fn tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TestTensor {
    TestTensor {
        name: name.into(),
        shape: shape.into(),
        values,
    }
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

fn config() -> PooledVisionConfig {
    PooledVisionConfig {
        hidden_size: 4,
        output_hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        patch_size: 1,
        pooling_kernel_size: 1,
        position_embedding_size: 2,
        rms_norm_eps: 1.0e-6,
        rope_theta: 100.0,
        hidden_activation: "gelu_pytorch_tanh".into(),
        use_clipped_linears: false,
        standardize: false,
        image_token_id: 1,
        image_begin_token_id: 2,
        image_end_token_id: 3,
        soft_tokens_per_image: 1,
        bidirectional_image_attention: true,
    }
}
