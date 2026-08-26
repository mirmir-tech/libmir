use std::{fs, path::Path};

use models::{
    layout::{PooledImageProcessorConfig, PooledVisionConfig},
    weights::{LogicalTensorRole, TensorBinding, TensorStorage},
};
use serde_json::{Map, Value, json};

use super::{PooledVisionTower, pooler::VisionPooler, rope::VisionRope};
use crate::engine::{Array, ModelTensors, Result, Stream};

#[test]
fn applies_independent_row_and_column_rope() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 4])?;
    let positions = Array::from_u32(&[0, 0, 1, 0], &[1, 2, 2])?;
    let rope = VisionRope::new(4, 100.0)?;
    let (query, key) = rope.apply(&input, &input, &positions, &stream)?;
    let values = query.to_vec_f32(&stream)?;

    assert_close(&values[..4], &[1.0, 2.0, 3.0, 4.0], 1.0e-6);
    assert_close(&values[4..], &[1.0_f32.cos(), 1.0_f32.sin(), 0.0, 1.0], 1.0e-6);
    assert_close(&values, &key.to_vec_f32(&stream)?, 1.0e-6);
    Ok(())
}

#[test]
fn pools_a_two_dimensional_window_before_projection() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let hidden = Array::from_f32(&[3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], &[1, 4, 2])?;
    let identity = Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let pooler = VisionPooler::from_projection(&identity, 2, 2, 0.0, &stream)?;
    let output = pooler.forward(&hidden, 2, 2, &stream)?;

    assert_eq!(output.shape()?, vec![1, 1, 2]);
    assert_close(&output.to_vec_f32(&stream)?, &[0.848_528_15, 1.131_370_9], 1.0e-6);
    Ok(())
}

#[test]
fn loads_and_executes_a_complete_synthetic_tower() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-pooled-vision-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_tower_weights(&root.join("model.safetensors"))?;

    let result = execute_synthetic_tower(&root);
    fs::remove_dir_all(root)?;
    result
}

fn execute_synthetic_tower(root: &Path) -> Result<()> {
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let projection = projection_binding();
    let tower = PooledVisionTower::load(&tensors, &test_config(), &projection, &stream)?;
    let image = test_processor().preprocess_rgb(&[255, 128, 0], 1, 1)?;
    let output = tower.forward_preprocessed(&image, &stream)?;

    assert_eq!(output.shape()?, vec![1, 1, 4]);
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    Ok(())
}

fn projection_binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::VisionProjection,
        source: "embed_vision.embedding_projection.weight".into(),
        shape: vec![4, 4],
        logical_shape: Some(vec![4, 4]),
        transforms: Vec::new(),
        storage: TensorStorage::Dense { dtype: "F32".into(), bias: None },
    }
}

fn write_tower_weights(path: &Path) -> Result<()> {
    let mut tensors = vec![
        tensor("vision_tower.patch_embedder.input_proj.weight", &[4, 3], identity(4, 3)),
        tensor("vision_tower.patch_embedder.position_embedding_table", &[2, 2, 4], zeros(16)),
        tensor("vision_tower.std_bias", &[4], zeros(4)),
        tensor("vision_tower.std_scale", &[4], vec![1.0; 4]),
        tensor("embed_vision.embedding_projection.weight", &[4, 4], identity(4, 4)),
    ];
    let prefix = "vision_tower.encoder.layers.0";
    for name in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ] {
        tensors.push(tensor(&format!("{prefix}.{name}.weight"), &[4], vec![1.0; 4]));
    }
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        push_clippable(
            &mut tensors,
            &format!("{prefix}.self_attn.{name}"),
            &[4, 4],
            identity(4, 4),
        );
    }
    push_clippable(&mut tensors, &format!("{prefix}.mlp.gate_proj"), &[8, 4], zeros(32));
    push_clippable(&mut tensors, &format!("{prefix}.mlp.up_proj"), &[8, 4], zeros(32));
    push_clippable(&mut tensors, &format!("{prefix}.mlp.down_proj"), &[4, 8], zeros(32));
    write_safetensors(path, &tensors)
}

fn push_clippable(tensors: &mut Vec<TestTensor>, prefix: &str, shape: &[usize], values: Vec<f32>) {
    tensors.push(tensor(&format!("{prefix}.linear.weight"), shape, values));
    for (suffix, value) in [
        ("input_min", f32::NEG_INFINITY),
        ("input_max", f32::INFINITY),
        ("output_min", f32::NEG_INFINITY),
        ("output_max", f32::INFINITY),
    ] {
        tensors.push(tensor(&format!("{prefix}.{suffix}"), &[], vec![value]));
    }
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
        .map(|index| {
            if index / columns == index % columns {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn zeros(length: usize) -> Vec<f32> {
    vec![0.0; length]
}

fn test_config() -> PooledVisionConfig {
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
        use_clipped_linears: true,
        standardize: true,
        image_token_id: 1,
        image_begin_token_id: 2,
        image_end_token_id: 3,
        soft_tokens_per_image: 1,
        bidirectional_image_attention: true,
    }
}

fn test_processor() -> PooledImageProcessorConfig {
    PooledImageProcessorConfig {
        patch_size: 1,
        pooling_kernel_size: 1,
        max_soft_tokens: 70,
        rescale_factor: 1.0 / 255.0,
        do_resize: false,
        do_rescale: true,
        do_normalize: false,
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance),
        "actual {actual:?} differs from expected {expected:?}"
    );
}
