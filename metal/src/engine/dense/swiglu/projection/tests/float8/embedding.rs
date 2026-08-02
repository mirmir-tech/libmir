use super::*;

#[test]
fn executes_selected_e5m2_embedding_rows_on_metal() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-fp8-embedding-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let indices = Array::from_u32(&[1, 0], &[2])?;

    let scaled = BoundEmbedding::load(&tensors, &embedding_binding(true), &stream)?;
    assert_eq!(
        scaled.lookup(&indices, &stream)?.to_vec_f32_on_stream(&stream)?,
        [-4.0, 2.0, 2.0, 4.0]
    );
    let input =
        Array::from_f32(&[1.0, 2.0], &[1, 2])?.astype(crate::engine::Dtype::Bfloat16, &stream)?;
    assert_eq!(scaled.project(&input, &stream)?.to_vec_f32_on_stream(&stream)?, [10.0, 0.0]);

    let unscaled = BoundEmbedding::load(&tensors, &embedding_binding(false), &stream)?;
    assert_eq!(
        unscaled.lookup(&indices, &stream)?.to_vec_f32_on_stream(&stream)?,
        [-1.0, 0.5, 1.0, 2.0]
    );
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn embedding_binding(scaled: bool) -> TensorBinding {
    let mut binding = e5m2_binding(scaled);
    binding.role = LogicalTensorRole::Embedding;
    if let TensorStorage::Float8 { bias, .. } = &mut binding.storage {
        *bias = None;
    }
    binding
}
