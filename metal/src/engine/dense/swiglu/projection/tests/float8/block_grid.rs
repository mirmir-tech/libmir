use std::{fs, path::Path};

use models::weights::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode, LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::*;

#[test]
fn executes_exact_and_padded_block_grids() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-fp8-grid-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;

    let exact_input =
        Array::from_f32(&[1.0, 2.0], &[1, 2])?.astype(crate::engine::Dtype::Bfloat16, &stream)?;
    for activation in [Float8ActivationScale::None, Float8ActivationScale::DynamicToken] {
        let exact = BoundLinear::load(
            &tensors,
            &binding("exact", [2, 2], [2, 2], None, activation),
            &stream,
        )?;
        assert_eq!(
            exact.forward(&exact_input, &stream)?.to_vec_f32_on_stream(&stream)?,
            [14.0, 1.0]
        );
    }

    let padded_input = Array::from_f32(&[1.0, 2.0, 3.0], &[1, 3])?
        .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let padded = BoundLinear::load(
        &tensors,
        &binding("padded", [3, 3], [2, 2], Some([2, 2]), Float8ActivationScale::None),
        &stream,
    )?;
    assert_eq!(
        padded.forward(&padded_input, &stream)?.to_vec_f32_on_stream(&stream)?,
        [9.0, 9.0, 21.0]
    );

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding(
    source: &str,
    shape: [usize; 2],
    groups: [usize; 2],
    blocks: Option<[usize; 2]>,
    activation_scale: Float8ActivationScale,
) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: source.into(),
        shape: shape.to_vec(),
        logical_shape: Some(shape.to_vec()),
        transforms: Vec::new(),
        storage: TensorStorage::Float8 {
            format: Float8Quantization {
                format: Float8Format::E4M3,
                scale_mode: Float8ScaleMode::Multiplier,
                scale_granularity: Float8ScaleGranularity::BlockGrid {
                    output_groups: groups[0],
                    input_groups: groups[1],
                    output_block_size: blocks.map(|value| value[0]),
                    input_block_size: blocks.map(|value| value[1]),
                },
                scale_dtype: Some(Float8ParameterDType::F32),
                activation_scale,
                input_scale_dtype: None,
            },
            scale: Some(format!("{source}_scale")),
            input_scale: None,
            bias: None,
        },
    }
}

fn write_safetensors(path: &Path) -> Result<()> {
    let mut payload = vec![0x38, 0x40, 0xb8, 0x30];
    for scale in [2.0_f32, 3.0, 4.0, 5.0] {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    payload.extend_from_slice(&[0x38; 9]);
    for scale in [2.0_f32, 1.0, 4.0, 3.0] {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    let mut header = r#"{"exact":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]},"exact_scale":{"dtype":"F32","shape":[2,2],"data_offsets":[4,20]},"padded":{"dtype":"F8_E4M3","shape":[3,3],"data_offsets":[20,29]},"padded_scale":{"dtype":"F32","shape":[2,2],"data_offsets":[29,45]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}
