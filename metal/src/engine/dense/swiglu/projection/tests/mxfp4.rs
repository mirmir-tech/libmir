use std::{fs, path::Path};

use models::weights::{
    BlockQuantization, LogicalTensorRole, TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::engine::Dtype;

#[test]
fn executes_pinned_mxfp4_blocks_and_e8m0_scales() -> Result<()> {
    execute(BlockQuantization::MXFP4, "U8", &[2, 1, 16], "u8")
}

#[test]
fn executes_mlx_u32_mxfp4_without_repacking() -> Result<()> {
    execute(BlockQuantization::MXFP4_MLX, "U32", &[2, 4], "u32")
}

fn execute(format: BlockQuantization, dtype: &str, shape: &[usize], label: &str) -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-mxfp4-{label}-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"), dtype, shape)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0; 32], &[1, 32])?.astype(Dtype::Bfloat16, &stream)?;

    let linear = BoundLinear::load(&tensors, &binding(format, shape, true), &stream)?;
    let output = linear.forward(&input, &stream)?;
    assert_eq!(output.dtype()?, Dtype::Bfloat16);
    assert_eq!(output.to_vec_f32_on_stream(&stream)?, [33.0, 94.0]);
    assert!(linear.has_bias());

    let embedding = BoundEmbedding::load(&tensors, &binding(format, shape, false), &stream)?;
    let selected = Array::from_u32(&[1], &[1])?;
    let embedded = embedding.lookup(&selected, &stream)?;
    assert_eq!(embedded.dtype()?, Dtype::Bfloat16);
    assert_eq!(embedded.to_vec_f32_on_stream(&stream)?, [3.0; 32]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding(format: BlockQuantization, shape: &[usize], with_bias: bool) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: shape.into(),
        logical_shape: Some(vec![2, 32]),
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: with_bias.then(|| "bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn write_safetensors(path: &Path, dtype: &str, shape: &[usize]) -> Result<()> {
    let mut payload = [0x22_u8; 16].into_iter().chain([0x33_u8; 16]).collect::<Vec<_>>();
    let weight_end = payload.len();
    payload.extend([127_u8, 128]);
    let scales_end = payload.len();
    for value in [0x3f80_u16, 0xc000] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,1],"data_offsets":[{weight_end},{scales_end}]}},"bias":{{"dtype":"BF16","shape":[2],"data_offsets":[{scales_end},{end}]}}}}"#
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}
