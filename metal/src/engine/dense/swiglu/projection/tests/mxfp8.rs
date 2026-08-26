use std::{fs, path::Path};

use models::weights::{
    BlockQuantization, LogicalTensorRole, TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::engine::Dtype;

#[test]
fn executes_pinned_mxfp8_words_and_e8m0_scales() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-mxfp8-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0; 32], &[1, 32])?.astype(Dtype::Bfloat16, &stream)?;

    let linear = BoundLinear::load(&tensors, &binding(), &stream)?;
    let output = linear.forward(&input, &stream)?;
    assert_eq!(output.dtype()?, Dtype::Bfloat16);
    assert_eq!(output.to_vec_f32(&stream)?, [33.0, -34.0]);
    assert!(linear.has_bias());

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: vec![2, 8],
        logical_shape: Some(vec![2, 32]),
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP8,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: Some("bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn write_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    for word in [0x3838_3838_u32; 8].into_iter().chain([0xb0b0_b0b0; 8]) {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    let weight_end = payload.len();
    payload.extend_from_slice(&[127_u8, 128]);
    let scales_end = payload.len();
    for value in [0x3f80_u16, 0xc000] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"U32","shape":[2,8],"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,1],"data_offsets":[{weight_end},{scales_end}]}},"bias":{{"dtype":"BF16","shape":[2],"data_offsets":[{scales_end},{end}]}}}}"#
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
