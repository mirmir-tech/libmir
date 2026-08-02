use std::{fs, path::Path};

use models::weights::{
    CompressedIntegerActivationOrder, CompressedIntegerBits, CompressedIntegerPacking,
    CompressedIntegerQuantization, CompressedIntegerScaleDType, CompressedIntegerScaleStrategy,
    CompressedIntegerSignedness, CompressedIntegerStorageDType, CompressedIntegerZeroPointMode,
    LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::*;

const INPUT: usize = 1_024;
const OUTPUT: usize = 2;

#[test]
fn executes_packed_int8_embedding_and_output_with_native_affine_qmm() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-packed-int8-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_int8_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;

    let embedding =
        BoundEmbedding::load(&tensors, &int8_binding(LogicalTensorRole::Embedding), &stream)?;
    let selected = Array::from_u32(&[1], &[1])?;
    assert_eq!(
        embedding.lookup(&selected, &stream)?.to_vec_f32_on_stream(&stream)?,
        [0.75; INPUT]
    );

    let output = BoundLinear::load(&tensors, &int8_binding(LogicalTensorRole::Output), &stream)?;
    let input = Array::from_f32(&[1.0; INPUT], &[1, i32::try_from(INPUT)?])?;
    assert_eq!(
        output.forward(&input, &stream)?.to_vec_f32_on_stream(&stream)?,
        [-1_024.0, 768.0]
    );
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn executes_packed_int4_embedding_and_output_with_native_affine_qmm() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-packed-int4-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_int4_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;

    let embedding =
        BoundEmbedding::load(&tensors, &int4_binding(LogicalTensorRole::Embedding), &stream)?;
    let selected = Array::from_u32(&[1], &[1])?;
    assert_eq!(
        embedding.lookup(&selected, &stream)?.to_vec_f32_on_stream(&stream)?,
        [0.75; INPUT]
    );

    let output = BoundLinear::load(&tensors, &int4_binding(LogicalTensorRole::Output), &stream)?;
    let input = Array::from_f32(&[1.0; INPUT], &[1, i32::try_from(INPUT)?])?;
    assert_eq!(
        output.forward(&input, &stream)?.to_vec_f32_on_stream(&stream)?,
        [-1_024.0, 768.0]
    );
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn int8_binding(role: LogicalTensorRole) -> TensorBinding {
    TensorBinding {
        role,
        source: "weight_packed".into(),
        shape: vec![OUTPUT, INPUT / 4],
        logical_shape: Some(vec![OUTPUT, INPUT]),
        transforms: Vec::new(),
        storage: TensorStorage::PackedInt8 {
            format: CompressedIntegerQuantization {
                bits: CompressedIntegerBits::Eight,
                scale_strategy: CompressedIntegerScaleStrategy::Channel,
                signedness: CompressedIntegerSignedness::OffsetBinary,
                zero_point: CompressedIntegerZeroPointMode::None,
                activation_order: CompressedIntegerActivationOrder::None,
                packing: CompressedIntegerPacking::DenseLittleEndian,
                storage_dtype: CompressedIntegerStorageDType::I32,
                scale_dtype: CompressedIntegerScaleDType::BF16,
            },
            scales: "weight_scale".into(),
            shape: "weight_shape".into(),
            zero_points: None,
            group_indices: None,
        },
    }
}

fn int4_binding(role: LogicalTensorRole) -> TensorBinding {
    TensorBinding {
        role,
        source: "weight_packed".into(),
        shape: vec![OUTPUT, INPUT / 8],
        logical_shape: Some(vec![OUTPUT, INPUT]),
        transforms: Vec::new(),
        storage: TensorStorage::PackedInt4 {
            format: CompressedIntegerQuantization {
                bits: CompressedIntegerBits::Four,
                scale_strategy: CompressedIntegerScaleStrategy::Group { group_size: 128 },
                signedness: CompressedIntegerSignedness::OffsetBinary,
                zero_point: CompressedIntegerZeroPointMode::None,
                activation_order: CompressedIntegerActivationOrder::None,
                packing: CompressedIntegerPacking::DenseLittleEndian,
                storage_dtype: CompressedIntegerStorageDType::I32,
                scale_dtype: CompressedIntegerScaleDType::BF16,
            },
            scales: "weight_scale".into(),
            shape: "weight_shape".into(),
            zero_points: None,
            group_indices: None,
        },
    }
}

fn write_int8_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    append_int8_row(&mut payload, -2);
    append_int8_row(&mut payload, 3);
    let weight_end = payload.len();
    for bits in [0x3f00_u16, 0x3e80] {
        payload.extend_from_slice(&bits.to_le_bytes());
    }
    write_safetensors(path, &payload, weight_end, INPUT / 4, 1)
}

fn write_int4_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    append_int4_row(&mut payload, -2);
    append_int4_row(&mut payload, 3);
    let weight_end = payload.len();
    for bits in [0x3f00_u16, 0x3e80] {
        for _ in 0..INPUT / 128 {
            payload.extend_from_slice(&bits.to_le_bytes());
        }
    }
    write_safetensors(path, &payload, weight_end, INPUT / 8, INPUT / 128)
}

fn write_safetensors(
    path: &Path,
    payload: &[u8],
    weight_end: usize,
    weight_width: usize,
    scale_width: usize,
) -> Result<()> {
    let scale_end = payload.len();
    let mut header = format!(
        r#"{{"weight_packed":{{"dtype":"I32","shape":[{OUTPUT},{weight_width}],"data_offsets":[0,{weight_end}]}},"weight_scale":{{"dtype":"BF16","shape":[{OUTPUT},{scale_width}],"data_offsets":[{weight_end},{scale_end}]}}}}"#
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(payload);
    fs::write(path, data)?;
    Ok(())
}

fn append_int8_row(bytes: &mut Vec<u8>, value: i8) {
    let encoded = value.to_ne_bytes()[0].wrapping_add(128);
    let word = i32::from_le_bytes([encoded; 4]);
    for _ in 0..INPUT / 4 {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}

fn append_int4_row(bytes: &mut Vec<u8>, value: i8) {
    let encoded = u32::from(value.to_ne_bytes()[0].wrapping_add(8));
    let word = (0..8).fold(0, |word, shift| word | (encoded << (shift * 4)));
    for _ in 0..INPUT / 8 {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}
