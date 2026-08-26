use std::{fs, path::Path};

use models::weights::{
    GptqBits, GptqCheckpointFormat, GptqPacking, GptqQuantization, GptqScaleDType,
    GptqStorageDType, LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::*;

const INPUT: usize = 1_024;
const OUTPUT: usize = 8;
const GROUP: usize = 128;

#[test]
fn executes_gptq_v1_and_v2_with_contiguous_and_ordered_groups() -> Result<()> {
    for format in [GptqCheckpointFormat::Gptq, GptqCheckpointFormat::GptqV2] {
        for activation_order in [false, true] {
            check(format, activation_order)?;
        }
    }
    Ok(())
}

fn check(format: GptqCheckpointFormat, activation_order: bool) -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-gptq-{}-{format:?}-{activation_order}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_gptq_safetensors(&root.join("model.safetensors"), format, activation_order)?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &binding(format, activation_order), &stream)?;
    let input = Array::from_f32(&[1.0; 2 * INPUT], &[2, i32::try_from(INPUT)?])?;
    let expected = (0..2)
        .flat_map(|_| (0..OUTPUT).map(|row| expected(row, activation_order)))
        .collect::<Vec<_>>();
    assert_eq!(linear.forward(&input, &stream)?.to_vec_f32(&stream)?, expected);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding(checkpoint_format: GptqCheckpointFormat, activation_order: bool) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "qweight".into(),
        shape: vec![INPUT / 8, OUTPUT],
        logical_shape: Some(vec![OUTPUT, INPUT]),
        transforms: Vec::new(),
        storage: TensorStorage::Gptq {
            format: GptqQuantization {
                bits: GptqBits::Four,
                group_size: GROUP,
                packing: GptqPacking::InputLittleEndian,
                storage_dtype: GptqStorageDType::I32,
                scale_dtype: GptqScaleDType::F16,
                checkpoint_format,
                symmetric: true,
                activation_order,
                packed_zero_points: true,
            },
            scales: "scales".into(),
            zero_points: "qzeros".into(),
            group_indices: "g_idx".into(),
        },
    }
}

fn write_gptq_safetensors(
    path: &Path,
    format: GptqCheckpointFormat,
    activation_order: bool,
) -> Result<()> {
    let mut payload = Vec::new();
    for _ in 0..INPUT / 8 {
        for row in 0..OUTPUT {
            let nibble = u32::try_from(row + 3)?;
            let word = (0..8).fold(0_u32, |word, feature| word | (nibble << (feature * 4)));
            payload.extend_from_slice(&word.to_le_bytes());
        }
    }
    let weight_end = payload.len();
    for group in 0..INPUT / GROUP {
        let zero = zero(group);
        let encoded = match format {
            GptqCheckpointFormat::Gptq => zero.wrapping_sub(1) & 15,
            GptqCheckpointFormat::GptqV2 => zero,
        };
        let zero_word = (0..8).fold(0_u32, |word, lane| word | (encoded << (lane * 4)));
        payload.extend_from_slice(&zero_word.to_le_bytes());
    }
    let zero_end = payload.len();
    for group in 0..INPUT / GROUP {
        for _ in 0..OUTPUT {
            let bits = if group.is_multiple_of(2) {
                0x3800_u16
            } else {
                0x3400_u16
            };
            payload.extend_from_slice(&bits.to_le_bytes());
        }
    }
    let scale_end = payload.len();
    for feature in 0..INPUT {
        payload.extend_from_slice(&i32::try_from(group(feature, activation_order))?.to_le_bytes());
    }
    let index_end = payload.len();
    let mut header = format!(
        r#"{{"qweight":{{"dtype":"I32","shape":[{},{OUTPUT}],"data_offsets":[0,{weight_end}]}},"qzeros":{{"dtype":"I32","shape":[{},1],"data_offsets":[{weight_end},{zero_end}]}},"scales":{{"dtype":"F16","shape":[{},{OUTPUT}],"data_offsets":[{zero_end},{scale_end}]}},"g_idx":{{"dtype":"I32","shape":[{INPUT}],"data_offsets":[{scale_end},{index_end}]}}}}"#,
        INPUT / 8,
        INPUT / GROUP,
        INPUT / GROUP,
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

fn group(feature: usize, activation_order: bool) -> usize {
    if activation_order {
        (feature / 16) % (INPUT / GROUP)
    } else {
        feature / GROUP
    }
}

fn zero(group: usize) -> u32 {
    u32::try_from(group % 4 + 1).unwrap_or_default()
}

fn scale(group: usize) -> f32 {
    if group.is_multiple_of(2) {
        0.5
    } else {
        0.25
    }
}

fn expected(row: usize, activation_order: bool) -> f32 {
    (0..INPUT)
        .map(|feature| {
            let group = group(feature, activation_order);
            (f32::from(u16::try_from(row + 3).unwrap_or_default())
                - f32::from(u8::try_from(zero(group)).unwrap_or_default()))
                * scale(group)
        })
        .sum()
}
