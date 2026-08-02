use std::{fs, path::Path};

use models::weights::{
    AwqBits, AwqPacking, AwqQuantization, AwqScaleDType, AwqStorageDType, LogicalTensorRole,
    TensorBinding, TensorStorage,
};

use super::*;

const INPUT: usize = 1_024;
const OUTPUT: usize = 8;
const GROUP: usize = 128;

#[test]
fn repacks_awq_on_gpu_and_executes_native_affine_qmm() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-awq-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_awq_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &binding(), &stream)?;
    let input = Array::from_f32(&[1.0; 2 * INPUT], &[2, i32::try_from(INPUT)?])?;
    let expected = (0..2)
        .flat_map(|_| (1_u16..=8).map(|row| 512.0 * f32::from(row)))
        .collect::<Vec<_>>();
    assert_eq!(linear.forward(&input, &stream)?.to_vec_f32_on_stream(&stream)?, expected);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "qweight".into(),
        shape: vec![INPUT, OUTPUT / 8],
        logical_shape: Some(vec![OUTPUT, INPUT]),
        transforms: Vec::new(),
        storage: TensorStorage::Awq {
            format: AwqQuantization {
                bits: AwqBits::Four,
                group_size: GROUP,
                packing: AwqPacking::GemmOutputInterleaved,
                storage_dtype: AwqStorageDType::I32,
                scale_dtype: AwqScaleDType::F16,
                packed_zero_points: true,
            },
            scales: "scales".into(),
            zero_points: "qzeros".into(),
        },
    }
}

fn write_awq_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    let weight_word = pack(std::array::from_fn(|row| u8::try_from(row + 3).unwrap_or_default()));
    for _ in 0..INPUT {
        payload.extend_from_slice(&weight_word.to_le_bytes());
    }
    let weight_end = payload.len();
    let zero_word = pack([2; OUTPUT]);
    for _ in 0..INPUT / GROUP {
        payload.extend_from_slice(&zero_word.to_le_bytes());
    }
    let zero_end = payload.len();
    for _ in 0..INPUT / GROUP * OUTPUT {
        payload.extend_from_slice(&0x3800_u16.to_le_bytes());
    }
    let scale_end = payload.len();
    let mut header = format!(
        r#"{{"qweight":{{"dtype":"I32","shape":[{INPUT},1],"data_offsets":[0,{weight_end}]}},"qzeros":{{"dtype":"I32","shape":[{},1],"data_offsets":[{weight_end},{zero_end}]}},"scales":{{"dtype":"F16","shape":[{},{OUTPUT}],"data_offsets":[{zero_end},{scale_end}]}}}}"#,
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

fn pack(values: [u8; OUTPUT]) -> i32 {
    let word = values
        .into_iter()
        .enumerate()
        .fold(0_u32, |word, (row, value)| word | (u32::from(value) << shift(row)));
    i32::from_ne_bytes(word.to_ne_bytes())
}

const fn shift(row: usize) -> usize {
    (((row & 1) << 2) | (row >> 1)) * 4
}
