use std::{fs, path::Path};

use models::weights::{
    BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
    BitsAndBytesStorageDType, LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::*;
use crate::engine::Dtype;

#[test]
fn executes_fp4_with_direct_scales() -> Result<()> {
    execute(false, BitsAndBytes4BitType::Fp4, "U8", &[64, 1], Dtype::Float16)
}

#[test]
fn executes_nf4_with_nested_scales_and_bf16_container() -> Result<()> {
    execute(true, BitsAndBytes4BitType::Nf4, "BF16", &[32, 1], Dtype::Bfloat16)
}

fn execute(
    nested: bool,
    kind: BitsAndBytes4BitType,
    dtype: &str,
    shape: &[usize],
    activation_dtype: Dtype,
) -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-bnb4-{}-{}",
        kind.as_str(),
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"), nested, kind, dtype, shape)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0; 64], &[1, 64])?.astype(activation_dtype, &stream)?;
    let output = BoundLinear::load(&tensors, &binding(nested, kind, dtype, shape), &stream)?
        .forward(&input, &stream)?;
    assert_eq!(output.dtype()?, activation_dtype);
    assert_eq!(output.to_vec_f32(&stream)?, [128.0, -192.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding(
    nested: bool,
    kind: BitsAndBytes4BitType,
    dtype: &str,
    shape: &[usize],
) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: shape.into(),
        logical_shape: Some(vec![2, 64]),
        transforms: Vec::new(),
        storage: TensorStorage::BitsAndBytes4Bit {
            format: BitsAndBytes4BitQuantization {
                quant_type: kind,
                block_size: 64,
                compute_dtype: BitsAndBytesComputeDType::Bf16,
                storage_dtype: if dtype == "U8" {
                    BitsAndBytesStorageDType::U8
                } else {
                    BitsAndBytesStorageDType::Bf16
                },
                nested_block_size: nested.then_some(256),
            },
            absmax: "absmax".into(),
            quant_map: "quant_map".into(),
            nested_absmax: nested.then(|| "nested_absmax".into()),
            nested_quant_map: nested.then(|| "nested_quant_map".into()),
            quant_state: "quant_state".into(),
            nested_offset_bits: nested.then(|| 0.0_f32.to_bits()),
        },
    }
}

fn write_safetensors(
    path: &Path,
    nested: bool,
    kind: BitsAndBytes4BitType,
    dtype: &str,
    shape: &[usize],
) -> Result<()> {
    let positive = if kind == BitsAndBytes4BitType::Fp4 {
        3
    } else {
        15
    };
    let negative = if kind == BitsAndBytes4BitType::Fp4 {
        11
    } else {
        0
    };
    let mut payload = vec![(positive << 4) | positive; 32];
    payload.extend(vec![(negative << 4) | negative; 32]);
    let weight_end = payload.len();
    if nested {
        payload.extend([0_u8, 1]);
    } else {
        append_f32(&mut payload, &[2.0, 3.0]);
    }
    let absmax_end = payload.len();
    let codebook = match kind {
        BitsAndBytes4BitType::Fp4 => fp4(),
        BitsAndBytes4BitType::Nf4 => nf4(),
    };
    append_f32(&mut payload, &codebook);
    let map_end = payload.len();
    if nested {
        append_f32(&mut payload, &[1.0]);
    }
    let nested_absmax_end = payload.len();
    if nested {
        let mut map = [0.0_f32; 256];
        map[0] = 2.0;
        map[1] = 3.0;
        append_f32(&mut payload, &map);
    }
    let end = payload.len();
    let absmax_dtype = if nested {
        "U8"
    } else {
        "F32"
    };
    let absmax_bytes = absmax_end - weight_end;
    let mut header = format!(
        r#"{{"weight":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{weight_end}]}},"absmax":{{"dtype":"{absmax_dtype}","shape":[2],"data_offsets":[{weight_end},{absmax_end}]}},"quant_map":{{"dtype":"F32","shape":[16],"data_offsets":[{absmax_end},{map_end}]}},"nested_absmax":{{"dtype":"F32","shape":[{}],"data_offsets":[{map_end},{nested_absmax_end}]}},"nested_quant_map":{{"dtype":"F32","shape":[{}],"data_offsets":[{nested_absmax_end},{end}]}}}}"#,
        usize::from(nested),
        if nested {
            256
        } else {
            0
        },
    );
    debug_assert_eq!(
        absmax_bytes,
        if nested {
            2
        } else {
            8
        }
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

fn append_f32(output: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn fp4() -> [f32; 16] {
    [
        0.0,
        1.0 / 192.0,
        2.0 / 3.0,
        1.0,
        1.0 / 3.0,
        0.5,
        1.0 / 6.0,
        0.25,
        -0.0,
        -1.0 / 192.0,
        -2.0 / 3.0,
        -1.0,
        -1.0 / 3.0,
        -0.5,
        -1.0 / 6.0,
        -0.25,
    ]
}

fn nf4() -> [f32; 16] {
    [
        -1.0, -0.696_192_8, -0.525_073_05, -0.394_917_5, -0.284_441_38, -0.184_773_43,
        -0.091_050_04, 0.0, 0.079_580_3, 0.160_930_2, 0.246_112_3, 0.337_915_24, 0.440_709_83,
        0.562_617, 0.722_956_84, 1.0,
    ]
}
