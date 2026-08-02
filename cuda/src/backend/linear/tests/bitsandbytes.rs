use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::{
    BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
    BitsAndBytesStorageDType, LogicalTensorRole, TensorBinding, TensorInfo, TensorStorage,
};

use super::*;
use crate::backend::linear::bitsandbytes::{BitsAndBytes4BitBf16Linear, BitsAndBytes4BitWeight};

#[test]
fn executes_direct_fp4_gemv() -> Result<()> {
    execute(false, BitsAndBytes4BitType::Fp4, "U8", &[64, 1], 1)
}

#[test]
fn executes_nested_nf4_qmm_from_bf16_container() -> Result<()> {
    execute(true, BitsAndBytes4BitType::Nf4, "BF16", &[32, 1], 2)
}

fn execute(
    nested: bool,
    kind: BitsAndBytes4BitType,
    dtype: &str,
    shape: &[usize],
    tokens: usize,
) -> Result<()> {
    let path = temp_path(kind);
    let (bytes, infos) = fixture(&path, nested, kind, dtype, shape)?;
    fs::write(&path, bytes)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    let binding = binding(nested, kind, dtype, shape);
    let weight = BitsAndBytes4BitWeight::load_binding(&tensors, &binding, 64, 2)?;
    let linear = BitsAndBytes4BitBf16Linear::new(&backend, tokens, &weight)?;
    let mut host_input = backend.inner.context.allocate_pinned::<bf16>(tokens * 64)?;
    host_input.copy_from_slice(&vec![bf16::from_f32(1.0); tokens * 64])?;
    let mut input = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, tokens * 64)?;
    backend.inner.stream.copy_to_device(&mut host_input, &mut input)?;
    let mut output =
        backend.inner.pool.allocate_zeroed::<bf16>(&backend.inner.stream, tokens * 2)?;
    linear.execute(&input, &weight, &mut output)?;
    let mut host = backend.inner.context.allocate_pinned::<bf16>(tokens * 2)?;
    backend.inner.stream.copy_to_host(&output, &mut host)?;
    let expected = (0..tokens)
        .flat_map(|_| [128.0, -192.0])
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    assert_eq!(host.to_vec()?, expected);
    fs::remove_file(path)?;
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

fn fixture(
    path: &std::path::Path,
    nested: bool,
    kind: BitsAndBytes4BitType,
    dtype: &str,
    shape: &[usize],
) -> Result<(Vec<u8>, Vec<TensorInfo>)> {
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
    let mut bytes = vec![(positive << 4) | positive; 32];
    bytes.extend(vec![(negative << 4) | negative; 32]);
    let weight_end = u64::try_from(bytes.len())?;
    if nested {
        bytes.extend([0_u8, 1]);
    } else {
        append_f32(&mut bytes, &[2.0, 3.0]);
    }
    let absmax_end = u64::try_from(bytes.len())?;
    append_f32(
        &mut bytes,
        &if kind == BitsAndBytes4BitType::Fp4 {
            fp4()
        } else {
            nf4()
        },
    );
    let map_end = u64::try_from(bytes.len())?;
    if nested {
        append_f32(&mut bytes, &[1.0]);
    }
    let nested_absmax_end = u64::try_from(bytes.len())?;
    if nested {
        let mut map = [0.0_f32; 256];
        map[0] = 2.0;
        map[1] = 3.0;
        append_f32(&mut bytes, &map);
    }
    let end = u64::try_from(bytes.len())?;
    let mut infos = vec![
        info("weight", path, dtype, shape, 0, weight_end),
        info(
            "absmax",
            path,
            if nested {
                "U8"
            } else {
                "F32"
            },
            &[2],
            weight_end,
            absmax_end,
        ),
        info("quant_map", path, "F32", &[16], absmax_end, map_end),
    ];
    if nested {
        infos.push(info("nested_absmax", path, "F32", &[1], map_end, nested_absmax_end));
        infos.push(info("nested_quant_map", path, "F32", &[256], nested_absmax_end, end));
    }
    Ok((bytes, infos))
}

fn info(
    name: &str,
    path: &std::path::Path,
    dtype: &str,
    shape: &[usize],
    start: u64,
    end: u64,
) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.into(),
        dtype: dtype.into(),
        shape: shape.into(),
        data_start: 0,
        data_offsets: [start, end],
    }
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

fn temp_path(kind: BitsAndBytes4BitType) -> PathBuf {
    std::env::temp_dir().join(format!(
        "libmir-cuda-bnb4-{}-{}.bin",
        kind.as_str(),
        std::process::id()
    ))
}
