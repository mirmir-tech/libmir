use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::TensorInfo;

use super::super::*;
use crate::{AffineQuantizedConfig, CudaBackend, CudaConfig, CudaTensorSet};

const INPUT: usize = 64;
const OUTPUT: usize = 2;
const EXPERTS: usize = 3;
const SELECTED: usize = 2;

#[test]
fn matches_selected_int4_and_int8_gate_up_projections() -> Result<()> {
    for bits in [4, 8] {
        check_selected_pair(bits)?;
    }
    Ok(())
}

fn check_selected_pair(bits: usize) -> Result<()> {
    let (path, infos) = fixture(bits)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    let pair = AffineQuantizedPairTensors {
        gate: bank(&tensors, &infos[0..3])?,
        up: bank(&tensors, &infos[3..6])?,
    };

    let mut host_input = backend.inner.context.allocate_pinned::<bf16>(INPUT)?;
    host_input.copy_from_slice(&[bf16::from_f32(1.0); INPUT])?;
    let mut input = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, INPUT)?;
    backend.inner.stream.copy_to_device(&mut host_input, &mut input)?;
    let mut host_selected = backend.inner.context.allocate_pinned::<u32>(SELECTED)?;
    host_selected.copy_from_slice(&[2, 0])?;
    let mut selected = backend.inner.pool.allocate::<u32>(&backend.inner.stream, SELECTED)?;
    backend.inner.stream.copy_to_device(&mut host_selected, &mut selected)?;

    let config = AffineQuantizedConfig::new(INPUT, OUTPUT, 64, bits);
    let operation = backend.prepare_selected_affine_pair_bf16_linear(config, EXPERTS, SELECTED)?;
    let output_elements = operation.output_elements()?;
    let mut gate_output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, output_elements)?;
    let mut up_output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, output_elements)?;
    operation.execute(&input, &selected, pair, &mut gate_output, &mut up_output)?;

    let mut host_gate = backend.inner.context.allocate_pinned::<bf16>(output_elements)?;
    let mut host_up = backend.inner.context.allocate_pinned::<bf16>(output_elements)?;
    backend.inner.stream.copy_to_host(&gate_output, &mut host_gate)?;
    backend.inner.stream.copy_to_host(&up_output, &mut host_up)?;
    assert_eq!(host_gate.to_vec()?, [192.0_f32, 256.0, 64.0, 128.0].map(bf16::from_f32));
    assert_eq!(host_up.to_vec()?, [384.0_f32, 448.0, 128.0, 192.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

fn fixture(bits: usize) -> Result<(PathBuf, [TensorInfo; 6])> {
    let values_per_word = 32 / bits;
    let words_per_row = INPUT / values_per_word;
    let mut bytes = Vec::new();
    append_bank(&mut bytes, bits, words_per_row, false)?;
    let gate_weight_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, &[1.0; EXPERTS * OUTPUT]);
    let gate_scale_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, &[0.0; EXPERTS * OUTPUT]);
    let gate_bias_end = u64::try_from(bytes.len())?;
    append_bank(&mut bytes, bits, words_per_row, true)?;
    let up_weight_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, &[1.0; EXPERTS * OUTPUT]);
    let up_scale_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, &[0.0; EXPERTS * OUTPUT]);
    let up_bias_end = u64::try_from(bytes.len())?;
    let path = temp_path(bits);
    fs::write(&path, bytes)?;
    let weight_shape = vec![EXPERTS, OUTPUT, words_per_row];
    let group_shape = vec![EXPERTS, OUTPUT, 1];
    let infos = [
        info("gate.weight", &path, "U32", weight_shape.clone(), 0, gate_weight_end),
        info(
            "gate.scales",
            &path,
            "BF16",
            group_shape.clone(),
            gate_weight_end,
            gate_scale_end,
        ),
        info("gate.biases", &path, "BF16", group_shape.clone(), gate_scale_end, gate_bias_end),
        info("up.weight", &path, "U32", weight_shape, gate_bias_end, up_weight_end),
        info("up.scales", &path, "BF16", group_shape.clone(), up_weight_end, up_scale_end),
        info("up.biases", &path, "BF16", group_shape, up_scale_end, up_bias_end),
    ];
    Ok((path, infos))
}

fn append_bank(bytes: &mut Vec<u8>, bits: usize, words_per_row: usize, up: bool) -> Result<()> {
    let values_per_word = 32 / bits;
    for expert in 0..EXPERTS {
        for row in 0..OUTPUT {
            let value = if up {
                2 * (expert + 1) + row
            } else {
                expert + row + 1
            };
            let value = u32::try_from(value)?;
            let word =
                (0..values_per_word).fold(0_u32, |word, lane| word | (value << (lane * bits)));
            for _ in 0..words_per_row {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn append_bf16(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values.iter().copied().map(bf16::from_f32) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn info(
    name: &str,
    path: &std::path::Path,
    dtype: &str,
    shape: Vec<usize>,
    start: u64,
    end: u64,
) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.into(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [start, end],
    }
}

fn bank<'a>(
    tensors: &'a CudaTensorSet,
    infos: &[TensorInfo],
) -> Result<AffineQuantizedTensors<'a>> {
    Ok(AffineQuantizedTensors {
        weight: required(tensors, &infos[0].name)?,
        scales: required(tensors, &infos[1].name)?,
        biases: required(tensors, &infos[2].name)?,
    })
}

fn required<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

fn temp_path(bits: usize) -> PathBuf {
    std::env::temp_dir()
        .join(format!("libmir-cuda-selected-pair-{bits}-{}.bin", std::process::id()))
}
