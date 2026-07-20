use std::{fs, path::PathBuf};

use mircuda::{DeviceBuffer, bf16};
use models::weights::TensorInfo;

use super::super::*;
use crate::{AffineQuantizedConfig, CudaBackend, CudaConfig, CudaTensor, CudaTensorSet};

const WIDTH: usize = 64;
const HIDDEN: usize = 2;
const EXPERTS: usize = 3;
const SELECTED: usize = 2;

#[test]
fn matches_selected_gated_and_reduced_int4_and_int8() -> Result<()> {
    for bits in [4, 8] {
        check_moe(bits)?;
    }
    Ok(())
}

fn check_moe(bits: usize) -> Result<()> {
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
    let down = bank(&tensors, &infos[6..9])?;
    let input = copy_device(&backend, &[bf16::from_f32(1.0); WIDTH])?;
    let selected = copy_device(&backend, &[2_u32, 0])?;
    let routing = copy_device(&backend, &[bf16::from_f32(0.25), bf16::from_f32(0.75)])?;
    for activation in [GatedActivation::GeluTanh, GatedActivation::Silu] {
        let gated_config = AffineQuantizedConfig::new(WIDTH, WIDTH, WIDTH, bits);
        let down_config = AffineQuantizedConfig::new(WIDTH, HIDDEN, WIDTH, bits);
        let gated = backend.prepare_selected_affine_gated_bf16_linear(
            gated_config, EXPERTS, SELECTED, activation,
        )?;
        let reduce =
            backend.prepare_selected_affine_reduce_bf16_linear(down_config, EXPERTS, SELECTED)?;
        let mut intermediate = backend
            .inner
            .pool
            .allocate_zeroed::<bf16>(&backend.inner.stream, gated.output_elements()?)?;
        let mut output = backend
            .inner
            .pool
            .allocate_zeroed::<bf16>(&backend.inner.stream, reduce.output_elements()?)?;
        gated.execute(&input, &selected, pair, &mut intermediate)?;
        reduce.execute(&intermediate, &selected, &routing, down, &mut output)?;
        let mut host = backend.inner.context.allocate_pinned::<bf16>(HIDDEN)?;
        backend.inner.stream.copy_to_host(&output, &mut host)?;
        let actual = host.to_vec()?;
        let gate_three = rounded_activation(3.0, activation);
        let gate_one = rounded_activation(1.0, activation);
        let expected = [
            rounded(rounded(gate_three * 3.0) * 0.25 + rounded(gate_one) * 0.75),
            rounded(rounded(gate_three * 4.0) * 0.25 + rounded(gate_one * 2.0) * 0.75),
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.to_f32() - expected).abs() < 0.04);
        }
    }
    fs::remove_file(path)?;
    Ok(())
}

fn rounded_activation(value: f32, activation: GatedActivation) -> f32 {
    let activated = match activation {
        GatedActivation::GeluTanh => {
            let inner = 0.797_884_6 * 0.044_715_f32.mul_add(value.powi(3), value);
            0.5 * value * (1.0 + inner.tanh())
        },
        GatedActivation::Silu => value / (1.0 + (-value).exp()),
    };
    rounded(activated)
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn fixture(bits: usize) -> Result<(PathBuf, Vec<TensorInfo>)> {
    let path = temp_path(bits);
    let mut bytes = Vec::new();
    let mut infos = Vec::new();
    append_bank(&path, &mut bytes, &mut infos, "gate", bits, WIDTH, |expert, _| expert + 1)?;
    append_bank(&path, &mut bytes, &mut infos, "up", bits, WIDTH, |_, _| 1)?;
    append_bank(&path, &mut bytes, &mut infos, "down", bits, HIDDEN, |expert, row| {
        expert + row + 1
    })?;
    fs::write(&path, bytes)?;
    Ok((path, infos))
}

fn append_bank(
    path: &std::path::Path,
    bytes: &mut Vec<u8>,
    infos: &mut Vec<TensorInfo>,
    name: &str,
    bits: usize,
    output: usize,
    value: impl Fn(usize, usize) -> usize,
) -> Result<()> {
    let values_per_word = 32 / bits;
    let words_per_row = WIDTH / values_per_word;
    let start = u64::try_from(bytes.len())?;
    for expert in 0..EXPERTS {
        for row in 0..output {
            let quantized = u32::try_from(value(expert, row))?;
            let word = (0..values_per_word)
                .fold(0_u32, |packed, lane| packed | (quantized << (lane * bits)));
            for _ in 0..words_per_row {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    let weight_end = u64::try_from(bytes.len())?;
    append_bf16(bytes, &vec![0.015_625; EXPERTS * output]);
    let scale_end = u64::try_from(bytes.len())?;
    append_bf16(bytes, &vec![0.0; EXPERTS * output]);
    let end = u64::try_from(bytes.len())?;
    let weight_shape = vec![EXPERTS, output, words_per_row];
    let group_shape = vec![EXPERTS, output, 1];
    infos.extend([
        info(&format!("{name}.weight"), path, "U32", weight_shape, start, weight_end),
        info(
            &format!("{name}.scales"),
            path,
            "BF16",
            group_shape.clone(),
            weight_end,
            scale_end,
        ),
        info(&format!("{name}.biases"), path, "BF16", group_shape, scale_end, end),
    ]);
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

fn copy_device<T: mircuda::DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn temp_path(bits: usize) -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-selected-moe-{bits}-{}.bin", std::process::id()))
}
