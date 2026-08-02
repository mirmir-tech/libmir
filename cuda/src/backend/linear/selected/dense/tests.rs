use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::{DeviceBuffer, bf16};
use models::weights::{
    BindingTransform, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    RoutedExpertBindings, TensorBinding, TensorInfo, TensorStorage,
};

use super::*;
use crate::{CudaConfig, kernels::DenseGatedActivation};

const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 8;
const EXPERTS: usize = 2;
const SELECTED: usize = 2;

#[test]
#[allow(clippy::suboptimal_flops)]
fn executes_transposed_interleaved_dense_experts_with_biases() -> Result<()> {
    let (path, infos) = fixture()?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    let gate_up = binding(
        &infos[0].name,
        &infos[1].name,
        ExpertProjectionRole::GateUp,
        infos[0].shape.clone(),
        vec![
            BindingTransform::StackedExperts { count: EXPERTS },
            BindingTransform::FusedGateUp { interleaved: true },
            BindingTransform::Transpose,
        ],
    );
    let down = binding(
        &infos[2].name,
        &infos[3].name,
        ExpertProjectionRole::Down,
        infos[2].shape.clone(),
        vec![BindingTransform::StackedExperts { count: EXPERTS }, BindingTransform::Transpose],
    );
    let weights = DenseExpertWeights::load(
        &backend,
        &tensors,
        RoutedExpertBindings::InterleavedGateUp { gate_up: &gate_up, down: &down },
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
    )?;
    let expert0 = rounded(silu(rounded(2.5)) * rounded(2.0));
    let expert1 = rounded(silu(rounded(3.5)) * rounded(2.0));
    let expected =
        rounded(rounded(8.0 * expert0 + 1.0) * 0.25 + rounded(16.0 * expert1 - 1.0) * 0.75);
    for tokens in 1..=2 {
        let mut operation = SelectedDenseMoeBf16::new(
            &backend,
            tokens,
            SELECTED,
            &weights,
            DenseGatedActivation::SILU,
        )?;
        let input = copy(&backend, &vec![bf16::from_f32(1.0); tokens * HIDDEN])?;
        let selected = copy(&backend, &[0_u32, 1].repeat(tokens))?;
        let routing = copy(&backend, &[bf16::from_f32(0.25), bf16::from_f32(0.75)].repeat(tokens))?;
        let mut intermediate = allocate(&backend, tokens * SELECTED * INTERMEDIATE)?;
        let mut output = allocate(&backend, tokens * HIDDEN)?;
        operation.execute(&input, &selected, &routing, &weights, &mut intermediate, &mut output)?;
        for value in read(&backend, &output)? {
            assert!((value.to_f32() - expected).abs() < 0.04);
        }
    }
    fs::remove_file(path)?;
    Ok(())
}

fn fixture() -> Result<(PathBuf, [TensorInfo; 4])> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-selected-dense-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    for expert in 0..EXPERTS {
        let gate = if expert == 0 {
            0.25
        } else {
            0.5
        };
        for _column in 0..HIDDEN {
            for row in 0..INTERMEDIATE {
                let _ = row;
                push(&mut bytes, gate);
                push(&mut bytes, 0.25);
            }
        }
    }
    let gate_end = u64::try_from(bytes.len())?;
    for expert in 0..EXPERTS {
        for _row in 0..INTERMEDIATE {
            push(
                &mut bytes,
                if expert == 0 {
                    0.5
                } else {
                    -0.5
                },
            );
            push(&mut bytes, 0.0);
        }
    }
    let gate_bias_end = u64::try_from(bytes.len())?;
    for expert in 0..EXPERTS {
        for _input in 0..INTERMEDIATE {
            for _output in 0..HIDDEN {
                push(
                    &mut bytes,
                    if expert == 0 {
                        1.0
                    } else {
                        2.0
                    },
                );
            }
        }
    }
    let down_end = u64::try_from(bytes.len())?;
    for expert in 0..EXPERTS {
        for _output in 0..HIDDEN {
            push(
                &mut bytes,
                if expert == 0 {
                    1.0
                } else {
                    -1.0
                },
            );
        }
    }
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    Ok((
        path.clone(),
        [
            info("gate_up", &path, vec![EXPERTS, HIDDEN, 2 * INTERMEDIATE], 0, gate_end),
            info("gate_up_bias", &path, vec![EXPERTS, 2 * INTERMEDIATE], gate_end, gate_bias_end),
            info("down", &path, vec![EXPERTS, INTERMEDIATE, HIDDEN], gate_bias_end, down_end),
            info("down_bias", &path, vec![EXPERTS, HIDDEN], down_end, end),
        ],
    ))
}

fn binding(
    source: &str,
    bias: &str,
    projection: ExpertProjectionRole,
    shape: Vec<usize>,
    transforms: Vec<BindingTransform>,
) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
        },
        source: source.into(),
        shape,
        logical_shape: None,
        transforms,
        storage: TensorStorage::Dense {
            dtype: "BF16".into(),
            bias: Some(bias.into()),
        },
    }
}

fn info(name: &str, path: &Path, shape: Vec<usize>, start: u64, end: u64) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.to_path_buf(),
        dtype: "BF16".into(),
        shape,
        data_start: 0,
        data_offsets: [start, end],
    }
}

fn push(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
}

fn copy<T: mircuda::DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn allocate(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate_zeroed(&backend.inner.stream, elements)?)
}

fn read(backend: &CudaBackend, device: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.inner.context.allocate_pinned(device.len())?;
    backend.inner.stream.copy_to_host(device, &mut host)?;
    Ok(host.to_vec()?)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}
