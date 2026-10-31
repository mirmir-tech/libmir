mod batch;
mod chunked;
mod oracle;
mod reference;
mod residency;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use reference::{Parameters, Reference};

use crate::{
    CudaConfig, Result,
    backend::CudaBackend,
    kernels::{Mamba2Geometry, Mamba2Parameters},
};

const SMALL: Mamba2Geometry = Mamba2Geometry {
    heads: 4,
    head_dim: 16,
    state_size: 32,
    groups: 2,
    convolution_kernel_size: 4,
    time_step_minimum: 0.0,
    time_step_maximum: f32::INFINITY,
    norm_epsilon: 1e-5,
};

/// Deterministic values in `[-scale, scale]`.
fn pattern(elements: usize, seed: u64, scale: f32) -> Vec<bf16> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..elements)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit =
                f32::from(u16::try_from(state >> 48).unwrap_or_default()) / f32::from(u16::MAX);
            bf16::from_f32(unit.mul_add(2.0, -1.0) * scale)
        })
        .collect()
}

fn parameters(geometry: Mamba2Geometry) -> Parameters {
    let inner = geometry.heads * geometry.head_dim;
    let channels = inner + 2 * geometry.groups * geometry.state_size;
    Parameters {
        convolution_weight: pattern(channels * geometry.convolution_kernel_size, 1, 0.5),
        convolution_bias: pattern(channels, 2, 0.2),
        a_log: pattern(geometry.heads, 3, 1.5),
        skip: pattern(geometry.heads, 4, 1.0),
        dt_bias: pattern(geometry.heads, 5, 1.0),
        norm_weight: pattern(inner, 6, 1.0),
    }
}

struct Device {
    convolution_weight: DeviceBuffer<bf16>,
    convolution_bias: DeviceBuffer<bf16>,
    a_log: DeviceBuffer<bf16>,
    skip: DeviceBuffer<bf16>,
    dt_bias: DeviceBuffer<bf16>,
    norm_weight: DeviceBuffer<bf16>,
}

impl Device {
    fn upload(backend: &CudaBackend, parameters: &Parameters) -> Result<Self> {
        Ok(Self {
            convolution_weight: copy(backend, &parameters.convolution_weight)?,
            convolution_bias: copy(backend, &parameters.convolution_bias)?,
            a_log: copy(backend, &parameters.a_log)?,
            skip: copy(backend, &parameters.skip)?,
            dt_bias: copy(backend, &parameters.dt_bias)?,
            norm_weight: copy(backend, &parameters.norm_weight)?,
        })
    }

    fn view(&self) -> Mamba2Parameters<'_> {
        Mamba2Parameters {
            convolution_weight: &self.convolution_weight,
            convolution_bias: Some(&self.convolution_bias),
            a_log: &self.a_log,
            skip: &self.skip,
            dt_bias: &self.dt_bias,
            norm_weight: &self.norm_weight,
        }
    }
}

fn width(geometry: Mamba2Geometry) -> usize {
    let inner = geometry.heads * geometry.head_dim;
    2 * inner + 2 * geometry.groups * geometry.state_size + geometry.heads
}

fn run(
    backend: &CudaBackend,
    state: &mut crate::backend::CudaMamba2State,
    projection: &[bf16],
    device: &Device,
) -> Result<Vec<bf16>> {
    let geometry = state.geometry();
    let tokens = projection.len() / width(geometry);
    let input = copy(backend, projection)?;
    let mut output = backend
        .inner
        .pool
        .allocate(&backend.inner.stream, tokens * geometry.heads * geometry.head_dim)?;
    let mut workspace = backend.prepare_mamba2_workspace(geometry, tokens)?;
    state.execute(tokens, &input, device.view(), &mut workspace, &mut output)?;
    read(backend, &output)
}

/// Elementwise error within 3% of the value plus 2% of the output RMS, and
/// relative L2 error below 1%: BF16 rounding of near-cancelling sums makes a
/// purely relative bound meaningless for values close to zero.
fn assert_close(actual: &[bf16], expected: &[f64]) {
    let actual = actual.iter().map(|value| f64::from(value.to_f32())).collect::<Vec<_>>();
    assert_close_values(&actual, expected);
}

fn assert_close_values(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let norm = expected.iter().map(|value| value * value).sum::<f64>();
    let rms = (norm / reference::count(expected.len().max(1))).sqrt();
    let mut worst = (0.0, 0, 0.0, 0.0);
    let mut error = 0.0;
    for (index, (actual, expected)) in actual.iter().copied().zip(expected).enumerate() {
        let ratio = (actual - expected).abs() / 0.02f64.mul_add(rms, 0.03 * expected.abs());
        if ratio > worst.0 {
            worst = (ratio, index, actual, *expected);
        }
        error += (actual - expected).powi(2);
    }
    let l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(worst.0 <= 1.0 && l2 < 0.01, "worst {worst:?}, relative L2 {l2}");
}

#[test]
fn matches_the_fp64_reference_across_chunks() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let parameters = parameters(SMALL);
    let device = Device::upload(&backend, &parameters)?;
    let mut state = backend.prepare_mamba2_state(SMALL)?;
    let mut reference = Reference::new(SMALL);
    for (chunk, tokens) in [5, 1, 3, 1].into_iter().enumerate() {
        let projection = pattern(tokens * width(SMALL), 10 + chunk as u64, 1.0);
        let actual = run(&backend, &mut state, &projection, &device)?;
        assert_close(&actual, &reference.execute(&projection, &parameters));
    }
    assert_eq!(state.offset(), 10);
    let gpu_state = read(&backend, &state.state)?;
    let gpu_state = gpu_state.into_iter().map(f64::from).collect::<Vec<_>>();
    assert_close_values(&gpu_state, &reference.state);
    Ok(())
}

#[test]
fn splitting_a_prompt_into_chunks_preserves_state_and_output() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(SMALL))?;
    let projection = pattern(8 * width(SMALL), 20, 1.0);
    let mut whole = backend.prepare_mamba2_state(SMALL)?;
    let whole_output = run(&backend, &mut whole, &projection, &device)?;
    let mut split = backend.prepare_mamba2_state(SMALL)?;
    let boundary = 3 * width(SMALL);
    let mut split_output = run(&backend, &mut split, &projection[..boundary], &device)?;
    split_output.extend(run(&backend, &mut split, &projection[boundary..], &device)?);
    assert_eq!(whole_output, split_output);
    assert_eq!(read(&backend, &whole.state)?, read(&backend, &split.state)?);
    assert_eq!(read(&backend, &whole.history)?, read(&backend, &split.history)?);
    Ok(())
}

#[test]
fn restores_state_and_history_from_a_checkpoint() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(SMALL))?;
    let mut state = backend.prepare_mamba2_state(SMALL)?;
    run(&backend, &mut state, &pattern(4 * width(SMALL), 30, 1.0), &device)?;
    let checkpoint = state.checkpoint()?;
    let next = pattern(2 * width(SMALL), 31, 1.0);
    let expected = run(&backend, &mut state, &next, &device)?;
    state.restore(&checkpoint)?;
    assert_eq!(state.offset(), 4);
    assert_eq!(run(&backend, &mut state, &next, &device)?, expected);
    state.reset()?;
    assert_eq!(state.offset(), 0);
    Ok(())
}

#[test]
fn clamps_the_time_step_to_a_finite_limit() -> Result<()> {
    let geometry = Mamba2Geometry {
        time_step_minimum: 0.05,
        time_step_maximum: 0.2,
        ..SMALL
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let parameters = parameters(geometry);
    let device = Device::upload(&backend, &parameters)?;
    let mut state = backend.prepare_mamba2_state(geometry)?;
    let mut reference = Reference::new(geometry);
    let projection = pattern(6 * width(geometry), 40, 3.0);
    let actual = run(&backend, &mut state, &projection, &device)?;
    assert_close(&actual, &reference.execute(&projection, &parameters));
    Ok(())
}

#[test]
fn matches_the_reference_at_nemotron_super_geometry() -> Result<()> {
    let geometry = Mamba2Geometry {
        heads: 128,
        head_dim: 64,
        state_size: 128,
        groups: 8,
        ..SMALL
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let parameters = parameters(geometry);
    let device = Device::upload(&backend, &parameters)?;
    let mut state = backend.prepare_mamba2_state(geometry)?;
    let mut reference = Reference::new(geometry);
    for (chunk, tokens) in [3, 1].into_iter().enumerate() {
        let projection = pattern(tokens * width(geometry), 50 + chunk as u64, 1.0);
        let actual = run(&backend, &mut state, &projection, &device)?;
        assert_close(&actual, &reference.execute(&projection, &parameters));
    }
    Ok(())
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
