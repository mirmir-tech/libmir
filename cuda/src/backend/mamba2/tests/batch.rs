//! Packed decode of several sequences equals decoding each one alone.

use mircuda::bf16;

use super::{Device, SMALL, copy, parameters, pattern, read, run, width};
use crate::{
    CudaConfig, Result,
    backend::CudaBackend,
    kernels::{Mamba2DecodeBatch, Mamba2Geometry, Mamba2Kernels},
};

const PROMPTS: [usize; 3] = [3, 7, 70];
const STEPS: usize = 2;

#[test]
fn packed_decode_matches_independent_sequences() -> Result<()> {
    batch_matches(SMALL)
}

#[test]
fn packed_decode_matches_at_nemotron_super_geometry() -> Result<()> {
    batch_matches(Mamba2Geometry {
        heads: 128,
        head_dim: 64,
        state_size: 128,
        groups: 8,
        ..SMALL
    })
}

fn batch_matches(geometry: Mamba2Geometry) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(geometry))?;
    let rows = PROMPTS.len();
    let width = width(geometry);
    let (state_elements, history_elements) =
        (geometry.state_elements()?, geometry.history_elements()?);
    let kernels = Mamba2Kernels::compile(&backend.inner.compiler, geometry)?;
    let mut sequences = Vec::with_capacity(rows);
    let pool = &backend.inner.pool;
    let stream = &backend.inner.stream;
    let mut states = pool.allocate(stream, rows * state_elements)?;
    let mut histories = pool.allocate(stream, rows * history_elements)?;
    for (row, tokens) in PROMPTS.into_iter().enumerate() {
        let mut state = backend.prepare_mamba2_state(geometry)?;
        run(
            &backend,
            &mut state,
            &pattern(tokens * width, 10 + u64::try_from(row)?, 1.0),
            &device,
        )?;
        stream.copy_device_range(
            &state.state,
            0..state_elements,
            &mut states,
            row * state_elements,
        )?;
        stream.copy_device_range(
            &state.history,
            0..history_elements,
            &mut histories,
            row * history_elements,
        )?;
        sequences.push(state);
    }
    let inner = geometry.inner()?;
    let mut convolved = pool.allocate(stream, rows * geometry.convolution_channels()?)?;
    let mut scanned = pool.allocate(stream, rows * inner)?;
    let mut output = pool.allocate(stream, rows * inner)?;
    for step in 0..STEPS {
        let projection = pattern(rows * width, 100 + u64::try_from(step)?, 1.0);
        let mut expected = Vec::with_capacity(rows * inner);
        for (row, state) in sequences.iter_mut().enumerate() {
            expected.extend(run(
                &backend,
                state,
                &projection[row * width..(row + 1) * width],
                &device,
            )?);
        }
        let input = copy(&backend, &projection)?;
        kernels.execute_decode_batch(
            stream,
            &mut Mamba2DecodeBatch {
                rows,
                projection: &input,
                parameters: device.view(),
                states: &mut states,
                histories: &mut histories,
                convolved: &mut convolved,
                scanned: &mut scanned,
                output: &mut output,
            },
        )?;
        assert_eq!(read(&backend, &output)?, expected, "decode step {step}");
    }
    let (packed_states, packed_histories) =
        (read(&backend, &states)?, read::<bf16>(&backend, &histories)?);
    for (row, state) in sequences.iter().enumerate() {
        let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&packed_states[row * state_elements..(row + 1) * state_elements]),
            bits(&read(&backend, &state.state)?),
            "state of row {row}"
        );
        assert_eq!(
            packed_histories[row * history_elements..(row + 1) * history_elements],
            read(&backend, &state.history)?[..],
            "history of row {row}"
        );
    }
    Ok(())
}
