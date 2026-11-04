use super::{
    Device, SMALL, assert_close, assert_close_values, copy, parameters, pattern, read, width,
};
use crate::{
    CudaConfig, Mamba2ScanPath, Result,
    backend::{CudaBackend, CudaMamba2State},
    kernels::Mamba2Geometry,
};

const SUPER: Mamba2Geometry = Mamba2Geometry {
    heads: 128,
    head_dim: 64,
    state_size: 128,
    groups: 8,
    ..SMALL
};

fn run_with(
    backend: &CudaBackend,
    state: &mut CudaMamba2State,
    path: Mamba2ScanPath,
    projection: &[mircuda::bf16],
    device: &Device,
) -> Result<Vec<mircuda::bf16>> {
    let geometry = state.geometry();
    let tokens = projection.len() / width(geometry);
    let input = copy(backend, projection)?;
    let mut output = backend
        .inner
        .pool
        .allocate(&backend.inner.stream, tokens * geometry.heads * geometry.head_dim)?;
    let mut workspace = backend.prepare_mamba2_workspace(geometry, tokens)?;
    state.execute_with(path, tokens, &input, device.view(), &mut workspace, &mut output)?;
    read(backend, &output)
}

fn compare(geometry: Mamba2Geometry, chunks: &[(Mamba2ScanPath, usize)]) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let parameters = parameters(geometry);
    let device = Device::upload(&backend, &parameters)?;
    let mut oracle = backend.prepare_mamba2_state(geometry)?;
    let mut candidate = backend.prepare_mamba2_state(geometry)?;
    for (index, (path, tokens)) in chunks.iter().copied().enumerate() {
        let projection = pattern(tokens * width(geometry), 60 + index as u64, 1.0);
        let expected =
            run_with(&backend, &mut oracle, Mamba2ScanPath::Sequential, &projection, &device)?;
        let actual = run_with(&backend, &mut candidate, path, &projection, &device)?;
        let expected = expected.iter().map(|value| f64::from(value.to_f32())).collect::<Vec<_>>();
        assert_close(&actual, &expected);
    }
    let widen = |values: Vec<f32>| values.into_iter().map(f64::from).collect::<Vec<_>>();
    assert_close_values(
        &widen(read(&backend, &candidate.state)?),
        &widen(read(&backend, &oracle.state)?),
    );
    assert_eq!(read(&backend, &oracle.history)?, read(&backend, &candidate.history)?);
    Ok(())
}

#[test]
fn chunked_prefill_matches_the_sequential_scan_with_a_partial_chunk() -> Result<()> {
    compare(SMALL, &[(Mamba2ScanPath::Chunked, 150)])
}

#[test]
fn chunked_prefill_continues_a_nonzero_state() -> Result<()> {
    compare(
        SMALL,
        &[
            (Mamba2ScanPath::Sequential, 5),
            (Mamba2ScanPath::Chunked, 130),
            (Mamba2ScanPath::Sequential, 1),
            (Mamba2ScanPath::Chunked, 64),
        ],
    )
}

#[test]
fn chunked_prefill_handles_a_single_short_chunk() -> Result<()> {
    compare(SMALL, &[(Mamba2ScanPath::Chunked, 3), (Mamba2ScanPath::Chunked, 1)])
}

#[test]
fn chunked_prefill_matches_the_sequential_scan_at_nemotron_super_geometry() -> Result<()> {
    compare(SUPER, &[(Mamba2ScanPath::Sequential, 2), (Mamba2ScanPath::Chunked, 200)])
}
