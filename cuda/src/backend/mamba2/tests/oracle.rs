//! V2 gate: one real Mamba-2 layer against the vLLM reference operators.
//!
//! The oracle directory holds raw little-endian arrays written by
//! `benchmarks/tools/nemotron/mixer_oracle.py`: the fused input projection
//! (`projection.bf16`), the layer parameters, the gated, normalized mixer
//! output (`output.bf16`), and the final SSM state (`state.f32`).

use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::bf16;

use super::{copy, read};
use crate::{
    CudaConfig, Mamba2ScanPath, Result,
    backend::CudaBackend,
    kernels::{Mamba2Geometry, Mamba2Parameters},
};

const SUPER: Mamba2Geometry = Mamba2Geometry {
    heads: 128,
    head_dim: 64,
    state_size: 128,
    groups: 8,
    convolution_kernel_size: 4,
    time_step_minimum: 0.0,
    time_step_maximum: f32::INFINITY,
    norm_epsilon: 1e-5,
};

fn bf16s(directory: &Path, name: &str) -> Result<Vec<bf16>> {
    let bytes = fs::read(directory.join(name))?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|pair| bf16::from_le_bytes(*pair)).collect())
}

fn f32s(directory: &Path, name: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(directory.join(name))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

#[test]
#[ignore = "requires NEMOTRON_MIXER_ORACLE written by the vLLM mixer oracle"]
fn matches_the_vllm_reference_on_a_real_layer() -> Result<()> {
    let Some(directory) = std::env::var_os("NEMOTRON_MIXER_ORACLE").map(PathBuf::from) else {
        return Ok(());
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let projection = bf16s(&directory, "projection.bf16")?;
    let expected = bf16s(&directory, "output.bf16")?;
    let expected_state = f32s(&directory, "state.f32")?;
    let parameters = [
        "convolution_weight.bf16",
        "convolution_bias.bf16",
        "a_log.bf16",
        "skip.bf16",
        "dt_bias.bf16",
        "norm_weight.bf16",
    ]
    .into_iter()
    .map(|name| copy(&backend, &bf16s(&directory, name)?))
    .collect::<Result<Vec<_>>>()?;
    let view = Mamba2Parameters {
        convolution_weight: &parameters[0],
        convolution_bias: Some(&parameters[1]),
        a_log: &parameters[2],
        skip: &parameters[3],
        dt_bias: &parameters[4],
        norm_weight: &parameters[5],
    };
    let tokens = projection.len() / SUPER.projection_width()?;
    let input = copy(&backend, &projection)?;
    let expected = expected.iter().map(|value| f64::from(value.to_f32())).collect::<Vec<_>>();
    let expected_state = expected_state.into_iter().map(f64::from).collect::<Vec<_>>();
    for path in [Mamba2ScanPath::Chunked, Mamba2ScanPath::Sequential] {
        let mut state = backend.prepare_mamba2_state(SUPER)?;
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, expected.len())?;
        let mut workspace = backend.prepare_mamba2_workspace(SUPER, tokens)?;
        state.execute_with(path, tokens, &input, view, &mut workspace, &mut output)?;
        let actual = read(&backend, &output)?
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect::<Vec<_>>();
        assert_parity(path, "output", &actual, &expected, SUPER.inner()?);
        // A fresh state over the same input must give bit-identical output.
        let mut again = backend.prepare_mamba2_state(SUPER)?;
        let mut repeat = backend.inner.pool.allocate(&backend.inner.stream, expected.len())?;
        again.execute_with(path, tokens, &input, view, &mut workspace, &mut repeat)?;
        let (first, second) = (read(&backend, &output)?, read(&backend, &repeat)?);
        assert!(first.iter().zip(&second).all(|(a, b)| a.to_bits() == b.to_bits()), "{path:?}");
        let actual_state =
            read(&backend, &state.state)?.into_iter().map(f64::from).collect::<Vec<_>>();
        assert_parity(
            path,
            "state",
            &actual_state,
            &expected_state,
            SUPER.head_dim * SUPER.state_size,
        );
    }
    Ok(())
}

/// Relative L2 error below 1% and cosine similarity of every row (a token's
/// output or a head's state) of at least 0.999.
fn assert_parity(path: Mamba2ScanPath, name: &str, actual: &[f64], expected: &[f64], row: usize) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm) = (0.0, 0.0);
    let mut worst_cosine = 1.0_f64;
    for (actual, expected) in actual.chunks(row).zip(expected.chunks(row)) {
        let (mut dot, mut left, mut right) = (0.0, 0.0, 0.0);
        for (actual, expected) in actual.iter().zip(expected) {
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
            dot += actual * expected;
            left += actual * actual;
            right += expected * expected;
        }
        if left > 0.0 && right > 0.0 {
            worst_cosine = worst_cosine.min(dot / (left.sqrt() * right.sqrt()));
        }
    }
    let l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        l2 < 0.01 && worst_cosine >= 0.999,
        "{path:?} {name}: relative L2 {l2}, worst row cosine {worst_cosine}"
    );
}
