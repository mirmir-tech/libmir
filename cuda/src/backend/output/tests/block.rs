use mircuda::bf16;

use super::{read, read_f32, upload};
use crate::{
    CudaBackend, CudaConfig, Result,
    kernels::{BlockFp8LinearKernels, BlockFp8LinearSpec, Fp8OutputSpec, Fp8RefinementKernels},
};

const N: usize = 256;
const K: usize = 256;

#[test]
#[allow(clippy::cast_precision_loss)]
fn block_scaled_fp8_output_matches_cpu_reference() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input_values = (0..K)
        .map(|column| bf16::from_f32(((column % 17) as f32).mul_add(0.04, -0.3)))
        .collect::<Vec<_>>();
    let weight_values = (0..N * K)
        .map(|index| {
            let row = index / K;
            let column = index % K;
            let amplitude = if column < 128 {
                0.01
            } else {
                0.2
            };
            bf16::from_f32((((row + column) % 19) as f32 - 4.0) * amplitude)
        })
        .collect::<Vec<_>>();
    let input = upload(&backend, &input_values)?;
    let source = upload(&backend, &weight_values)?;
    let spec = BlockFp8LinearSpec::new(K, N)?;
    let kernels = BlockFp8LinearKernels::compile(&backend.inner.compiler, spec)?;
    let mut weight = backend.inner.pool.allocate::<u8>(&backend.inner.stream, N * K)?;
    let mut scales = backend.inner.pool.allocate::<f32>(&backend.inner.stream, N * (K / 128))?;
    kernels.quantize(&backend.inner.stream, &source, &mut weight, &mut scales)?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, N)?;
    kernels.project(&backend.inner.stream, &input, &weight, &scales, &mut output)?;
    let actual = read(&backend, &output)?;
    let expected = reference(&input_values, &weight_values);
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs() / expected.abs().max(0.01))
        .fold(0.0_f32, f32::max);
    assert!(error < 0.08, "block-scaled E4M3 maximum relative error: {error:.6}");
    let scales = read_f32(&backend, &scales)?;
    assert!(scales[0] < scales[1]);
    let refinement =
        Fp8RefinementKernels::compile(&backend.inner.compiler, Fp8OutputSpec::new(K, N)?)?;
    let workspace = Fp8RefinementKernels::workspace_elements(N)?;
    let mut first = backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?;
    let mut second = backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?;
    refinement.execute(
        &backend.inner.stream,
        &input,
        &source,
        &mut output,
        &mut first,
        &mut second,
    )?;
    let refined = read(&backend, &output)?;
    assert_eq!(maximum(&refined), maximum_f32(&expected));
    Ok(())
}

fn reference(input: &[bf16], weight: &[bf16]) -> Vec<f32> {
    weight
        .as_chunks::<K>()
        .0
        .iter()
        .map(|row| {
            input
                .iter()
                .zip(row)
                .fold(0.0_f32, |sum, (input, weight)| input.to_f32().mul_add(weight.to_f32(), sum))
        })
        .collect()
}

fn maximum(values: &[bf16]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
        .map(|(index, _)| index)
}

fn maximum_f32(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
}
