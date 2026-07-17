use mircuda::{BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec, bf16};

use crate::{
    CudaBackend, CudaConfig, Result,
    kernels::{Fp8OutputKernels, Fp8OutputSpec},
};

mod block;

#[test]
#[allow(clippy::cast_precision_loss)]
fn blockwise_fp8_output_matches_cpu_reference() -> Result<()> {
    const N: usize = 256;
    const K: usize = 256;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input_values = (0..K)
        .map(|column| {
            let amplitude = if column < 128 {
                0.03
            } else {
                0.2
            };
            bf16::from_f32(((column % 13) as f32 - 5.0) * amplitude)
        })
        .collect::<Vec<_>>();
    let weight_values = (0..N * K)
        .map(|index| {
            let row = index / K;
            let column = index % K;
            bf16::from_f32(
                ((row % 7) as f32).mul_add(0.02, ((column % 11) as f32).mul_add(0.01, -0.1)),
            )
        })
        .collect::<Vec<_>>();
    let input = upload(&backend, &input_values)?;
    let weight = upload(&backend, &weight_values)?;
    let spec = Fp8OutputSpec::new(K, N)?;
    let kernels = Fp8OutputKernels::compile(&backend.inner.compiler, spec)?;
    let mut fp8_input = backend.inner.pool.allocate::<u8>(&backend.inner.stream, K)?;
    let mut input_scales = backend.inner.pool.allocate::<f32>(&backend.inner.stream, K / 128)?;
    let mut fp8_weight = backend.inner.pool.allocate::<u8>(&backend.inner.stream, N * K)?;
    let mut weight_scales = backend
        .inner
        .pool
        .allocate::<f32>(&backend.inner.stream, (N / 128) * (K / 128))?;
    let mut row_scales = backend.inner.pool.allocate::<f32>(&backend.inner.stream, N)?;
    kernels.quantize_input(&backend.inner.stream, &input, &mut fp8_input, &mut input_scales)?;
    kernels.quantize_weight(
        &backend.inner.stream,
        &weight,
        &mut fp8_weight,
        &mut weight_scales,
        &mut row_scales,
    )?;
    let mut operation = BlockwiseFp8VectorPlan::new(
        &backend.inner.context,
        &backend.inner.stream,
        BlockwiseFp8VectorSpec::new(N, K)?,
    )?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, N)?;
    operation.execute(
        &backend.inner.stream,
        &fp8_input,
        &input_scales,
        &fp8_weight,
        &weight_scales,
        &mut output,
    )?;
    kernels.rescale_output(&backend.inner.stream, &mut output, &row_scales)?;
    let actual = read(&backend, &output)?;
    let input_scale = read_f32(&backend, &input_scales)?;
    let weight_scale = read_f32(&backend, &weight_scales)?;
    let expected = (0..N)
        .map(|row| {
            input_values
                .iter()
                .zip(&weight_values[row * K..(row + 1) * K])
                .fold(0.0_f32, |sum, (input, weight)| input.to_f32().mul_add(weight.to_f32(), sum))
        })
        .collect::<Vec<_>>();
    let relative = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs() / expected.abs().max(0.01))
        .fold(0.0_f32, f32::max);
    let actual_preview = actual.iter().take(8).map(|value| value.to_f32()).collect::<Vec<_>>();
    assert!(
        relative < 0.12,
        "blockwise FP8 maximum relative error: {relative:.6}; input scale: {input_scale:?}; weight scale: {weight_scale:?}; actual: {actual_preview:?}; expected: {:?}",
        &expected[..8]
    );
    assert_vectorized(&backend, &kernels, &input, &fp8_weight, &row_scales, &expected)?;
    assert_residual(&mut ResidualCase {
        backend: &backend,
        kernels: &kernels,
        input: &input,
        weight: &weight,
        fp8_weight: &mut fp8_weight,
        weight_scales: &mut weight_scales,
        row_scales: &mut row_scales,
        expected: &expected,
    })?;
    Ok(())
}

fn assert_vectorized(
    backend: &CudaBackend,
    kernels: &Fp8OutputKernels,
    input: &mircuda::DeviceBuffer<bf16>,
    weight: &mircuda::DeviceBuffer<u8>,
    scales: &mircuda::DeviceBuffer<f32>,
    expected: &[f32],
) -> Result<()> {
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, expected.len())?;
    kernels.project_vectorized(&backend.inner.stream, input, weight, scales, &mut output)?;
    let output = read(backend, &output)?;
    let relative = output
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs() / expected.abs().max(0.01))
        .fold(0.0_f32, f32::max);
    assert!(relative < 0.12, "vectorized FP8 maximum relative error: {relative:.6}");
    Ok(())
}

struct ResidualCase<'a> {
    backend: &'a CudaBackend,
    kernels: &'a Fp8OutputKernels,
    input: &'a mircuda::DeviceBuffer<bf16>,
    weight: &'a mircuda::DeviceBuffer<bf16>,
    fp8_weight: &'a mut mircuda::DeviceBuffer<u8>,
    weight_scales: &'a mut mircuda::DeviceBuffer<f32>,
    row_scales: &'a mut mircuda::DeviceBuffer<f32>,
    expected: &'a [f32],
}

fn assert_residual(case: &mut ResidualCase<'_>) -> Result<()> {
    let rows = case.expected.len();
    let columns = case.input.len();
    let mut residual = case
        .backend
        .inner
        .pool
        .allocate::<u8>(&case.backend.inner.stream, rows * columns / 2)?;
    let mut residual_scales = case
        .backend
        .inner
        .pool
        .allocate::<f32>(&case.backend.inner.stream, rows * (columns / 128))?;
    {
        let mut buffers = crate::kernels::Fp8ResidualWeightBuffers::new(
            &mut *case.fp8_weight,
            &mut *case.weight_scales,
            &mut *case.row_scales,
            &mut residual,
            &mut residual_scales,
        );
        case.kernels.quantize_weight_residual(
            &case.backend.inner.stream,
            case.weight,
            &mut buffers,
        )?;
    }
    let mut corrected =
        case.backend.inner.pool.allocate::<bf16>(&case.backend.inner.stream, rows)?;
    case.kernels.project_residual(
        &case.backend.inner.stream,
        case.input,
        case.fp8_weight,
        case.row_scales,
        &residual,
        &residual_scales,
        &mut corrected,
    )?;
    let corrected = read(case.backend, &corrected)?;
    let relative = corrected
        .iter()
        .zip(case.expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs() / expected.abs().max(0.01))
        .fold(0.0_f32, f32::max);
    assert!(relative < 0.03, "FP8 plus INT4 residual maximum relative error: {relative:.6}");
    Ok(())
}

fn upload(backend: &CudaBackend, values: &[bf16]) -> Result<mircuda::DeviceBuffer<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, source: &mircuda::DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn read_f32(backend: &CudaBackend, source: &mircuda::DeviceBuffer<f32>) -> Result<Vec<f32>> {
    let mut host = backend.inner.context.allocate_pinned::<f32>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
