//! GGML output scales against the same products without them.

use mircuda::{DeviceBuffer, DeviceElement, bf16, f16};
use models::gguf::GgmlType;

use super::GgmlCheckpointWeight;
use crate::{CudaBackend, CudaConfig, DenseRole, Result, kernels::GgmlKernelType};

const COLUMNS: usize = 256;
const ROWS: usize = 48;

pub(in crate::backend) fn copy<T: DeviceElement>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.inner.stream.synchronize()?;
    Ok(device)
}

pub(in crate::backend) fn read<T: DeviceElement>(
    backend: &CudaBackend,
    values: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    backend.inner.stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}

/// `Q8_0` blocks of `matrices` `[rows, COLUMNS]` matrices; every block's
/// half scale of matrix `m` is multiplied by `factors[m]`, which powers of two
/// keep exact.
pub(in crate::backend) fn q8_0(
    matrices: usize,
    rows: usize,
    columns: usize,
    factors: &[f32],
) -> Vec<u8> {
    let blocks = rows * columns / 32;
    let mut bytes = Vec::with_capacity(matrices * blocks * 34);
    for matrix in 0..matrices {
        for block in 0..blocks {
            let seed = matrix * 7919 + block * 104_729;
            #[allow(clippy::cast_precision_loss)]
            let d = 0.001_f32.mul_add((seed % 5) as f32, 0.004);
            let d = f16::from_f32(d * factors.get(matrix).copied().unwrap_or(1.0));
            bytes.extend(d.to_le_bytes());
            for value in 0..32 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let quant = ((seed + value * 37) % 255) as i32 - 127;
                bytes.push(i8::try_from(quant).unwrap_or(0).to_le_bytes()[0]);
            }
        }
    }
    bytes
}

pub(in crate::backend) fn input(elements: usize) -> Vec<bf16> {
    #[allow(clippy::cast_precision_loss)]
    (0..elements)
        .map(|index| bf16::from_f32((index % 23) as f32 / 23.0 - 0.5))
        .collect()
}

pub(in crate::backend) fn assert_close(expected: &[bf16], actual: &[bf16], label: &str) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let (expected, actual) = (expected.to_f32(), actual.to_f32());
        assert!(
            (expected - actual).abs() <= expected.abs().mul_add(1.0e-2, 1.0e-3),
            "{label} [{index}]: {actual} vs {expected}"
        );
    }
}

/// A one-value scale multiplies every product of the GEMV, grouped int8 and
/// dequantized prefill paths.
#[test]
fn output_scale_multiplies_a_projections_products() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let blocks = copy(&backend, &q8_0(1, ROWS, COLUMNS, &[]))?;
    let geometry = (1, ROWS, COLUMNS);
    let unscaled = GgmlCheckpointWeight::from_blocks(
        blocks.clone(),
        GgmlKernelType::new(GgmlType::Q8_0)?,
        geometry,
        None,
    );
    let scale = Some(copy(&backend, &[-0.75_f32])?);
    let scaled = GgmlCheckpointWeight::from_blocks(
        blocks,
        GgmlKernelType::new(GgmlType::Q8_0)?,
        geometry,
        scale,
    );
    for tokens in [1, 100, 300] {
        let x = copy(&backend, &input(tokens * COLUMNS))?;
        let plan = unscaled.prepare(&backend, tokens, DenseRole::DenseDown)?;
        let allocate = || backend.inner.pool.allocate::<bf16>(&backend.inner.stream, tokens * ROWS);
        let (mut first, mut second) = (allocate()?, allocate()?);
        plan.execute(&x, &unscaled, &mut first)?;
        plan.execute(&x, &scaled, &mut second)?;
        let expected = read(&backend, &first)?
            .into_iter()
            .map(|value| bf16::from_f32(value.to_f32() * -0.75))
            .collect::<Vec<_>>();
        assert_close(&expected, &read(&backend, &second)?, &format!("{tokens} tokens"));
        // A folded factor (Granite's residual multiplier) composes with it.
        let mut third = allocate()?;
        plan.execute(&x, &scaled.scaled(&backend, 2.0)?, &mut third)?;
        let doubled = expected
            .iter()
            .map(|value| bf16::from_f32(value.to_f32() * 2.0))
            .collect::<Vec<_>>();
        assert_close(&doubled, &read(&backend, &third)?, &format!("{tokens} tokens, folded"));
    }
    Ok(())
}
