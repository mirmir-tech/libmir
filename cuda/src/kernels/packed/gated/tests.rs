use mircuda::{DeviceBuffer, bf16};

use super::{GatedActivation, PackedGatedBf16};
use crate::{CudaBackend, CudaConfig, Result};

#[test]
fn all_rows_are_written_for_packed_and_separate_inputs() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    for columns in [128, 257] {
        let rows = 3;
        let gate = (0..rows * columns)
            .map(|i| {
                bf16::from_f32(f32::from(u16::try_from(i % 23).unwrap_or_default()) / 8.0 - 1.0)
            })
            .collect::<Vec<_>>();
        let up = vec![bf16::from_f32(0.5); rows * columns];
        let packed = gate
            .chunks(columns)
            .zip(up.chunks(columns))
            .flat_map(|(gate, up)| gate.iter().chain(up).copied())
            .collect::<Vec<_>>();
        let gate_device = upload(&backend, &gate)?;
        let up_device = upload(&backend, &up)?;
        let packed_device = upload(&backend, &packed)?;
        let kernel = PackedGatedBf16::compile(backend.compiler(), rows, columns)?;
        for separate in [false, true] {
            let mut output = upload(&backend, &vec![bf16::NAN; rows * columns])?;
            if separate {
                kernel.execute_separate(
                    backend.stream(),
                    &gate_device,
                    &up_device,
                    &mut output,
                    GatedActivation::Silu,
                )?;
            } else {
                kernel.execute(
                    backend.stream(),
                    &packed_device,
                    &mut output,
                    GatedActivation::Silu,
                )?;
            }
            let mut host = backend.context().allocate_pinned(output.len())?;
            backend.stream().copy_to_host(&output, &mut host)?;
            for (actual, gate) in host.to_vec()?.iter().zip(&gate) {
                let value = gate.to_f32();
                let rounded = bf16::from_f32(value / (1.0 + (-value).exp())).to_f32();
                let expected = bf16::from_f32(rounded * 0.5);
                assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.003_906_25);
            }
        }
    }
    Ok(())
}

fn upload(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.context().allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
