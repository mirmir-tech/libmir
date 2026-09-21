use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::{super::tuning::marlin_candidates, *};
use crate::{
    CudaConfig, CudaTensor, NvFp4ScaleMode, NvFp4Tensors,
    kernels::{NvFp4Spec, NvFp4WeightOnly, NvFp4WeightOnlyLaunch},
};

const INPUT: usize = 512;
const OUTPUT: usize = 2_048;

#[test]
fn row_groups_match_compressed_reference() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = NvFp4Config {
        input_features: INPUT,
        output_features: OUTPUT,
    };
    let weight = weight(&backend, config)?;
    // One launch of eight rows, one sixteen-row block, and a second group.
    for tokens in [8, 10, 16, 20] {
        let input = copy(&backend, &inputs(tokens)?)?;
        let mut expected = backend.inner.pool.allocate(&backend.inner.stream, tokens * OUTPUT)?;
        let spec = NvFp4Spec::new(INPUT, OUTPUT)?;
        NvFp4WeightOnly::compile(&backend.inner.compiler, spec, tokens)?.execute(
            &backend.inner.stream,
            &mut NvFp4WeightOnlyLaunch {
                input: &input,
                weight: &weight.weight,
                block_scales: &weight.scales,
                global_scale: &weight.global_scale,
                output: &mut expected,
            },
        )?;
        let expected = read(&backend, &expected)?;
        let peak = expected.iter().map(|value| value.to_f32().abs()).fold(0.0, f32::max);
        assert!(peak > 16.0, "reference peak {peak}");
        let mut marlin = MarlinNvFp4Bf16Linear::new(&backend, tokens, &weight)?;
        for (execution, threads) in marlin_candidates() {
            let mut output = backend.inner.pool.allocate(&backend.inner.stream, tokens * OUTPUT)?;
            marlin.execute(&input, &mut output, threads)?;
            for (index, (expected, actual)) in
                expected.iter().zip(read(&backend, &output)?).enumerate()
            {
                let expected = expected.to_f32();
                assert!(
                    (actual.to_f32() - expected).abs() <= (expected.abs() * 0.01).max(0.125),
                    "{execution:?} tokens {tokens} row {} output {}",
                    index / OUTPUT,
                    index % OUTPUT
                );
            }
        }
    }
    Ok(())
}

fn weight(backend: &CudaBackend, config: NvFp4Config) -> Result<NvFp4WeightOnlyWeight> {
    let packed = (0..OUTPUT * INPUT / 2)
        .map(|index| u8::try_from((index * 37 + index / 251) % 256))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // E4M3 block scales between 0.5 and 1.875.
    let scales = (0..OUTPUT * INPUT / 16)
        .map(|index| u8::try_from(0x30 + (index * 7) % 16))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let tensor =
        |name: &str, shape: Vec<usize>, values: &[u8], scale: bool| -> Result<CudaTensor> {
            let buffer = copy(backend, values)?;
            Ok(if scale {
                CudaTensor::from_f8_e4m3(name.into(), shape, buffer)
            } else {
                CudaTensor::from_u8(name.into(), shape, buffer)
            })
        };
    let scalar = |name: &str, value: f32| -> Result<CudaTensor> {
        Ok(CudaTensor::from_f32(name.into(), vec![1], copy(backend, &[value])?))
    };
    NvFp4WeightOnlyWeight::load(
        backend,
        config,
        NvFp4Tensors {
            weight: &tensor("weight", vec![OUTPUT, INPUT / 2], &packed, false)?,
            weight_scale: &tensor("weight_scale", vec![OUTPUT, INPUT / 16], &scales, true)?,
            weight_scale_2: &scalar("weight_scale_2", 0.5)?,
            input_scale: &scalar("input_scale", 1.0)?,
            scale_mode: NvFp4ScaleMode::Multiplier,
        },
    )
}

/// Distinct values per row, so a misplaced row group changes the result.
fn inputs(tokens: usize) -> Result<Vec<bf16>> {
    (0..tokens * INPUT)
        .map(|index| {
            let (row, column) = (index / INPUT, index % INPUT);
            let value = f32::from(u8::try_from((column * 13 + row * 29) % 61)?) / 32.0 - 0.9375;
            Ok(bf16::from_f32(value))
        })
        .collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
