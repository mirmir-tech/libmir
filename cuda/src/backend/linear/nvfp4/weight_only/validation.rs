use mircuda::{Context, DeviceBuffer, Stream, bf16};

use super::{
    NvFp4WeightOnly, NvFp4WeightOnlyTensorCore, NvFp4WeightOnlyWeight,
    marlin::MarlinNvFp4Bf16Linear,
};
use crate::{Result, kernels::NvFp4WeightOnlyLaunch};

const ABSOLUTE_TOLERANCE: f32 = 0.125;
const RELATIVE_TOLERANCE: f32 = 0.01;

#[allow(clippy::too_many_arguments)]
pub(super) fn tensor_core_compatible(
    context: &Context,
    stream: &Stream,
    compressed: &NvFp4WeightOnly,
    tensor_core: &NvFp4WeightOnlyTensorCore,
    weight: &NvFp4WeightOnlyWeight,
    input: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    sample: &mut DeviceBuffer<bf16>,
) -> Result<bool> {
    compressed.execute(stream, &mut launch(input, output, weight))?;
    let reference = read_sample(context, stream, output, sample)?;
    tensor_core.execute(stream, &mut launch(input, output, weight))?;
    let candidate = read_sample(context, stream, output, sample)?;
    Ok(equivalent(&reference, &candidate))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn marlin_compatible(
    context: &Context,
    stream: &Stream,
    compressed: &NvFp4WeightOnly,
    marlin: &mut MarlinNvFp4Bf16Linear,
    weight: &NvFp4WeightOnlyWeight,
    input: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
    sample: &mut DeviceBuffer<bf16>,
) -> Result<[bool; 3]> {
    compressed.execute(stream, &mut launch(input, output, weight))?;
    let reference = read_stratified(context, stream, output, sample)?;
    let mut compatible = [false; 3];
    for (index, (_, config)) in super::tuning::marlin_candidates().into_iter().enumerate() {
        marlin.execute(input, output, config)?;
        compatible[index] =
            equivalent(&reference, &read_stratified(context, stream, output, sample)?);
    }
    Ok(compatible)
}

fn launch<'a>(
    input: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
    weight: &'a NvFp4WeightOnlyWeight,
) -> NvFp4WeightOnlyLaunch<'a> {
    NvFp4WeightOnlyLaunch {
        input,
        weight: &weight.weight,
        block_scales: &weight.scales,
        global_scale: &weight.global_scale,
        output,
    }
}

fn read_sample(
    context: &Context,
    stream: &Stream,
    output: &DeviceBuffer<bf16>,
    sample: &mut DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    stream.copy_device_range(output, 0..sample.len(), sample, 0)?;
    let mut host = context.allocate_pinned(sample.len())?;
    stream.copy_to_host(sample, &mut host)?;
    Ok(host.to_vec()?)
}

fn read_stratified(
    context: &Context,
    stream: &Stream,
    output: &DeviceBuffer<bf16>,
    sample: &mut DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    if sample.len() == output.len() {
        return read_sample(context, stream, output, sample);
    }
    let chunk = sample.len() / 3;
    let starts = [0, (output.len() - chunk) / 2, output.len() - chunk];
    for (index, start) in starts.into_iter().enumerate() {
        stream.copy_device_range(output, start..start + chunk, sample, index * chunk)?;
    }
    let mut host = context.allocate_pinned(sample.len())?;
    stream.copy_to_host(sample, &mut host)?;
    Ok(host.to_vec()?)
}

fn equivalent(reference: &[bf16], candidate: &[bf16]) -> bool {
    reference.len() == candidate.len()
        && reference.iter().zip(candidate).all(|(reference, candidate)| {
            let reference = reference.to_f32();
            let candidate = candidate.to_f32();
            reference.is_finite()
                && candidate.is_finite()
                && (reference - candidate).abs()
                    <= ABSOLUTE_TOLERANCE.max(reference.abs() * RELATIVE_TOLERANCE)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_rounding_and_rejects_drift() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = [1.125, -20.125, 0.125].map(bf16::from_f32);
        let drift = [1.5, -20.0, 0.0].map(bf16::from_f32);
        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
