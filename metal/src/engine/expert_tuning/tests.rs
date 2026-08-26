use std::sync::Arc;

use tracing_subscriber::util::SubscriberInitExt;

use super::{Result, elements, forward};
use crate::{
    MetalConfig,
    engine::{Array, Error, QuantizedArrays, QuantizedLinear, Stream, binding::BoundLinear},
};

#[test]
fn counts_routes_without_reading_indices_to_the_host() -> Result<()> {
    assert_eq!(elements(&[1, 8, 4])?, 32);
    Ok(())
}

#[test]
fn tuned_forward_matches_separate_expert_projections() -> Result<()> {
    let mut config = MetalConfig::default();
    config.tuning.measurement_iterations = 1;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let gate = affine(4, 64, 96, &stream)?;
    let up = affine(4, 64, 96, &stream)?;
    let fused = gate.fuse_expert_gate_up(&up, &stream)?.ok_or(Error::ShapeOverflow)?;
    let input = Array::from_f32(&values(64), &[1, 1, 1, 64])?;
    let indices = Array::from_u32(&[1, 3], &[1, 1, 2])?;
    let actual = forward(&gate, &up, &fused, &input, &indices, &stream)?;
    let expected = (
        gate.gather(&input, &indices, false, &stream)?,
        up.gather(&input, &indices, false, &stream)?,
    );
    assert_close(&actual.0, &expected.0, &stream)?;
    assert_close(&actual.1, &expected.1, &stream)
}

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_representative_expert_gate_up_profile() -> Result<()> {
    drop(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .finish()
            .try_init(),
    );
    let mut config = MetalConfig::default();
    config.tuning.warmup_iterations = 3;
    config.tuning.measurement_iterations = 10;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let gate = affine(8, 512, 1_024, &stream)?;
    let up = affine(8, 512, 1_024, &stream)?;
    let fused = gate.fuse_expert_gate_up(&up, &stream)?.ok_or(Error::ShapeOverflow)?;
    let input = Array::from_f32(&values(512), &[1, 1, 1, 512])?;
    let indices = Array::from_u32(&[1, 3, 5, 7], &[1, 1, 4])?;
    let output = forward(&gate, &up, &fused, &input, &indices, &stream)?;
    output.0.async_eval(&stream)?;
    output.1.async_eval(&stream)?;
    stream.synchronize()?;
    println!("metal_expert_gate_up_profile experts=8 input=512 output=1024 routes=4");
    Ok(())
}

fn affine(experts: usize, input: usize, output: usize, stream: &Stream) -> Result<BoundLinear> {
    let elements = experts
        .checked_mul(input)
        .and_then(|value| value.checked_mul(output))
        .ok_or(Error::ShapeOverflow)?;
    let shape = [i32::try_from(experts)?, i32::try_from(output)?, i32::try_from(input)?];
    let dense = Array::from_f32(&values(elements), &shape)?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}

fn values(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let value = u8::try_from(index % 17).map_or(0.0, f32::from);
            (value - 8.0) / 32.0
        })
        .collect()
}

fn assert_close(actual: &Array, expected: &Array, stream: &Stream) -> Result<()> {
    let actual = actual.to_vec_f32(stream)?;
    let expected = expected.to_vec_f32(stream)?;
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-4)
    );
    Ok(())
}
