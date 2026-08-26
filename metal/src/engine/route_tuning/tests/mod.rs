use std::sync::Arc;

use tracing_subscriber::util::SubscriberInitExt;

use super::{ExpertActivation, RoutingExecution, RoutingSpec, fallback, forward, key};
use crate::{
    MetalConfig,
    engine::{
        Array, Error, FusedExpertGateUp, QuantizedArrays, QuantizedLinear, Result, Stream,
        binding::BoundLinear, expert_tuning,
    },
};

mod benchmark;
mod patterns;

#[test]
fn buckets_route_shapes_without_reading_indices() -> Result<()> {
    let input = Array::from_f32(&vec![0.0; 17 * 64], &[1, 17, 64])?;
    let indices = Array::from_u32(&vec![0; 17 * 4], &[1, 17, 4])?;
    let profile = key(spec(8, 96, false), &input, &indices)?;
    assert_eq!(profile.route_bucket, 128);
    assert_eq!(profile.top_k, 4);
    let below_input = Array::from_f32(&vec![0.0; 9 * 64], &[1, 9, 64])?;
    let below_threshold = Array::from_u32(&[0; 9 * 4], &[1, 9, 4])?;
    assert_eq!(key(spec(8, 96, false), &below_input, &below_threshold)?.route_bucket, 64);
    assert_eq!(fallback(&below_threshold)?, RoutingExecution::Unsorted);
    Ok(())
}

#[test]
fn tuned_routing_matches_unsorted_expert_mlp() -> Result<()> {
    let mut config = MetalConfig::default();
    config.tuning.measurement_iterations = 1;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let weights = weights(4, 64, 128, &stream)?;
    let input = Array::from_f32(&values(2 * 64), &[1, 2, 64])?;
    let indices = Array::from_u32(&[0, 2, 1, 3], &[1, 2, 2])?;
    let actual = forward(
        spec(4, 128, true),
        &input,
        &indices,
        &stream,
        (
            |indices| sorted(&weights, &input, indices, &stream),
            |indices| sorted(&weights, &input, indices, &stream),
            |indices| sorted(&weights, &input, indices, &stream),
            |indices| unsorted(&weights, &input, indices, true, &stream),
            |indices| unsorted(&weights, &input, indices, false, &stream),
        ),
    )?;
    let expected = unsorted(&weights, &input, &indices, false, &stream)?;
    assert_close(&actual, &expected, &stream)
}

#[test]
fn fused_restore_reduction_matches_graph_operations() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let weights = weights(4, 64, 128, &stream)?;
    let input = Array::from_f32(&values(2 * 64), &[1, 2, 64])?;
    let indices = Array::from_u32(&[0, 2, 1, 3], &[1, 2, 2])?;
    let routing = Array::from_f32(&[0.25, 0.75, 0.6, 0.4], &[1, 2, 2])?;
    let sorted = input.sort_expert_inputs(&indices, &stream)?;
    let output = mlp(&weights, &sorted.input, &sorted.indices, true, &stream)?;
    let actual = sorted.restore_weighted(&output, &routing, &stream)?;
    let expected = sorted.restore(&output, &stream)?.weighted_sum(&routing, -2, &stream)?;
    assert_close(&actual, &expected, &stream)
}

#[test]
fn kernel_grouping_is_sorted_and_restorable() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 4, 2])?;
    let indices = Array::from_u32(&[3, 0, 3, 1, 3, 2, 3, 0], &[1, 4, 2])?;
    let grouped = input.group_expert_inputs(&indices, 4, &stream)?;
    assert_eq!(grouped.indices.to_vec_u32(&stream)?, [0, 0, 1, 2, 3, 3, 3, 3]);
    assert_eq!(
        grouped.restore(&grouped.input, &stream)?.to_vec_f32(&stream)?,
        [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0]
    );
    Ok(())
}

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_sorted_unsorted_crossover() -> Result<()> {
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
    let weights = weights(8, 256, 512, &stream)?;
    for fused in [false, true] {
        for tokens in [1, 4, 16, 64] {
            let input = Array::from_f32(&values(tokens * 256), &[1, i32::try_from(tokens)?, 256])?;
            let routes = tokens * 4;
            let indices = (0..routes)
                .map(|index| Ok(u32::try_from(index % 8)?))
                .collect::<Result<Vec<_>>>()?;
            let indices = Array::from_u32(&indices, &[1, i32::try_from(tokens)?, 4])?;
            let routing = Array::from_f32(&vec![0.25; routes], &[1, i32::try_from(tokens)?, 4])?;
            forward(
                spec(8, 512, fused),
                &input,
                &indices,
                &stream,
                (
                    |indices| {
                        sorted(&weights, &input, indices, &stream)?
                            .weighted_sum(&routing, -2, &stream)
                    },
                    |indices| {
                        let sorted = input.sort_expert_inputs(indices, &stream)?;
                        let output = mlp(&weights, &sorted.input, &sorted.indices, true, &stream)?;
                        sorted.restore_weighted(&output, &routing, &stream)
                    },
                    |indices| {
                        let grouped = input.group_expert_inputs(indices, 8, &stream)?;
                        let output =
                            mlp(&weights, &grouped.input, &grouped.indices, true, &stream)?;
                        grouped.restore_weighted(&output, &routing, &stream)
                    },
                    |indices| {
                        unsorted(&weights, &input, indices, fused, &stream)?
                            .weighted_sum(&routing, -2, &stream)
                    },
                    |indices| {
                        unsorted(&weights, &input, indices, fused, &stream)?
                            .weighted_sum(&routing, -2, &stream)
                    },
                ),
            )?
            .async_eval(&stream)?;
            stream.synchronize()?;
            println!(
                "metal_expert_routing_profile fused_unsorted={fused} tokens={tokens} routes={routes}"
            );
        }
    }
    Ok(())
}

fn spec(experts: usize, intermediate: usize, fused_unsorted: bool) -> RoutingSpec {
    RoutingSpec {
        experts,
        intermediate,
        group_size: 64,
        bits: 4,
        activation: ExpertActivation::Silu,
        fused_unsorted,
    }
}

struct Weights {
    gate: BoundLinear,
    up: BoundLinear,
    down: BoundLinear,
    fused: FusedExpertGateUp,
}

fn weights(experts: usize, input: usize, intermediate: usize, stream: &Stream) -> Result<Weights> {
    let gate = affine(experts, input, intermediate, stream)?;
    let up = affine(experts, input, intermediate, stream)?;
    let fused = gate.fuse_expert_gate_up(&up, stream)?.ok_or(Error::ShapeOverflow)?;
    Ok(Weights {
        gate,
        up,
        down: affine(experts, intermediate, input, stream)?,
        fused,
    })
}

fn sorted(weights: &Weights, input: &Array, indices: &Array, stream: &Stream) -> Result<Array> {
    let sorted = input.sort_expert_inputs(indices, stream)?;
    let output = mlp(weights, &sorted.input, &sorted.indices, true, stream)?;
    sorted.restore(&output, stream)
}

fn unsorted(
    weights: &Weights,
    input: &Array,
    indices: &Array,
    fused: bool,
    stream: &Stream,
) -> Result<Array> {
    let input = input.expand_dims(&[-2, -3], stream)?;
    let (gate, up) = if fused {
        expert_tuning::forward(&weights.gate, &weights.up, &weights.fused, &input, indices, stream)?
    } else {
        (
            weights.gate.gather(&input, indices, false, stream)?,
            weights.up.gather(&input, indices, false, stream)?,
        )
    };
    weights
        .down
        .gather(&gate.silu_mul(&up, stream)?, indices, false, stream)?
        .squeeze_axis(-2, stream)
}

fn mlp(
    weights: &Weights,
    input: &Array,
    indices: &Array,
    sorted: bool,
    stream: &Stream,
) -> Result<Array> {
    let gate = weights.gate.gather(input, indices, sorted, stream)?;
    let up = weights.up.gather(input, indices, sorted, stream)?;
    weights.down.gather(&gate.silu_mul(&up, stream)?, indices, sorted, stream)
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
