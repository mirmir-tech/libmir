use std::{fs, sync::Arc, time::Duration};

use runtime::tuning::{TuningConfig, TuningMode};
use tracing_subscriber::util::SubscriberInitExt;

use super::{GateUpExecution, GateUpKey, MetalTuner, TuneAction, forward, key};
use crate::{
    MetalConfig,
    engine::{
        Array, Dtype, Error, QuantizedArrays, QuantizedLinear, Result, Stream,
        attention_batch_tuning::{BatchAttentionExecution, BatchAttentionKey},
        attention_tuning::AttentionKey,
        expert_tuning::{ExpertExecution, ExpertKey},
        kernels::PagedExecution,
        route_tuning::{ExpertActivation, RoutingExecution, RoutingKey},
    },
};

mod budget;

pub(super) fn fixture_key() -> GateUpKey {
    GateUpKey {
        tokens: 1,
        input: 2_816,
        gate: 16_384,
        up: 16_384,
        group_size: 64,
        bits: 4,
        dtype: Dtype::Bfloat16,
    }
}

#[test]
fn cached_mode_reuses_a_persisted_shape_decision()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory =
        std::env::temp_dir().join(format!("libmir-metal-tuning-{}", std::process::id()));
    let config = TuningConfig {
        cache_directory: Some(directory.clone()),
        ..TuningConfig::default()
    };
    let mut startup = MetalTuner::new(config);
    startup.record(fixture_key(), GateUpExecution::Separate, Duration::from_millis(1));
    startup.record_attention(
        attention_key(),
        PagedExecution::TwoPass { blocks: 128, reduction_groups: 16 },
        Duration::from_millis(1),
    );
    startup.record_batch_attention(
        batch_attention_key(),
        BatchAttentionExecution::Rows,
        Duration::from_millis(1),
    );
    startup.record_expert(expert_key(), ExpertExecution::Separate, Duration::from_millis(1));
    startup.record_routing(routing_key(), RoutingExecution::SortedFused, Duration::from_millis(1));
    startup.persist();
    let cached = MetalTuner::new(TuningConfig {
        mode: TuningMode::Cached,
        cache_directory: Some(directory.clone()),
        ..TuningConfig::default()
    });

    assert_eq!(cached.plan(fixture_key()), TuneAction::Execute(GateUpExecution::Separate));
    assert_eq!(
        cached.attention_decision(attention_key()),
        Some(PagedExecution::TwoPass { blocks: 128, reduction_groups: 16 })
    );
    assert_eq!(
        cached.batch_attention_decision(batch_attention_key()),
        Some(BatchAttentionExecution::Rows)
    );
    assert_eq!(cached.expert_decision(expert_key()), Some(ExpertExecution::Separate));
    assert_eq!(cached.routing_decision(routing_key()), Some(RoutingExecution::SortedFused));
    fs::remove_dir_all(directory)?;
    Ok(())
}

pub(super) fn batch_attention_key() -> BatchAttentionKey {
    BatchAttentionKey {
        batch: 10,
        sequence: 1,
        context_bucket: 8_192,
        query_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        dtype: 5,
        causal: false,
        fragmented: false,
        view: true,
    }
}

fn routing_key() -> RoutingKey {
    RoutingKey {
        route_bucket: 128,
        experts: 32,
        top_k: 4,
        input: 2_880,
        intermediate: 2_880,
        group_size: 64,
        bits: 4,
        dtype: Dtype::Bfloat16,
        activation: ExpertActivation::Silu,
        fused_unsorted: true,
    }
}

fn expert_key() -> ExpertKey {
    ExpertKey {
        routes: 4,
        experts: 32,
        input: 2_880,
        gate: 2_880,
        up: 2_880,
        group_size: 64,
        bits: 4,
        dtype: Dtype::Bfloat16,
    }
}

fn attention_key() -> AttentionKey {
    AttentionKey {
        context_bucket: 8_192,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 256,
        page_size: 16,
        dtype: 5,
    }
}

#[test]
fn tuned_forward_executes_real_affine_candidates() -> Result<()> {
    let mut config = MetalConfig::default();
    config.tuning.measurement_iterations = 1;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let gate = affine(64, 96, &stream)?;
    let up = affine(64, 96, &stream)?;
    let fused = gate.fuse_gate_up(&up, &stream)?.ok_or(Error::ShapeOverflow)?;
    let input = Array::from_f32(&values(64), &[1, 1, 64])?;

    let actual = forward(&gate, &up, Some(&fused), &input, &stream)?;
    let expected = (gate.forward(&input, &stream)?, up.forward(&input, &stream)?);
    assert_close(&actual.0, &expected.0, &stream)?;
    assert_close(&actual.1, &expected.1, &stream)?;
    let Ok(tuner) = stream.tuner.lock() else {
        return Err(Error::ShapeOverflow);
    };
    assert!(matches!(tuner.plan(key(&fused, &input)?), TuneAction::Execute(_)));
    Ok(())
}

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_representative_affine_decode_profile() -> Result<()> {
    drop(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .finish()
            .try_init(),
    );
    let mut config = MetalConfig::default();
    config.tuning.startup_budget_ms = 60_000;
    config.tuning.warmup_iterations = 3;
    config.tuning.measurement_iterations = 10;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let gate = affine(2_048, 8_192, &stream)?;
    let up = affine(2_048, 8_192, &stream)?;
    let fused = gate.fuse_gate_up(&up, &stream)?.ok_or(Error::ShapeOverflow)?;
    let input = Array::from_f32(&values(2_048), &[1, 1, 2_048])?;
    let output = forward(&gate, &up, Some(&fused), &input, &stream)?;
    output.0.async_eval()?;
    output.1.async_eval()?;
    stream.synchronize()?;
    let Ok(tuner) = stream.tuner.lock() else {
        return Err(Error::ShapeOverflow);
    };
    let selected = tuner.plan(key(&fused, &input)?);
    println!("metal_gate_up_profile input=2048 output=8192 selected={selected:?}");
    Ok(())
}

fn affine(input: usize, output: usize, stream: &Stream) -> Result<super::BoundLinear> {
    let shape = [i32::try_from(output)?, i32::try_from(input)?];
    let dense =
        Array::from_f32(&values(input.checked_mul(output).ok_or(Error::ShapeOverflow)?), &shape)?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(super::BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
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
    let actual = actual.to_vec_f32_on_stream(stream)?;
    let expected = expected.to_vec_f32_on_stream(stream)?;
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-4)
    );
    Ok(())
}
