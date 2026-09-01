use std::{
    sync::MutexGuard,
    time::{Duration, Instant},
};

use runtime::tuning::{TuningConfig, select_fastest_candidate};
pub(super) use tuner::MetalTuner;

use super::{Array, Dtype, FusedGateUp, Result, Stream, binding::BoundLinear};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub(super) struct GateUpKey {
    tokens: usize,
    input: usize,
    gate: usize,
    up: usize,
    group_size: i32,
    bits: i32,
    dtype: Dtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(super) enum GateUpExecution {
    Separate,
    Fused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TuneAction {
    Execute(GateUpExecution),
    Measure,
}

pub(super) fn forward(
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: Option<&FusedGateUp>,
    input: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let Some(fused) = fused else {
        return execute(GateUpExecution::Separate, gate, up, None, input, stream);
    };
    let active_decode_plan = tuner(stream).active_decode_plan();
    if let Some(plan) = active_decode_plan {
        let execution = if plan.fused_gate_up() {
            GateUpExecution::Fused
        } else {
            GateUpExecution::Separate
        };
        return execute(execution, gate, up, Some(fused), input, stream);
    }
    let key = key(fused, input)?;
    let action = tuner(stream).plan(key);
    match action {
        TuneAction::Execute(execution) => execute(execution, gate, up, Some(fused), input, stream),
        TuneAction::Measure => {
            let started = Instant::now();
            tune(key, gate, up, fused, input, stream).or_else(|error| {
                tuner(stream).record(key, GateUpExecution::Fused, started.elapsed());
                tracing::warn!(
                    target: "libmir::metal::tuning",
                    %error,
                    "Metal gate/up tuning failed; retaining fused execution"
                );
                execute(GateUpExecution::Fused, gate, up, Some(fused), input, stream)
            })
        },
    }
}

pub(super) fn forward_decode_plan(
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedGateUp,
    input: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let plan = tuner(stream).active_decode_plan();
    let execution = if plan.is_some_and(super::DecodePlan::fused_gate_up) {
        GateUpExecution::Fused
    } else {
        GateUpExecution::Separate
    };
    execute(execution, gate, up, Some(fused), input, stream)
}

fn tune(
    key: GateUpKey,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedGateUp,
    input: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let config = stream.config().tuning.clone();
    let started = Instant::now();
    let separate = measure(GateUpExecution::Separate, gate, up, fused, input, stream, &config)?;
    let fused_time = measure(GateUpExecution::Fused, gate, up, fused, input, stream, &config)?;
    let timings = [separate, fused_time];
    let fastest = usize::from(fused_time < separate);
    let selected = select_fastest_candidate(fastest, 1, &timings, config.minimum_improvement_bps);
    let execution = if selected == 0 {
        GateUpExecution::Separate
    } else {
        GateUpExecution::Fused
    };
    {
        let mut tuner = tuner(stream);
        tuner.record(key, execution, started.elapsed());
        tuner.persist();
    }
    tracing::info!(
        target: "libmir::metal::tuning",
        ?execution,
        tokens = key.tokens,
        input_features = key.input,
        gate_features = key.gate,
        up_features = key.up,
        group_size = key.group_size,
        bits = key.bits,
        ?key.dtype,
        separate_us = separate.as_secs_f64() * 1_000_000.0,
        fused_us = fused_time.as_secs_f64() * 1_000_000.0,
        "selected Metal gate/up execution profile"
    );
    execute(execution, gate, up, Some(fused), input, stream)
}

fn measure(
    execution: GateUpExecution,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedGateUp,
    input: &Array,
    stream: &Stream,
    config: &TuningConfig,
) -> Result<Duration> {
    for _ in 0..config.warmup_iterations {
        evaluate(execute(execution, gate, up, Some(fused), input, stream)?, stream)?;
    }
    stream.synchronize()?;
    let iterations = config.measurement_iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        evaluate(execute(execution, gate, up, Some(fused), input, stream)?, stream)?;
    }
    stream.synchronize()?;
    Ok(started.elapsed() / iterations)
}

fn execute(
    execution: GateUpExecution,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: Option<&FusedGateUp>,
    input: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    match (execution, fused) {
        (GateUpExecution::Fused, Some(fused)) => fused.forward_pair(input, stream),
        _ => Ok((gate.forward(input, stream)?, up.forward(input, stream)?)),
    }
}

fn evaluate((gate, up): (Array, Array), stream: &Stream) -> Result<()> {
    gate.async_eval(stream)?;
    up.async_eval(stream)
}

pub(super) fn is_single_token(input: &Array) -> Result<bool> {
    token_count(&input.shape()?).map(|tokens| tokens == 1)
}

fn key(fused: &FusedGateUp, input: &Array) -> Result<GateUpKey> {
    let shape = input.shape()?;
    let tokens = token_count(&shape)?;
    let (input_features, gate, up, group_size, bits) = fused.tuning_geometry();
    Ok(GateUpKey {
        tokens,
        input: input_features,
        gate,
        up,
        group_size,
        bits,
        dtype: input_dtype(input_features, shape.last().copied(), input)?,
    })
}

fn input_dtype(expected: usize, actual: Option<i32>, input: &Array) -> Result<Dtype> {
    if actual.map(usize::try_from).transpose()? != Some(expected) {
        return Err(super::Error::ShapeOverflow);
    }
    input.dtype()
}

fn token_count(shape: &[i32]) -> Result<usize> {
    shape[..shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(usize::try_from(*dimension)?)
                .ok_or(super::Error::ShapeOverflow)
        })
}

fn tuner(stream: &Stream) -> MutexGuard<'_, MetalTuner> {
    stream.tuner.lock().unwrap_or_else(|error| {
        tracing::warn!(
            target: "libmir::metal::tuning",
            "recovering poisoned Metal gate/up tuning state"
        );
        error.into_inner()
    })
}

mod storage;
#[cfg(test)]
mod tests;
mod tuner;
