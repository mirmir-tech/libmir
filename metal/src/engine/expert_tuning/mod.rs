use std::time::{Duration, Instant};

use runtime::tuning::{TuningMode, select_fastest_candidate};

use super::{Array, Dtype, Error, FusedExpertGateUp, Result, Stream, binding::BoundLinear};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ExpertKey {
    pub routes: usize,
    pub experts: usize,
    pub input: usize,
    pub gate: usize,
    pub up: usize,
    pub group_size: i32,
    pub bits: i32,
    pub dtype: Dtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ExpertExecution {
    Separate,
    Fused,
}

pub(super) fn forward(
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedExpertGateUp,
    input: &Array,
    indices: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let key = key(fused, input, indices)?;
    let action = {
        let tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if tuner.config().mode == TuningMode::Disabled {
            Some(ExpertExecution::Fused)
        } else if let Some(execution) = tuner.expert_decision(key) {
            Some(execution)
        } else if tuner.config().mode == TuningMode::Startup && tuner.expert_budget_available() {
            None
        } else {
            Some(ExpertExecution::Fused)
        }
    };
    action.map_or_else(
        || {
            let started = Instant::now();
            tune(key, gate, up, fused, input, indices, stream).or_else(|error| {
                stream
                    .tuner
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .record_expert(key, ExpertExecution::Fused, started.elapsed());
                tracing::warn!(
                    target: "libmir::metal::tuning",
                    %error,
                    "Metal expert gate/up tuning failed; retaining fused execution"
                );
                execute(ExpertExecution::Fused, gate, up, fused, input, indices, stream)
            })
        },
        |execution| execute(execution, gate, up, fused, input, indices, stream),
    )
}

fn tune(
    key: ExpertKey,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedExpertGateUp,
    input: &Array,
    indices: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let config = stream.config().tuning.clone();
    let started = Instant::now();
    let separate = measure(ExpertExecution::Separate, gate, up, fused, input, indices, stream)?;
    let fused_time = measure(ExpertExecution::Fused, gate, up, fused, input, indices, stream)?;
    let timings = [separate, fused_time];
    let fastest = usize::from(fused_time < separate);
    let selected = select_fastest_candidate(fastest, 1, &timings, config.minimum_improvement_bps);
    let execution = if selected == 0 {
        ExpertExecution::Separate
    } else {
        ExpertExecution::Fused
    };
    {
        let mut tuner = stream.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        tuner.record_expert(key, execution, started.elapsed());
        tuner.persist();
    }
    tracing::info!(
        target: "libmir::metal::tuning",
        ?execution,
        routes = key.routes,
        experts = key.experts,
        input_features = key.input,
        gate_features = key.gate,
        up_features = key.up,
        separate_us = separate.as_secs_f64() * 1_000_000.0,
        fused_us = fused_time.as_secs_f64() * 1_000_000.0,
        "selected Metal expert gate/up execution profile"
    );
    execute(execution, gate, up, fused, input, indices, stream)
}

fn measure(
    execution: ExpertExecution,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedExpertGateUp,
    input: &Array,
    indices: &Array,
    stream: &Stream,
) -> Result<Duration> {
    let config = &stream.config().tuning;
    for _ in 0..config.warmup_iterations {
        evaluate(execute(execution, gate, up, fused, input, indices, stream)?, stream)?;
    }
    stream.synchronize()?;
    let iterations = config.measurement_iterations.max(1);
    let started = Instant::now();
    for _ in 0..iterations {
        evaluate(execute(execution, gate, up, fused, input, indices, stream)?, stream)?;
    }
    stream.synchronize()?;
    Ok(started.elapsed() / iterations)
}

fn execute(
    execution: ExpertExecution,
    gate: &BoundLinear,
    up: &BoundLinear,
    fused: &FusedExpertGateUp,
    input: &Array,
    indices: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    match execution {
        ExpertExecution::Separate => Ok((
            gate.gather(input, indices, false, stream)?,
            up.gather(input, indices, false, stream)?,
        )),
        ExpertExecution::Fused => fused.forward(input, indices, stream),
    }
}

fn evaluate((gate, up): (Array, Array), stream: &Stream) -> Result<()> {
    gate.async_eval(stream)?;
    up.async_eval(stream)
}

fn key(fused: &FusedExpertGateUp, input: &Array, indices: &Array) -> Result<ExpertKey> {
    let (experts, input_width, gate, up, group_size, bits) = fused.tuning_geometry();
    let input_shape = input.shape()?;
    if input_shape.last().copied().map(usize::try_from).transpose()? != Some(input_width) {
        return Err(Error::ShapeOverflow);
    }
    Ok(ExpertKey {
        routes: elements(&indices.shape()?)?,
        experts,
        input: input_width,
        gate,
        up,
        group_size,
        bits,
        dtype: input.dtype()?,
    })
}

fn elements(shape: &[i32]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
    })
}

#[cfg(test)]
mod tests;
