use std::time::{Duration, Instant};

use runtime::{backend::SamplingLogits, tuning::select_fastest_candidate};

use super::{
    error::{Error, Result},
    model::NativeOutput,
    session::SessionState,
    step,
};
use crate::engine::{DecodePlan, DecodePlanAction, DecodePlanKey, DecoderModel, Stream};

const CANDIDATES: [DecodePlan; 2] = [DecodePlan::SeparateGateUp, DecodePlan::FusedGateUp];

pub(super) fn decode_pending(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    key: DecodePlanKey,
    token: u32,
    sampling: SamplingLogits,
) -> Result<NativeOutput> {
    match stream.decode_plan_action(&key) {
        DecodePlanAction::Execute(plan) => {
            execute(model, stream, state, token, sampling, plan, false)
        },
        DecodePlanAction::Measure => tune(model, stream, state, key, token, sampling),
    }
}

fn tune(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    key: DecodePlanKey,
    token: u32,
    sampling: SamplingLogits,
) -> Result<NativeOutput> {
    let started = Instant::now();
    let result = measure_candidates(model, stream, state, token, sampling).and_then(|timings| {
        let fastest = usize::from(timings[1] < timings[0]);
        let selected = select_fastest_candidate(
            fastest,
            0,
            &timings,
            stream.config().tuning.minimum_improvement_bps,
        );
        let plan = CANDIDATES[selected];
        stream.record_decode_plan(key.clone(), plan, started.elapsed());
        tracing::info!(
            target: "libmir::metal::tuning",
            ?plan,
            model = %key.model,
            weight_bytes = key.weight_bytes,
            context_bucket = key.context_bucket,
            separate_us = timings[0].as_secs_f64() * 1_000_000.0,
            fused_us = timings[1].as_secs_f64() * 1_000_000.0,
            "selected complete Metal decode execution plan"
        );
        execute(model, stream, state, token, sampling, plan, false)
    });
    result.or_else(|error| {
        stream.record_decode_plan(key, DecodePlan::SeparateGateUp, started.elapsed());
        tracing::warn!(
            target: "libmir::metal::tuning",
            %error,
            "complete Metal decode plan tuning failed; retaining separate gate/up"
        );
        execute(model, stream, state, token, sampling, DecodePlan::SeparateGateUp, false)
    })
}

fn measure_candidates(
    model: &DecoderModel,
    stream: &Stream,
    state: &SessionState,
    token: u32,
    sampling: SamplingLogits,
) -> Result<[Duration; 2]> {
    let config = stream.config().tuning.clone();
    let _ = run_snapshot(model, stream, state, token, sampling, DecodePlan::SeparateGateUp, false)?;
    for _ in 0..config.warmup_iterations {
        for plan in CANDIDATES {
            let _ = run_snapshot(model, stream, state, token, sampling, plan, true)?;
        }
    }
    let mut samples = [Vec::new(), Vec::new()];
    let mut reference = None;
    for iteration in 0..config.measurement_iterations.max(1) {
        for offset in 0..CANDIDATES.len() {
            let index = (usize::try_from(iteration)? + offset) % CANDIDATES.len();
            let (elapsed, output) =
                run_snapshot(model, stream, state, token, sampling, CANDIDATES[index], true)?;
            if reference.replace(output).is_some_and(|expected| expected != output) {
                return Err(Error::InvalidDecodeBatch(
                    "complete decode plan candidates produced different tokens".into(),
                ));
            }
            samples[index].push(elapsed);
        }
    }
    Ok(samples.map(median))
}

fn run_snapshot(
    model: &DecoderModel,
    stream: &Stream,
    state: &SessionState,
    token: u32,
    sampling: SamplingLogits,
    plan: DecodePlan,
    suppress_operator_tuning: bool,
) -> Result<(Duration, u32)> {
    let mut snapshot = state.snapshot()?;
    let started = Instant::now();
    let output =
        execute(model, stream, &mut snapshot, token, sampling, plan, suppress_operator_tuning)?;
    stream.synchronize()?;
    let elapsed = started.elapsed();
    let NativeOutput::Greedy(token) = output else {
        return Err(Error::InvalidDecodeBatch(
            "complete decode tuning requires device-token sampling".into(),
        ));
    };
    Ok((elapsed, token))
}

fn execute(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token: u32,
    sampling: SamplingLogits,
    plan: DecodePlan,
    suppress_operator_tuning: bool,
) -> Result<NativeOutput> {
    stream.with_decode_plan(plan, suppress_operator_tuning, || {
        step::decode_pending(model, stream, state, token, sampling)
    })
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}
