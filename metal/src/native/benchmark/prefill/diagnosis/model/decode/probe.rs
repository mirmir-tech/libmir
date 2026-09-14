use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::native::model::DecodeProfile;

#[derive(serde::Serialize)]
pub(super) struct Sample {
    start_unix_ns: u128,
    submitted_ms: f64,
    tail_wait_ms: f64,
    settled_ms: f64,
    steps: Vec<DecodeProfile>,
}

#[test]
#[ignore = "one bounded native C5/2049 decode CPU/GPU trace; set MIRMIR_BENCH_MODEL"]
fn diagnoses_decode_boundaries() -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference: Option<super::super::Observation> = None;
    // One cold run and three observations; this diagnoses boundaries, not speedup.
    for run in 0..4 {
        let mut sample = None;
        let observation =
            run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
                let (observation, measured) = measure(model, inputs)?;
                sample = Some(measured);
                Ok(observation)
            })?;
        let matches = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "decode.boundaries: {}",
            json!({
                "run": run, "warmup": run == 0, "matches_reference": matches,
                "observation": observation, "sample": sample,
            })
        )?;
        assert!(matches, "native decode tokens or schedule changed");
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}

pub(super) fn measure(
    model: &mut LoadedModel,
    inputs: &mut [DecodeInput],
) -> Result<(Observation, Sample)> {
    let mut tokens = vec![Vec::with_capacity(32); inputs.len()];
    let mut steps = Vec::with_capacity(32);
    let start_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let started = Instant::now();
    for _ in 0..32 {
        for (row, input) in inputs.iter().enumerate() {
            tokens[row].push(input.token);
        }
        let (outputs, profile) = model.decode_batch_profiled(inputs)?;
        steps.push(profile);
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
    }
    let submitted_ms = started.elapsed().as_secs_f64() * 1000.0;
    let tail = Instant::now();
    model.stream.synchronize()?;
    let tail_wait_ms = tail.elapsed().as_secs_f64() * 1000.0;
    let settled_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((
        Observation { tokens, elapsed_ms: settled_ms },
        Sample {
            start_unix_ns,
            submitted_ms,
            tail_wait_ms,
            settled_ms,
            steps,
        },
    ))
}
