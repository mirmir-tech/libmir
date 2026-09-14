use std::{
    io::Write,
    num::NonZeroUsize,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use super::*;

mod lengths;
mod windows;

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Logging {
    Immediate,
    Buffered,
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(tag = "stage", content = "index", rename_all = "snake_case")]
pub(super) enum Phase {
    Warmup(usize),
    Measured(usize),
}

#[derive(serde::Serialize)]
struct Sample {
    block: usize,
    phase: Phase,
    logging: Logging,
    timing: Timing,
}

#[derive(serde::Serialize)]
pub(super) struct Timing {
    start_unix_ns: u128,
    submit_ms: f64,
    tail_ms: f64,
    total_ms: f64,
    before: Memory,
    after: Memory,
}

#[derive(serde::Serialize)]
struct Memory {
    active: usize,
    cached: usize,
    peak_process: usize,
}

impl Memory {
    fn read() -> Result<Self> {
        let stats = crate::engine::memory_stats()?;
        Ok(Self {
            active: stats.active,
            cached: stats.cached,
            peak_process: stats.peak,
        })
    }
}

impl SharedExpertMoe {
    pub(super) fn diagnose_submission_drift(
        &self,
        input: &Array,
        case: Case,
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
        // Fixed ABBA observer comparison. All data buffered, with additional
        // per-sample writes only in the Immediate blocks. No performance gate.
        let mut samples = Vec::with_capacity(72);
        for (block, logging) in
            [Logging::Immediate, Logging::Buffered, Logging::Buffered, Logging::Immediate]
                .into_iter()
                .enumerate()
        {
            for index in 0..18 {
                let phase = if index < 6 {
                    Phase::Warmup(index)
                } else {
                    Phase::Measured(index - 6)
                };
                let sample = Sample {
                    block,
                    phase,
                    logging,
                    timing: sample(stream, || self.forward(input, stream))?,
                };
                if matches!(phase, Phase::Measured(_)) && matches!(logging, Logging::Immediate) {
                    writeln!(
                        std::io::stderr().lock(),
                        "moe.drift_immediate: {}",
                        serde_json::json!(sample)
                    )?;
                }
                samples.push(sample);
            }
        }
        let actual = self.forward(input, stream)?.to_vec_f32(stream)?;
        let difference = Difference::between(&actual, &reference);
        writeln!(
            std::io::stderr().lock(),
            "moe.submission_drift: {}",
            serde_json::json!({
                "case": case, "repeats": 16, "samples": samples, "difference": difference,
            })
        )?;
        assert_eq!(difference.differing, 0, "native replay output changed");
        Ok(())
    }
}

pub(super) fn sample(stream: &Stream, forward: impl FnMut() -> Result<Array>) -> Result<Timing> {
    sample_repeated(stream, NonZeroUsize::try_from(16)?, forward)
}

fn sample_repeated(
    stream: &Stream,
    repeats: NonZeroUsize,
    mut forward: impl FnMut() -> Result<Array>,
) -> Result<Timing> {
    stream.synchronize()?;
    let before = Memory::read()?;
    let start_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let started = Instant::now();
    for _ in 0..repeats.get() {
        let output = forward()?;
        output.async_eval(stream)?;
    }
    let submit_ms = started.elapsed().as_secs_f64() * 1000.0;
    let tail = Instant::now();
    stream.synchronize()?;
    let tail_ms = tail.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let after = Memory::read()?;
    Ok(Timing {
        start_unix_ns,
        submit_ms,
        tail_ms,
        total_ms,
        before,
        after,
    })
}
