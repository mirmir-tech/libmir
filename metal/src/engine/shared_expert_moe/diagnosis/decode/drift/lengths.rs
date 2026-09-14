use std::time::Duration;

use super::*;

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Cadence {
    Continuous,
    Spaced(Duration),
}

#[derive(serde::Serialize)]
struct Sample {
    block: usize,
    phase: Phase,
    repeats: NonZeroUsize,
    cadence: Cadence,
    pause_ms: f64,
    calls_before: usize,
    timing: Timing,
}

impl SharedExpertMoe {
    pub(in super::super) fn diagnose_submission_lengths(
        &self,
        input: &Array,
        case: Case,
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
        // Equal measured work per block; mirrored order exposes temporal drift.
        // Buffer everything, and preserve all samples irrespective of timings.
        let mut samples = Vec::with_capacity(944);
        let mut calls_before = 0;
        let continuous = Cadence::Continuous;
        let spaced = Cadence::Spaced(Duration::from_millis(2));
        for (block, (length, cadence)) in [
            (8, continuous),
            (16, continuous),
            (24, continuous),
            (16, spaced),
            (16, spaced),
            (24, continuous),
            (16, continuous),
            (8, continuous),
        ]
        .into_iter()
        .enumerate()
        {
            let repeats = NonZeroUsize::try_from(length)?;
            for index in 0..(6 + 1536 / length) {
                let phase = if index < 6 {
                    Phase::Warmup(index)
                } else {
                    Phase::Measured(index - 6)
                };
                let pause_ms = match cadence {
                    Cadence::Continuous => 0.0,
                    Cadence::Spaced(delay) => {
                        let started = Instant::now();
                        std::thread::sleep(delay);
                        started.elapsed().as_secs_f64() * 1000.0
                    },
                };
                samples.push(Sample {
                    block,
                    phase,
                    repeats,
                    cadence,
                    pause_ms,
                    calls_before,
                    timing: sample_repeated(stream, repeats, || self.forward(input, stream))?,
                });
                calls_before += length;
            }
        }
        let actual = self.forward(input, stream)?.to_vec_f32(stream)?;
        let difference = Difference::between(&actual, &reference);
        writeln!(
            std::io::stderr().lock(),
            "moe.submission_lengths: {}",
            serde_json::json!({
                "case": case, "samples": samples, "difference": difference,
            })
        )?;
        assert_eq!(difference.differing, 0, "sample length changed native replay output");
        Ok(())
    }
}
