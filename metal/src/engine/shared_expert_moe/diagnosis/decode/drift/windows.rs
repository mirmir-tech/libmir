use super::*;

#[derive(serde::Serialize)]
struct Window {
    phase: Phase,
    timing: Timing,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Stability {
    Stable,
    ExcessiveSpread,
}

impl SharedExpertMoe {
    pub(in super::super) fn diagnose_continuous_windows(
        &self,
        input: &Array,
        case: Case,
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
        let repeats = NonZeroUsize::try_from(1536)?;
        let mut windows = Vec::with_capacity(15);
        for index in 0..15 {
            let phase = if index < 3 {
                Phase::Warmup(index)
            } else {
                Phase::Measured(index - 3)
            };
            windows.push(Window {
                phase,
                timing: sample_repeated(stream, repeats, || self.forward(input, stream))?,
            });
        }
        let min = windows[3..].iter().map(|w| w.timing.total_ms).fold(f64::INFINITY, f64::min);
        let max = windows[3..].iter().map(|w| w.timing.total_ms).fold(0.0, f64::max);
        let stability = if max <= min * 1.15 {
            Stability::Stable
        } else {
            Stability::ExcessiveSpread
        };
        let actual = self.forward(input, stream)?.to_vec_f32(stream)?;
        let difference = Difference::between(&actual, &reference);
        writeln!(
            std::io::stderr().lock(),
            "moe.continuous_windows: {}",
            serde_json::json!({
                "case": case, "repeats": repeats, "windows": windows,
                "stability": stability, "difference": difference,
            })
        )?;
        assert_eq!(difference.differing, 0, "continuous window changed native replay output");
        Ok(())
    }
}
