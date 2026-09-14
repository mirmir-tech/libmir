use std::time::Instant;

use super::*;

const REPEATS: usize = 16;

impl Probe<'_> {
    pub(super) fn measure(&self) -> Result<()> {
        for _ in 0..3 {
            for variant in [Variant::Separate, Variant::Fused] {
                self.sample(variant)?;
            }
        }
        let mut samples = Vec::new();
        for block in 0..3 {
            let order = if block % 2 == 0 {
                [Variant::Separate, Variant::Fused, Variant::Fused, Variant::Separate]
            } else {
                [Variant::Fused, Variant::Separate, Variant::Separate, Variant::Fused]
            };
            for (slot, variant) in order.into_iter().enumerate() {
                let milliseconds = self.sample(variant)?;
                samples.push((variant, milliseconds));
                let stable = [Variant::Separate, Variant::Fused].into_iter().all(|variant| {
                    let values =
                        samples.iter().filter(|s| s.0 == variant).map(|s| s.1).collect::<Vec<_>>();
                    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = values.iter().copied().fold(0.0_f64, f64::max);
                    max <= min * 1.15
                });
                writeln!(
                    std::io::stderr().lock(),
                    "moe.decode_fusion_timing: {}",
                    serde_json::json!({
                        "case": self.case, "pattern": self.pattern, "block": block, "slot": slot,
                        "variant": variant, "per_call_ms": milliseconds, "repeats": REPEATS, "stable": stable,
                    })
                )?;
                if !stable {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn sample(&self, variant: Variant) -> Result<f64> {
        self.stream.synchronize()?;
        let started = Instant::now();
        for _ in 0..REPEATS {
            let output = self.forward(variant)?;
            output.async_eval(self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(started.elapsed().as_secs_f64() * 1000.0 / 16.0)
    }
}
