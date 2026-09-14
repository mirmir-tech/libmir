use super::{super::drift as timing, *};

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Condition {
    BaselineBefore,
    ResidentSeparate,
    ResidentFused,
    ResidentAlternating,
    ResidentSeparateAfter,
    BaselineAfter,
}

impl Condition {
    fn variant(self, index: usize) -> Variant {
        match self {
            Self::ResidentFused => Variant::Fused,
            Self::ResidentAlternating => {
                if index < 6 {
                    [Variant::Separate, Variant::Fused][index % 2]
                } else {
                    use Variant::{Fused, Separate};
                    [Separate, Fused, Fused, Separate, Fused, Separate, Separate, Fused]
                        [(index - 6) % 8]
                }
            },
            _ => Variant::Separate,
        }
    }
}

#[derive(serde::Serialize)]
struct Sample {
    condition: Condition,
    phase: timing::Phase,
    variant: Variant,
    timing: timing::Timing,
}

impl SharedExpertMoe {
    pub(in super::super) fn diagnose_fusion_drift(
        &self,
        input: &Array,
        case: Case,
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
        let mut samples = Vec::with_capacity(108);
        let mut differences = Vec::with_capacity(6);
        let mut run = |condition, forward: &mut dyn FnMut(Variant) -> Result<Array>| {
            collect(condition, stream, forward, &mut samples)?;
            let actual = forward(condition.variant(17))?.to_vec_f32(stream)?;
            let difference = Difference::between(&actual, &reference);
            assert_eq!(difference.differing, 0, "fusion drift replay changed output");
            differences.push((condition, difference));
            Ok::<_, Error>(())
        };
        run(Condition::BaselineBefore, &mut |_| self.forward(input, stream))?;
        {
            let RoutedGateUp::Separate { gate, up, fused: None } = &self.routed_gate_up else {
                return Err(Error::InvalidModel("fusion drift requires separate banks".into()));
            };
            let (projection, width) =
                gate.fuse_mxfp4_expert_projection(up, stream)?.ok_or_else(|| {
                    Error::InvalidModel("fusion drift requires compatible MXFP4 banks".into())
                })?;
            let fused = RoutedGateUp::Fused { projection, width, interleaved: false };
            let probe = Probe {
                model: self,
                fused: &fused,
                input,
                case,
                pattern: Pattern::Actual,
                stream,
            };
            for condition in [
                Condition::ResidentSeparate,
                Condition::ResidentFused,
                Condition::ResidentAlternating,
                Condition::ResidentSeparateAfter,
            ] {
                run(condition, &mut |variant| probe.forward(variant))?;
            }
            stream.synchronize()?;
        }
        // Dropping the candidate returns it to the allocator cache; do not clear
        // that cache or alter the normal memory policy for this diagnostic.
        run(Condition::BaselineAfter, &mut |_| self.forward(input, stream))?;
        writeln!(
            std::io::stderr().lock(),
            "moe.fusion_drift: {}",
            serde_json::json!({
                "case": case, "repeats": 16, "samples": samples, "differences": differences,
            })
        )?;
        Ok(())
    }
}

fn collect(
    condition: Condition,
    stream: &Stream,
    forward: &mut dyn FnMut(Variant) -> Result<Array>,
    samples: &mut Vec<Sample>,
) -> Result<()> {
    for index in 0..18 {
        let phase = if index < 6 {
            timing::Phase::Warmup(index)
        } else {
            timing::Phase::Measured(index - 6)
        };
        let variant = condition.variant(index);
        samples.push(Sample {
            condition,
            phase,
            variant,
            timing: timing::sample(stream, || forward(variant))?,
        });
    }
    Ok(())
}
