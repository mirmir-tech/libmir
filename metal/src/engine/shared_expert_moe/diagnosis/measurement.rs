use std::{io::Write, time::Instant};

use super::{Array, Plan, Result, SharedExpertMoe, Stream, numerics::Difference};

impl SharedExpertMoe {
    pub(super) fn compare_plans(
        &self,
        input: &Array,
        candidates: &[Plan<'_>],
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
        let exploratory = stream.config().diagnostics.moe_prefill
            == crate::config::MoePrefill::MeasureIndexedNumerics;
        for &candidate in candidates {
            let output = self.diagnostic_forward(input, candidate, stream)?;
            let values = output.to_vec_f32(stream)?;
            let difference = Difference::between(&values, &reference);
            writeln!(
                std::io::stderr().lock(),
                "moe.parity: {}",
                serde_json::json!({
                    "shape": input.shape()?, "candidate": candidate.name(),
                    "differing": difference.differing, "elements": values.len(),
                    "max_abs": difference.max_abs, "difference": difference,
                    "exploratory_timing": exploratory,
                })
            )?;
            // Cost exploration is explicit and never qualifies model quality.
            if difference.differing != 0 && !exploratory {
                continue;
            }
            for _ in 0..4 {
                for plan in [Plan::Grouped, candidate] {
                    drop(self.measure_plan(input, plan, stream)?);
                }
            }
            for block in 0..3 {
                for (slot, plan) in
                    [Plan::Grouped, candidate, candidate, Plan::Grouped].into_iter().enumerate()
                {
                    let (output, graph_ms, total_ms) = self.measure_plan(input, plan, stream)?;
                    writeln!(
                        std::io::stderr().lock(),
                        "moe.paired: {}",
                        serde_json::json!({
                            "shape": input.shape()?, "candidate": candidate.name(), "plan": plan.name(),
                            "block": block, "slot": slot, "graph_ms": graph_ms, "total_ms": total_ms,
                            "exploratory_timing": exploratory,
                        })
                    )?;
                    drop(output);
                }
            }
        }
        Ok(())
    }

    fn measure_plan(
        &self,
        input: &Array,
        plan: Plan<'_>,
        stream: &Stream,
    ) -> Result<(Array, f64, f64)> {
        stream.synchronize()?;
        let started = Instant::now();
        let output = self.diagnostic_forward(input, plan, stream)?;
        let graph_ms = started.elapsed().as_secs_f64() * 1000.0;
        stream.eval_many(&[&output])?;
        stream.synchronize()?;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        output.detach_graph(stream)?;
        Ok((output, graph_ms, total_ms))
    }
}
