use std::{io::Write, time::Instant};

use super::{
    super::numerics::Difference, Array, Case, Plan, Result, RouterOutput, SharedExpertMoe, Stream,
};

impl SharedExpertMoe {
    pub(super) fn measure_budgets(
        &self,
        case: Case,
        input: &Array,
        routing: &RouterOutput,
        plans: &[Plan<'_>],
        stream: &Stream,
    ) -> Result<()> {
        let reference = self.replay_routed(input, routing, plans[0], stream)?.to_vec_f32(stream)?;
        let mut valid = true;
        for &plan in &plans[1..] {
            let values = self.replay_routed(input, routing, plan, stream)?.to_vec_f32(stream)?;
            let difference = Difference::between(&values, &reference);
            valid &= difference.differing == 0;
            writeln!(
                std::io::stderr().lock(),
                "moe.budget_parity: {}",
                serde_json::json!({
                    "case": case, "plan": plan.name(), "difference": difference, "elements": values.len(),
                })
            )?;
        }
        if !valid {
            return Ok(());
        }
        for _ in 0..3 {
            for &plan in plans {
                self.measure_budget(input, routing, plan, stream)?;
            }
        }
        let mut samples = vec![Vec::new(); plans.len()];
        for block in 0..3 {
            // Rotate a palindrome so every candidate sees both time directions.
            let forward = (0..plans.len())
                .map(|offset| (block + offset) % plans.len())
                .collect::<Vec<_>>();
            let order = forward.iter().chain(forward.iter().rev());
            for (slot, &index) in order.enumerate() {
                let (graph_ms, total_ms) =
                    self.measure_budget(input, routing, plans[index], stream)?;
                samples[index].push(total_ms);
                writeln!(
                    std::io::stderr().lock(),
                    "moe.budget_timing: {}",
                    serde_json::json!({
                        "case": case, "plan": plans[index].name(), "block": block, "slot": slot,
                        "graph_ms": graph_ms, "total_ms": total_ms,
                    })
                )?;
            }
            let stable = samples.iter().all(|values| {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(0.0, f64::max);
                max <= min * 1.15
            });
            if !stable {
                writeln!(
                    std::io::stderr().lock(),
                    "moe.budget_stop: {}",
                    serde_json::json!({"case": case, "block": block})
                )?;
                break;
            }
        }
        Ok(())
    }

    fn measure_budget(
        &self,
        input: &Array,
        routing: &RouterOutput,
        plan: Plan<'_>,
        stream: &Stream,
    ) -> Result<(f64, f64)> {
        stream.synchronize()?;
        let started = Instant::now();
        // Routing is captured outside timing. Include grouping, projections,
        // activation, restoration and the unchanged shared-expert branch.
        let output = self.replay_routed(input, routing, plan, stream)?;
        let graph_ms = started.elapsed().as_secs_f64() * 1000.0;
        stream.eval_many(&[&output])?;
        stream.synchronize()?;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        output.detach_graph(stream)?;
        Ok((graph_ms, total_ms))
    }
}
