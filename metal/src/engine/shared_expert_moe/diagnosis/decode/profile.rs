use std::{io::Write, time::Instant};

use super::*;

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    Router,
    GateUp,
    Activation,
    Down,
    Reduction,
    Shared,
    Add,
}

impl SharedExpertMoe {
    pub(super) fn profile_decode(
        &self,
        input: &Array,
        case: Case,
        reference: &[f32],
        stream: &Stream,
    ) -> Result<()> {
        for iteration in 0..4 {
            stream.synchronize()?;
            let mut timer = Timer {
                started: Instant::now(),
                records: Vec::with_capacity(7),
            };
            let routing = self.diagnostic_routing(input, stream)?;
            timer.record(Stage::Router, &[&routing.indices, &routing.weights], stream)?;
            let expanded = input.expand_dims(&[-2, -3], stream)?;
            let (gate, up) =
                self.routed_gate_up.gather(&expanded, &routing.indices, false, stream)?;
            timer.record(Stage::GateUp, &[&gate, &up], stream)?;
            let activated = gate.silu_mul(&up, stream)?;
            timer.record(Stage::Activation, &[&activated], stream)?;
            let output = self.routed_down.gather(&activated, &routing.indices, false, stream)?;
            timer.record(Stage::Down, &[&output], stream)?;
            let routed =
                output.squeeze_axis(-2, stream)?.weighted_sum(&routing.weights, -2, stream)?;
            timer.record(Stage::Reduction, &[&routed], stream)?;
            let shared = self.shared(input, stream)?;
            timer.record(Stage::Shared, &[&shared], stream)?;
            let output = routed.add(&shared, stream)?;
            timer.record(Stage::Add, &[&output], stream)?;
            let values = output.to_vec_f32(stream)?;
            let difference = Difference::between(&values, reference);
            writeln!(
                std::io::stderr().lock(),
                "moe.decode_profile: {}",
                serde_json::json!({
                    "case": case, "iteration": iteration, "warmup": iteration == 0,
                    "records": timer.records, "difference": difference, "elements": values.len(),
                })
            )?;
            assert_eq!(difference.differing, 0, "profile barriers changed MoE output");
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
struct Record {
    stage: Stage,
    graph_ms: f64,
    total_ms: f64,
}

struct Timer {
    started: Instant,
    records: Vec<Record>,
}

impl Timer {
    fn record(&mut self, stage: Stage, roots: &[&Array], stream: &Stream) -> Result<()> {
        let graph_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        stream.eval_many(roots)?;
        stream.synchronize()?;
        self.records.push(Record {
            stage,
            graph_ms,
            total_ms: self.started.elapsed().as_secs_f64() * 1000.0,
        });
        self.started = Instant::now();
        Ok(())
    }
}
