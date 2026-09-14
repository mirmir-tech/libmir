use std::{io::Write, time::Instant};

use super::{Array, Plan, Result, SharedExpertMoe, Stream};

#[derive(Clone, Copy)]
enum Stage {
    Router,
    Group,
    GateUp,
    Activation,
    Down,
    Restore,
    Shared,
    Add,
}

impl SharedExpertMoe {
    pub(super) fn profile_prefill(&self, input: &Array, stream: &Stream) -> Result<()> {
        // Two settled repetitions; barriers deliberately change overlap.
        for iteration in 0..3 {
            let mut timer = Timer {
                started: Instant::now(),
                iteration,
                sequence: input.shape()?[1],
            };
            let routing = self.diagnostic_routing(input, stream)?;
            timer.record(Stage::Router, &[&routing.indices, &routing.weights], stream)?;
            let grouped =
                Plan::Grouped.prepare(input, &routing.indices, self.config.expert_count, stream)?;
            timer.record(Stage::Group, &[&grouped.input, &grouped.indices], stream)?;
            let (gate, up) =
                self.routed_gate_up.gather(&grouped.input, &grouped.indices, true, stream)?;
            timer.record(Stage::GateUp, &[&gate, &up], stream)?;
            let activated = gate.silu_mul(&up, stream)?;
            timer.record(Stage::Activation, &[&activated], stream)?;
            let output = self.routed_down.gather(&activated, &grouped.indices, true, stream)?;
            timer.record(Stage::Down, &[&output], stream)?;
            let routed = grouped.restore_weighted(&output, &routing.weights, stream)?;
            timer.record(Stage::Restore, &[&routed], stream)?;
            let shared = self.shared(input, stream)?;
            timer.record(Stage::Shared, &[&shared], stream)?;
            let output = routed.add(&shared, stream)?;
            timer.record(Stage::Add, &[&output], stream)?;
        }
        Ok(())
    }
}

struct Timer {
    started: Instant,
    iteration: usize,
    sequence: i32,
}

impl Timer {
    fn record(&mut self, stage: Stage, roots: &[&Array], stream: &Stream) -> Result<()> {
        let graph_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        stream.eval_many(roots)?;
        stream.synchronize()?;
        let total_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        if self.iteration > 0 {
            writeln!(
                std::io::stderr().lock(),
                "moe.component: {}",
                serde_json::json!({
                    "stage": stage.name(), "sequence": self.sequence, "iteration": self.iteration,
                    "graph_ms": graph_ms, "total_ms": total_ms,
                })
            )?;
        }
        self.started = Instant::now();
        Ok(())
    }
}

impl Stage {
    const fn name(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Group => "group",
            Self::GateUp => "gate_up",
            Self::Activation => "activation",
            Self::Down => "down",
            Self::Restore => "restore",
            Self::Shared => "shared",
            Self::Add => "add",
        }
    }
}
