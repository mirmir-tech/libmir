use std::time::Duration;

use mircuda::{DeviceBuffer, bf16};
use runtime::tuning::{select_fastest_candidate, select_robust_candidate};

use super::AutoNvFp4Experts;
use crate::{
    Error, ExecutionPhase, Result,
    kernels::{RoutePattern, RoutePatternGenerator, RoutePatternSpec},
};

struct RoutePatterns {
    balanced: DeviceBuffer<u32>,
    hot_set: DeviceBuffer<u32>,
}

impl AutoNvFp4Experts {
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn measure(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(usize, Duration, Duration)> {
        let patterns = (self.request.phase == ExecutionPhase::Prefill)
            .then(|| self.route_patterns())
            .transpose()?;
        let mut route_sets = vec![("resident", selected)];
        if let Some(patterns) = &patterns {
            route_sets.extend([("balanced", &patterns.balanced), ("hot_set", &patterns.hot_set)]);
        }
        let (warmup, iterations) = self.backend.auto_tuner().iterations(self.request.tokens);
        let mut timings = vec![Vec::with_capacity(route_sets.len()); self.candidates.len()];
        let mut elapsed = Duration::ZERO;
        for (distribution, routes) in route_sets {
            for (index, candidate) in self.candidates.iter_mut().enumerate() {
                for _ in 0..warmup {
                    candidate.plan.execute(input, routes, routing, output)?;
                }
                let started = self.backend.context().create_event(true)?;
                let completed = self.backend.context().create_event(true)?;
                started.record(self.backend.stream())?;
                for _ in 0..iterations {
                    candidate.plan.execute(input, routes, routing, output)?;
                }
                completed.record(self.backend.stream())?;
                completed.synchronize()?;
                let average = Duration::from_secs_f32(
                    started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
                );
                tracing::debug!(
                    target: "libmir::cuda::tuning",
                    distribution,
                    execution = ?candidate.execution,
                    average_us = average.as_secs_f64() * 1_000_000.0,
                    "measured CUDA MoE route distribution"
                );
                elapsed = elapsed
                    .saturating_add(average.saturating_mul(iterations.saturating_add(warmup)));
                timings[index].push(average);
            }
        }
        let winner = if patterns.is_some() {
            select_robust_candidate(
                self.fallback,
                &timings,
                self.backend.auto_tuner().minimum_improvement_bps(),
            )
        } else {
            let fastest = timings
                .iter()
                .enumerate()
                .min_by_key(|(_, values)| values[0])
                .map(|value| value.0)
                .ok_or(Error::InvalidExecutionPlan("MoE tuner has no candidates"))?;
            let scalar = timings.iter().map(|values| values[0]).collect::<Vec<_>>();
            select_fastest_candidate(
                fastest,
                self.fallback,
                &scalar,
                self.backend.auto_tuner().minimum_improvement_bps(),
            )
        };
        let total = timings[winner].iter().copied().fold(Duration::ZERO, Duration::saturating_add);
        let scenarios = u32::try_from(timings[winner].len())?;
        Ok((winner, total / scenarios, elapsed))
    }

    fn route_patterns(&self) -> Result<RoutePatterns> {
        let spec = RoutePatternSpec {
            tokens: self.request.tokens,
            experts: self.request.experts,
            top_k: self.request.top_k,
        };
        let generator = RoutePatternGenerator::compile(self.backend.compiler(), spec)?;
        let routes = spec
            .tokens
            .checked_mul(spec.top_k)
            .ok_or(Error::InvalidRouter("route pattern size overflow"))?;
        let mut balanced = self.backend.pool().allocate(self.backend.stream(), routes)?;
        let mut hot_set = self.backend.pool().allocate(self.backend.stream(), routes)?;
        generator.execute(self.backend.stream(), RoutePattern::Balanced, &mut balanced)?;
        generator.execute(self.backend.stream(), RoutePattern::HotSet, &mut hot_set)?;
        Ok(RoutePatterns { balanced, hot_set })
    }
}
