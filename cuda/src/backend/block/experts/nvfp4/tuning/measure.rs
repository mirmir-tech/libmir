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
            let measured =
                self.measure_interleaved(input, routes, routing, output, warmup, iterations)?;
            for (index, average) in measured.into_iter().enumerate() {
                tracing::debug!(
                    target: "libmir::cuda::tuning",
                    distribution,
                    execution = ?self.candidates[index].execution,
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

    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    fn measure_interleaved(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        warmup: u32,
        iterations: u32,
    ) -> Result<Vec<Duration>> {
        for round in 0..warmup {
            self.execute_round(round as usize, input, selected, routing, output, None)?;
        }
        let mut totals = vec![Duration::ZERO; self.candidates.len()];
        for round in 0..iterations {
            self.execute_round(
                round as usize,
                input,
                selected,
                routing,
                output,
                Some(&mut totals),
            )?;
        }
        Ok(totals.into_iter().map(|total| total / iterations.max(1)).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_round(
        &mut self,
        offset: usize,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        mut totals: Option<&mut [Duration]>,
    ) -> Result<()> {
        let candidates = self.candidates.len();
        for step in 0..candidates {
            let index = (offset + step) % candidates;
            let started = self.backend.context().create_event(true)?;
            let completed = self.backend.context().create_event(true)?;
            started.record(self.backend.stream())?;
            self.candidates[index].plan.execute(input, selected, routing, output)?;
            completed.record(self.backend.stream())?;
            completed.synchronize()?;
            if let Some(totals) = totals.as_deref_mut() {
                totals[index] = totals[index].saturating_add(Duration::from_secs_f32(
                    started.elapsed_ms(&completed)? / 1_000.0,
                ));
            }
        }
        Ok(())
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
