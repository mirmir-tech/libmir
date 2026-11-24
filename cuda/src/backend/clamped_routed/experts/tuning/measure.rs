use std::time::Duration;

use mircuda::{DeviceBuffer, bf16};
use runtime::tuning::{select_fastest_candidate, select_robust_candidate};

use super::{AutoClampedExperts, ClampedRoutedExpertWeights};
use crate::{
    Error, ExecutionPhase, Result,
    kernels::{RoutePattern, RoutePatternGenerator, RoutePatternSpec},
};

/// A candidate whose probe exceeds the best probe this many times is not
/// measured further.
const HOPELESS_FACTOR: u32 = 8;

struct RoutePatterns {
    balanced: DeviceBuffer<u32>,
    hot_set: DeviceBuffer<u32>,
}

impl AutoClampedExperts {
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    pub(super) fn measure(
        &self,
        weights: &ClampedRoutedExpertWeights,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        activated: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(usize, Duration, Duration)> {
        let patterns = (self.phase == ExecutionPhase::Prefill)
            .then(|| self.route_patterns())
            .transpose()?;
        let mut route_sets = vec![("resident", selected)];
        if let Some(patterns) = &patterns {
            route_sets.extend([("balanced", &patterns.balanced), ("hot_set", &patterns.hot_set)]);
        }
        let (warmup, iterations) = self.backend.auto_tuner().iterations(self.tokens);
        let mut timings = vec![Vec::with_capacity(route_sets.len()); self.candidates.len()];
        let mut elapsed = Duration::ZERO;
        for (distribution, routes) in route_sets {
            // One timed probe per candidate first: a candidate far slower than
            // the best probe cannot win, so it keeps its probe time instead of
            // a full measurement (the unfused gpt-oss prefill paths ran ~200x
            // slower than Marlin and dominated startup tuning).
            let mut probes = Vec::with_capacity(self.candidates.len());
            for candidate in &self.candidates {
                let run = || {
                    candidate.execute(
                        self.backend.stream(),
                        weights,
                        input,
                        routes,
                        routing,
                        activated,
                        partial,
                        output,
                    )
                };
                probes.push(self.time(1, run)?);
            }
            let best = probes.iter().copied().min().unwrap_or(Duration::ZERO);
            for (index, candidate) in self.candidates.iter().enumerate() {
                let mut run = || {
                    candidate.execute(
                        self.backend.stream(),
                        weights,
                        input,
                        routes,
                        routing,
                        activated,
                        partial,
                        output,
                    )
                };
                let average = if probes[index] > best.saturating_mul(HOPELESS_FACTOR) {
                    probes[index]
                } else {
                    for _ in 0..warmup {
                        run()?;
                    }
                    self.time(iterations, run)?
                };
                tracing::debug!(
                    target: "libmir::cuda::tuning",
                    distribution,
                    execution = ?candidate.execution,
                    average_us = average.as_secs_f64() * 1_000_000.0,
                    "measured clamped CUDA MoE route distribution"
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
                .ok_or(Error::InvalidExecutionPlan("clamped MoE tuner has no candidates"))?;
            let scalar = timings.iter().map(|values| values[0]).collect::<Vec<_>>();
            select_fastest_candidate(
                fastest,
                self.fallback,
                &scalar,
                self.backend.auto_tuner().minimum_improvement_bps(),
            )
        };
        let total = timings[winner].iter().copied().fold(Duration::ZERO, Duration::saturating_add);
        Ok((winner, total / u32::try_from(timings[winner].len())?, elapsed))
    }

    /// Mean device time of `iterations` runs.
    #[allow(clippy::cast_precision_loss)]
    fn time(&self, iterations: u32, mut run: impl FnMut() -> Result<()>) -> Result<Duration> {
        let started = self.backend.context().create_event(true)?;
        let completed = self.backend.context().create_event(true)?;
        started.record(self.backend.stream())?;
        for _ in 0..iterations {
            run()?;
        }
        completed.record(self.backend.stream())?;
        completed.synchronize()?;
        Ok(Duration::from_secs_f32(
            started.elapsed_ms(&completed)? / (iterations as f32 * 1_000.0),
        ))
    }

    fn route_patterns(&self) -> Result<RoutePatterns> {
        let spec = RoutePatternSpec {
            tokens: self.tokens,
            experts: self.config.experts,
            top_k: self.config.top_k,
        };
        let generator = RoutePatternGenerator::compile(self.backend.compiler(), spec)?;
        let routes = spec
            .tokens
            .checked_mul(spec.top_k)
            .ok_or(Error::InvalidRouter("clamped route pattern size overflow"))?;
        let mut balanced = self.backend.pool().allocate(self.backend.stream(), routes)?;
        let mut hot_set = self.backend.pool().allocate(self.backend.stream(), routes)?;
        generator.execute(self.backend.stream(), RoutePattern::Balanced, &mut balanced)?;
        generator.execute(self.backend.stream(), RoutePattern::HotSet, &mut hot_set)?;
        Ok(RoutePatterns { balanced, hot_set })
    }
}
