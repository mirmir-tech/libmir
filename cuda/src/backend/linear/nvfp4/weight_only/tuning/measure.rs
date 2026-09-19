use std::time::Duration;

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{Execution, Selection};
use crate::{Result, kernels::NvFp4WeightOnlyLaunch};

/// Copied between timed runs so every candidate reads its weights from memory,
/// as it does between real decoder layers.
const EVICTION_ELEMENTS: usize = 16 << 20;

impl Selection {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn measure_candidates(
        &mut self,
        stream: &Stream,
        compressed: &super::NvFp4WeightOnly,
        tensor_core: &super::NvFp4WeightOnlyTensorCore,
        tensor_core_compatible: bool,
        marlin: Option<&mut super::MarlinNvFp4Bf16Linear>,
        marlin_compatible: [bool; 3],
        materialized: &mut super::super::super::super::Bf16Projection,
        weight: &super::NvFp4WeightOnlyWeight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<(Vec<(Execution, Duration)>, Duration)> {
        let (warmup, iterations) = self.tuner.iterations(self.request.tokens());
        let mut candidates = vec![Execution::Compressed];
        if tensor_core_compatible {
            candidates.push(Execution::TensorCore);
        }
        candidates.extend(
            super::marlin_candidates()
                .into_iter()
                .zip(marlin_compatible)
                .filter_map(|((execution, _), compatible)| compatible.then_some(execution)),
        );
        candidates.push(Execution::Materialized);
        let Self { context, scratch, pool, .. } = self;
        let eviction_source = pool.allocate_zeroed::<bf16>(stream, EVICTION_ELEMENTS)?;
        let mut eviction_target = pool.allocate::<bf16>(stream, EVICTION_ELEMENTS)?;
        let mut marlin = marlin;
        let mut run = |execution: Execution| -> Result<()> {
            let config = match execution {
                Execution::Compressed => {
                    return compressed.execute(stream, &mut launch(input, output, weight));
                },
                Execution::TensorCore => {
                    return tensor_core.execute(stream, &mut launch(input, output, weight));
                },
                Execution::Materialized => {
                    return materialized.execute(input, &weight.materialized, scratch);
                },
                Execution::MarlinN128K128 => mircuda::MarlinNvFp4ThreadConfig::N128K128,
                Execution::MarlinN128K64 => mircuda::MarlinNvFp4ThreadConfig::N128K64,
                Execution::MarlinN64K128 => mircuda::MarlinNvFp4ThreadConfig::N64K128,
            };
            marlin
                .as_deref_mut()
                .ok_or(crate::Error::InvalidExecutionPlan(
                    "validated dense Marlin candidate is unavailable",
                ))?
                .execute(input, output, config)
        };
        // Repeated runs leave a weight layout hot in accelerator cache and rank
        // layouts by that artifact; rotate candidates and evict between runs.
        let mut totals = vec![Duration::ZERO; candidates.len()];
        for round in 0..warmup.saturating_add(iterations) {
            for step in 0..candidates.len() {
                let index = (round as usize + step) % candidates.len();
                stream.copy_device_range(
                    &eviction_source,
                    0..EVICTION_ELEMENTS,
                    &mut eviction_target,
                    0,
                )?;
                let started = context.create_event(true)?;
                let completed = context.create_event(true)?;
                started.record(stream)?;
                run(candidates[index])?;
                completed.record(stream)?;
                completed.synchronize()?;
                if round >= warmup {
                    totals[index] = totals[index].saturating_add(Duration::from_secs_f32(
                        started.elapsed_ms(&completed)? / 1_000.0,
                    ));
                }
            }
        }
        let timings = candidates
            .into_iter()
            .zip(totals)
            .map(|(execution, total)| (execution, total / iterations.max(1)))
            .collect::<Vec<_>>();
        let executions = warmup.saturating_add(iterations);
        let tuning_elapsed = timings
            .iter()
            .map(|(_, duration)| *duration)
            .fold(Duration::ZERO, Duration::saturating_add)
            .saturating_mul(executions);
        Ok((timings, tuning_elapsed))
    }
}

fn launch<'a>(
    input: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
    weight: &'a super::NvFp4WeightOnlyWeight,
) -> NvFp4WeightOnlyLaunch<'a> {
    NvFp4WeightOnlyLaunch {
        input,
        weight: &weight.weight,
        block_scales: &weight.scales,
        global_scale: &weight.global_scale,
        output,
    }
}
