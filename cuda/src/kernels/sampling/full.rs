use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{Sampling, SamplingSpec, THREADS, validation::launch};
use crate::{Result, kernels::geometry::narrow};

pub(super) struct FullSampling<'a> {
    pub stream: &'a Stream,
    pub logits: &'a DeviceBuffer<bf16>,
    pub candidates: &'a DeviceBuffer<u64>,
    pub block_mass: &'a mut DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<u32>,
    pub spec: SamplingSpec,
    pub row: u32,
    pub stride: u32,
    pub block_count: usize,
}

impl Sampling {
    pub(super) fn execute_full(&self, request: FullSampling<'_>) -> Result<()> {
        let FullSampling {
            stream,
            logits,
            candidates,
            block_mass,
            output,
            spec,
            row,
            stride,
            block_count,
        } = request;
        self.full_mass.launch(
            stream,
            launch(block_count)?,
            (
                logits,
                candidates,
                &mut *block_mass,
                narrow(spec.vocab)?,
                spec.temperature,
                narrow(self.vocab)?,
                row,
                stride,
            ),
        )?;
        Ok(self.full_finalize.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (THREADS, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                logits,
                candidates,
                &*block_mass,
                output,
                spec.temperature,
                spec.draw,
                narrow(spec.vocab)?,
                narrow(self.vocab)?,
                row,
                stride,
                narrow(block_count)?,
            ),
        )?)
    }
}
