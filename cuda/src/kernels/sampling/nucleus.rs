use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{Sampling, SamplingSpec};
use crate::{Result, kernels::geometry::narrow};

/// Threads of the single-block nucleus kernel; matches `kNucleusThreads`.
const NUCLEUS_THREADS: u32 = 1_024;

pub(super) struct NucleusSampling<'a> {
    pub stream: &'a Stream,
    pub logits: &'a DeviceBuffer<bf16>,
    pub candidates: &'a DeviceBuffer<u64>,
    pub output: &'a mut DeviceBuffer<u32>,
    pub spec: SamplingSpec,
    pub row: u32,
    pub stride: u32,
}

impl Sampling {
    /// Top-p over the whole vocabulary, without a top-k bound.
    pub(super) fn execute_nucleus(&self, request: NucleusSampling<'_>) -> Result<()> {
        let NucleusSampling {
            stream,
            logits,
            candidates,
            output,
            spec,
            row,
            stride,
        } = request;
        Ok(self.nucleus.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (NUCLEUS_THREADS, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                logits,
                candidates,
                output,
                spec.top_p,
                spec.temperature,
                spec.draw,
                narrow(spec.vocab)?,
                narrow(self.vocab)?,
                row,
                stride,
            ),
        )?)
    }
}
