use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{Sampling, SamplingSpec, validation::launch};
use crate::{Result, kernels::geometry::narrow};

pub(super) struct BoundedSampling<'a> {
    pub stream: &'a Stream,
    pub logits: &'a DeviceBuffer<bf16>,
    pub candidates: &'a DeviceBuffer<u64>,
    pub denominator: &'a mut DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<u32>,
    pub spec: SamplingSpec,
    pub row: u32,
    pub stride: u32,
    pub top_k: u32,
}

impl Sampling {
    pub(super) fn execute_bounded(&self, request: BoundedSampling<'_>) -> Result<()> {
        let BoundedSampling {
            stream,
            logits,
            candidates,
            denominator,
            output,
            spec,
            row,
            stride,
            top_k,
        } = request;
        if spec.top_k > 1 {
            let mass_blocks = super::validation::blocks(spec.vocab)?.min(1_024);
            self.mass.launch(
                stream,
                launch(mass_blocks)?,
                (
                    logits,
                    candidates,
                    &mut *denominator,
                    narrow(spec.vocab)?,
                    narrow(self.vocab)?,
                    row,
                    stride,
                ),
            )?;
        }
        Ok(self.finalize.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (1, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                logits,
                candidates,
                &*denominator,
                output,
                top_k,
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
