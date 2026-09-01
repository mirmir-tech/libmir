use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use super::CudaBackend;
use crate::{
    Error, Result,
    kernels::{Sampling, SamplingSpec, SamplingWorkspace},
};

mod batch;

pub use batch::DeviceBatchSamplerBf16;

/// Prepared bounded sampler retaining its selected token on the device.
#[derive(Debug)]
pub struct DeviceSamplerBf16 {
    operation: Sampling,
    stream: Stream,
    selected: DeviceBuffer<u32>,
    first: DeviceBuffer<u64>,
    second: DeviceBuffer<u64>,
    denominator: DeviceBuffer<f32>,
    block_mass: DeviceBuffer<f32>,
    vocab: usize,
}

impl CudaBackend {
    pub fn prepare_device_sampler_bf16(&self, vocab: usize) -> Result<DeviceSamplerBf16> {
        let workspace = Sampling::workspace_elements(vocab)?;
        let block_mass = Sampling::block_mass_elements(vocab)?;
        Ok(DeviceSamplerBf16 {
            operation: Sampling::compile(&self.inner.compiler, vocab)?,
            stream: self.inner.stream.clone(),
            selected: self.inner.pool.allocate::<u32>(&self.inner.stream, 1)?,
            first: self.inner.pool.allocate::<u64>(&self.inner.stream, workspace)?,
            second: self.inner.pool.allocate::<u64>(&self.inner.stream, workspace)?,
            denominator: self.inner.pool.allocate::<f32>(&self.inner.stream, 1)?,
            block_mass: self.inner.pool.allocate::<f32>(&self.inner.stream, block_mass)?,
            vocab,
        })
    }
}

impl DeviceSamplerBf16 {
    /// Enqueues greedy or bounded top-k/top-p sampling without reading logits.
    pub fn sample(
        &mut self,
        logits: &DeviceBuffer<bf16>,
        policy: SamplingLogits,
    ) -> Result<&DeviceBuffer<u32>> {
        let spec = spec(self.vocab, policy)?;
        self.operation.execute(
            &self.stream,
            logits,
            &mut self.selected,
            SamplingWorkspace {
                first: &mut self.first,
                second: &mut self.second,
                denominator: &mut self.denominator,
                block_mass: &mut self.block_mass,
            },
            spec,
        )?;
        Ok(&self.selected)
    }

    #[must_use]
    pub const fn selected(&self) -> &DeviceBuffer<u32> {
        &self.selected
    }
}

fn spec(vocab: usize, policy: SamplingLogits) -> Result<SamplingSpec> {
    let (top_k, top_p, temperature, draw) = match policy {
        SamplingLogits::None => (1, 1.0, 1.0, 0.0),
        SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } if vocab_size <= vocab => {
            (k, 1.0, temperature, draw)
        },
        SamplingLogits::Sample {
            vocab_size,
            temperature,
            top_p,
            top_k,
            draw,
        } if vocab_size <= vocab && (top_k > 0 || top_p >= 1.0) => {
            (top_k, top_p, temperature, draw)
        },
        SamplingLogits::Full
        | SamplingLogits::TopK { .. }
        | SamplingLogits::SampleTopK { .. }
        | SamplingLogits::Sample { .. } => {
            return Err(Error::InvalidSampling(
                "policy requires host history or another vocabulary".into(),
            ));
        },
    };
    Ok(SamplingSpec { vocab, top_k, top_p, temperature, draw })
}

#[cfg(test)]
mod tests;
