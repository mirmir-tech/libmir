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
    vocab: usize,
}

impl CudaBackend {
    pub fn prepare_device_sampler_bf16(&self, vocab: usize) -> Result<DeviceSamplerBf16> {
        let workspace = Sampling::workspace_elements(vocab)?;
        Ok(DeviceSamplerBf16 {
            operation: Sampling::compile(&self.inner.compiler, vocab)?,
            stream: self.inner.stream.clone(),
            selected: self.inner.pool.allocate::<u32>(&self.inner.stream, 1)?,
            first: self.inner.pool.allocate::<u64>(&self.inner.stream, workspace)?,
            second: self.inner.pool.allocate::<u64>(&self.inner.stream, workspace)?,
            denominator: self.inner.pool.allocate::<f32>(&self.inner.stream, 1)?,
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
        SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } if vocab_size == vocab => {
            (k, 1.0, temperature, draw)
        },
        SamplingLogits::Sample {
            vocab_size,
            temperature,
            top_p,
            top_k,
            draw,
        } if vocab_size == vocab => (top_k, top_p, temperature, draw),
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
mod tests {
    use mircuda::{DeviceBuffer, bf16};
    use runtime::backend::SamplingLogits;

    use super::*;
    use crate::{CudaConfig, Result};

    #[test]
    fn samples_greedy_top_k_and_nucleus_on_device() -> Result<()> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let logits = copy(&backend, &[3.0, 2.0, 1.0, 0.0])?;
        let mut sampler = backend.prepare_device_sampler_bf16(4)?;
        assert_eq!(sample(&backend, &mut sampler, &logits, SamplingLogits::None)?, 0);
        assert_eq!(
            sample(
                &backend,
                &mut sampler,
                &logits,
                SamplingLogits::SampleTopK {
                    k: 2,
                    vocab_size: 4,
                    temperature: 1.0,
                    draw: 0.99,
                },
            )?,
            1
        );
        assert_eq!(
            sample(
                &backend,
                &mut sampler,
                &logits,
                SamplingLogits::Sample {
                    vocab_size: 4,
                    temperature: 1.0,
                    top_p: 0.6,
                    top_k: 4,
                    draw: 0.99,
                },
            )?,
            0
        );
        Ok(())
    }

    #[test]
    fn hierarchical_sampling_preserves_score_and_token_order() -> Result<()> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let mut values = vec![-100.0; 5_000];
        values[4_097] = 4.0;
        values[123] = 4.0;
        values[3_000] = 3.0;
        values[2_000] = 2.0;
        values[1_000] = 1.0;
        let logits = copy(&backend, &values)?;
        let mut sampler = backend.prepare_device_sampler_bf16(values.len())?;
        assert_eq!(sample(&backend, &mut sampler, &logits, SamplingLogits::None)?, 123);
        assert_eq!(
            sample(
                &backend,
                &mut sampler,
                &logits,
                SamplingLogits::SampleTopK {
                    k: 4,
                    vocab_size: values.len(),
                    temperature: 100.0,
                    draw: 0.99,
                },
            )?,
            2_000
        );
        Ok(())
    }

    fn sample(
        backend: &CudaBackend,
        sampler: &mut DeviceSamplerBf16,
        logits: &DeviceBuffer<bf16>,
        policy: SamplingLogits,
    ) -> Result<u32> {
        let selected = sampler.sample(logits, policy)?;
        let mut host = backend.inner.context.allocate_pinned::<u32>(1)?;
        backend.inner.stream.copy_to_host(selected, &mut host)?;
        Ok(host.to_vec()?[0])
    }

    fn copy(backend: &CudaBackend, values: &[f32]) -> Result<DeviceBuffer<bf16>> {
        let values = values.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
        let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
        host.copy_from_slice(&values)?;
        let mut device =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }
}
