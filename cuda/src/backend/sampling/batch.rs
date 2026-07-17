use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use super::{CudaBackend, spec};
use crate::{
    Error, Result,
    kernels::{Sampling, SamplingWorkspace},
};

/// Fixed-row device sampler with independent policies and no logits readback.
#[derive(Debug)]
pub struct DeviceBatchSamplerBf16 {
    operation: Sampling,
    stream: Stream,
    selected: DeviceBuffer<u32>,
    first: DeviceBuffer<u64>,
    second: DeviceBuffer<u64>,
    denominator: DeviceBuffer<f32>,
    vocab: usize,
    rows: usize,
}

impl CudaBackend {
    pub fn prepare_device_batch_sampler_bf16(
        &self,
        vocab: usize,
        rows: usize,
    ) -> Result<DeviceBatchSamplerBf16> {
        if rows == 0 {
            return Err(Error::InvalidSampling("sampling batch is empty".into()));
        }
        let per_row = Sampling::workspace_elements(vocab)?;
        let workspace = per_row
            .checked_mul(rows)
            .ok_or_else(|| Error::InvalidSampling("sampling batch workspace overflow".into()))?;
        Ok(DeviceBatchSamplerBf16 {
            operation: Sampling::compile(&self.inner.compiler, vocab)?,
            stream: self.inner.stream.clone(),
            selected: self.inner.pool.allocate(&self.inner.stream, rows)?,
            first: self.inner.pool.allocate(&self.inner.stream, workspace)?,
            second: self.inner.pool.allocate(&self.inner.stream, workspace)?,
            denominator: self.inner.pool.allocate(&self.inner.stream, rows)?,
            vocab,
            rows,
        })
    }
}

impl DeviceBatchSamplerBf16 {
    pub fn sample(
        &mut self,
        logits: &DeviceBuffer<bf16>,
        policies: &[SamplingLogits],
    ) -> Result<&DeviceBuffer<u32>> {
        if policies.len() != self.rows {
            return Err(Error::InvalidSampling("sampling policies differ from batch".into()));
        }
        for (row, policy) in policies.iter().copied().enumerate() {
            self.operation.execute_row(
                &self.stream,
                logits,
                &mut self.selected,
                SamplingWorkspace {
                    first: &mut self.first,
                    second: &mut self.second,
                    denominator: &mut self.denominator,
                },
                spec(self.vocab, policy)?,
                row,
            )?;
        }
        Ok(&self.selected)
    }

    #[must_use]
    pub const fn selected(&self) -> &DeviceBuffer<u32> {
        &self.selected
    }
}

#[cfg(test)]
mod tests {
    use mircuda::bf16;
    use runtime::backend::SamplingLogits;

    use super::*;
    use crate::{CudaConfig, Result};

    #[test]
    fn samples_independent_rows_without_logits_readback() -> Result<()> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let values = [3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0].map(bf16::from_f32);
        let mut host = backend.inner.context.allocate_pinned(values.len())?;
        host.copy_from_slice(&values)?;
        let mut logits = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut logits)?;
        let mut sampler = backend.prepare_device_batch_sampler_bf16(4, 2)?;
        let selected = sampler.sample(
            &logits,
            &[
                SamplingLogits::None,
                SamplingLogits::SampleTopK {
                    k: 2,
                    vocab_size: 4,
                    temperature: 1.0,
                    draw: 0.99,
                },
            ],
        )?;
        let mut actual = backend.inner.context.allocate_pinned(2)?;
        backend.inner.stream.copy_to_host(selected, &mut actual)?;
        assert_eq!(actual.to_vec()?, [0, 3]);
        Ok(())
    }
}
