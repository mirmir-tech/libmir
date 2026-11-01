//! The output head over a few gathered rows of a packed pass, sampled on
//! the device per row.

use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::backend::SamplingLogits;

use crate::{
    CudaBackend, CudaSharedRoutedModelTemplate, DeviceBatchSamplerBf16, Error, Result,
    backend::shared_routed::boundary::SharedRoutedOutputHead, kernels::GatherRowsBf16,
};

#[derive(Debug)]
pub(super) struct RowHead {
    backend: CudaBackend,
    rows: usize,
    gather: GatherRowsBf16,
    index_staging: PinnedBuffer<u32>,
    indices: DeviceBuffer<u32>,
    pub(super) selected: DeviceBuffer<bf16>,
    head: SharedRoutedOutputHead,
    pub(super) logits: DeviceBuffer<bf16>,
    pub(super) sampler: DeviceBatchSamplerBf16,
}

impl RowHead {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        let backend = &template.backend;
        let hidden = template.decoder.hidden_size;
        let vocab = template.decoder.vocab_size;
        let allocate = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        Ok(Self {
            backend: backend.clone(),
            rows,
            gather: GatherRowsBf16::compile(&backend.inner.compiler, hidden)?,
            index_staging: backend.inner.context.allocate_pinned(rows)?,
            indices: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            selected: allocate(rows * hidden)?,
            head: template.prepare_output_head(rows)?,
            logits: allocate(rows * vocab)?,
            sampler: backend.prepare_device_batch_sampler_bf16(vocab, rows)?,
        })
    }

    /// Gathers `indices` of the `source_rows` rows of `source` and computes
    /// their logits.
    pub(super) fn gather(
        &mut self,
        source: &DeviceBuffer<bf16>,
        source_rows: usize,
        indices: &[u32],
    ) -> Result<()> {
        if indices.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("gathered rows differ from the head"));
        }
        self.index_staging.copy_from_slice(indices)?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.index_staging, &mut self.indices)?;
        self.gather
            .execute(stream, source, &self.indices, &mut self.selected, source_rows)?;
        self.logits_of_selected()
    }

    /// Logits of the rows already in `selected`.
    pub(super) fn logits_of_selected(&mut self) -> Result<()> {
        self.head.execute(&self.selected, &mut self.logits)
    }

    /// Samples every row with its policy, on the device.
    pub(super) fn sample(&mut self, policies: &[SamplingLogits]) -> Result<&DeviceBuffer<u32>> {
        self.sampler.sample(&self.logits, policies)
    }
}
