use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::backend::SamplingLogits;

use super::super::graph::{bf16_tensor, checked};
use crate::{
    CudaBackend, CudaSharedRoutedModelTemplate, DeviceBatchSamplerBf16, Error, Result,
    backend::shared_routed::boundary::SharedRoutedOutputHead,
    kernels::{GatherRowsBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub(super) struct PackedOutput {
    backend: CudaBackend,
    gather: GatherRowsBf16,
    indices: DeviceBuffer<u32>,
    index_staging: PinnedBuffer<u32>,
    token_staging: PinnedBuffer<u32>,
    selected: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
    norm: ShiftedRmsNorm,
    head: SharedRoutedOutputHead,
    sampler: DeviceBatchSamplerBf16,
    norm_weight: crate::CudaTensor,
}

impl PackedOutput {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        let backend = &template.backend;
        let hidden = template.decoder.hidden_size;
        let vocab = template.decoder.vocab_size;
        let allocate_bf16 = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        Ok(Self {
            backend: backend.clone(),
            gather: GatherRowsBf16::compile(&backend.inner.compiler, hidden)?,
            indices: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            index_staging: backend.inner.context.allocate_pinned(rows)?,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            selected: allocate_bf16(checked(rows, hidden)?)?,
            normalized: allocate_bf16(checked(rows, hidden)?)?,
            logits: allocate_bf16(checked(rows, vocab)?)?,
            norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                rows,
                hidden,
                template.decoder.rms_norm_eps.to_string().parse()?,
                template.norm_shift,
            )?,
            head: template.prepare_output_head(rows)?,
            sampler: backend.prepare_device_batch_sampler_bf16(vocab, rows)?,
            norm_weight: template.final_norm.clone(),
        })
    }

    pub(super) fn execute(
        &mut self,
        hidden: &DeviceBuffer<bf16>,
        counts: &[usize],
        policies: &[SamplingLogits],
    ) -> Result<Vec<u32>> {
        if counts.len() != policies.len() {
            return Err(Error::InvalidDecoderKernel("packed sampling row mismatch"));
        }
        let source_rows = counts.iter().sum();
        let mut end = 0;
        let indices = counts
            .iter()
            .map(|count| {
                end += count;
                u32::try_from(end - 1)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.index_staging.copy_from_slice(&indices)?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.index_staging, &mut self.indices)?;
        self.gather
            .execute(stream, hidden, &self.indices, &mut self.selected, source_rows)?;
        self.norm.execute(
            stream,
            &self.selected,
            bf16_tensor(&self.norm_weight)?,
            &mut self.normalized,
        )?;
        self.head.execute(&self.normalized, &mut self.logits)?;
        self.sampler.sample(&self.logits, policies)?;
        stream.copy_to_host(self.sampler.selected(), &mut self.token_staging)?;
        self.token_staging.to_vec().map_err(Into::into)
    }
}
