use mircuda::{DeviceBuffer, PinnedBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use super::CudaClampedRoutedModelSession;
use crate::{
    CudaBackend, CudaTensor, DeviceBatchSamplerBf16, Error, Result, RmsNormBf16,
    backend::output::{CudaBatchOutputHead, CudaOutputHeadTemplate},
    kernels::GatherRowsBf16,
};

pub struct ClampedRoutedBatchResult {
    pub(crate) selected: Vec<u32>,
    pub(crate) logits: Option<Vec<f32>>,
    pub(crate) vocab: usize,
}

pub(super) struct ClampedRoutedPackedOutput {
    gather: GatherRowsBf16,
    indices: DeviceBuffer<u32>,
    index_staging: PinnedBuffer<u32>,
    token_staging: PinnedBuffer<u32>,
    final_norm: RmsNormBf16,
    output_head: CudaBatchOutputHead,
    sampler: DeviceBatchSamplerBf16,
    selected: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
}

impl ClampedRoutedPackedOutput {
    fn new(
        backend: &CudaBackend,
        template: &CudaOutputHeadTemplate,
        rows: usize,
        hidden: usize,
        vocab: usize,
        epsilon: f32,
    ) -> Result<Self> {
        let activations = elements(rows, hidden, "clamped-routed output activation overflow")?;
        let logits = elements(rows, vocab, "clamped-routed output logits overflow")?;
        Ok(Self {
            gather: GatherRowsBf16::compile(&backend.inner.compiler, hidden)?,
            indices: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            index_staging: backend.inner.context.allocate_pinned(rows)?,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            final_norm: backend.prepare_rms_norm_bf16(rows, hidden, epsilon)?,
            output_head: CudaBatchOutputHead::new(backend, template, rows)?,
            sampler: backend.prepare_device_batch_sampler_bf16(vocab, rows)?,
            selected: backend.inner.pool.allocate(&backend.inner.stream, activations)?,
            normalized: backend.inner.pool.allocate(&backend.inner.stream, activations)?,
            logits: backend.inner.pool.allocate(&backend.inner.stream, logits)?,
        })
    }

    fn execute(
        &mut self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        source_rows: usize,
        rows: &[u32],
        policies: &[SamplingLogits],
        final_norm: &CudaTensor,
    ) -> Result<()> {
        if rows.len() != self.indices.len() || policies.len() != rows.len() {
            return Err(Error::InvalidDecoderKernel(
                "clamped-routed packed output rows differ from plan",
            ));
        }
        self.index_staging.copy_from_slice(rows)?;
        stream.copy_to_device(&mut self.index_staging, &mut self.indices)?;
        self.gather
            .execute(stream, source, &self.indices, &mut self.selected, source_rows)?;
        self.final_norm.execute(&self.selected, final_norm, &mut self.normalized)?;
        self.output_head.execute(&self.normalized, &mut self.logits)?;
        self.sampler.sample(&self.logits, policies)?;
        Ok(())
    }

    fn read_selected(&mut self, stream: &Stream) -> Result<Vec<u32>> {
        stream.copy_to_host(self.sampler.selected(), &mut self.token_staging)?;
        Ok(self.token_staging.to_vec()?)
    }
}

impl CudaClampedRoutedModelSession {
    pub(crate) fn finish_packed_device_rows(
        &mut self,
        rows: &[usize],
        tokens: usize,
        policies: &[SamplingLogits],
        read_logits: bool,
    ) -> Result<Option<ClampedRoutedBatchResult>> {
        let super::super::projection::ClampedRoutedOutputProjection::Native(template) =
            &self.template.output
        else {
            return Ok(None);
        };
        if rows.is_empty() || rows.len() != policies.len() || rows.iter().any(|row| *row >= tokens)
        {
            return Err(Error::InvalidDecoderKernel("invalid clamped-routed packed output batch"));
        }
        let indices = rows
            .iter()
            .map(|row| u32::try_from(*row))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let count = rows.len();
        let mut output = self.packed_outputs.remove(&count).map_or_else(
            || {
                ClampedRoutedPackedOutput::new(
                    &self.template.backend,
                    template,
                    count,
                    self.template.config.hidden,
                    self.template.config.vocab,
                    self.template.config.epsilon,
                )
            },
            Ok,
        )?;
        let hidden = if self.last_packed_decode == Some(tokens) {
            self.decode_batches
                .get(&tokens)
                .ok_or(Error::InvalidDecoderKernel("missing clamped-routed decode batch"))?
                .hidden()?
        } else {
            self.plans
                .get(&tokens)
                .ok_or(Error::InvalidDecoderKernel("missing clamped-routed packed output plan"))?
                .hidden()
        };
        let stream = &self.template.backend.inner.stream;
        let result =
            output.execute(stream, hidden, tokens, &indices, policies, &self.template.final_norm);
        self.packed_outputs.insert(count, output);
        result?;
        let output = self
            .packed_outputs
            .get_mut(&count)
            .ok_or(Error::InvalidDecoderKernel("missing clamped-routed packed output bucket"))?;
        let selected = output.read_selected(stream)?;
        let logits = read_logits
            .then(|| self.template.backend.read_logits(&output.logits))
            .transpose()?;
        Ok(Some(ClampedRoutedBatchResult {
            selected,
            logits,
            vocab: self.template.config.vocab,
        }))
    }
}

fn elements(rows: usize, columns: usize, message: &'static str) -> Result<usize> {
    rows.checked_mul(columns).ok_or(Error::InvalidDecoderKernel(message))
}
