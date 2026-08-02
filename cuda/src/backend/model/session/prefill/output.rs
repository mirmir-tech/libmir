use mircuda::{DeviceBuffer, PinnedBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use super::CudaMoeModelSession;
use crate::{
    CudaBackend, CudaTensor, DeviceBatchSamplerBf16, Error, Result, RmsNormBf16,
    backend::model::boundary::{ModelBatchOutputHead, ModelOutputHeadTemplate},
    kernels::GatherRowsBf16,
};

pub(in crate::backend::model::session) struct PackedPrefillOutput {
    gather: GatherRowsBf16,
    indices: DeviceBuffer<u32>,
    index_staging: PinnedBuffer<u32>,
    token_staging: PinnedBuffer<u32>,
    final_norm: RmsNormBf16,
    output_head: ModelBatchOutputHead,
    sampler: DeviceBatchSamplerBf16,
    selected: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
    logit_softcap: Option<crate::kernels::LogitSoftcap>,
}

impl PackedPrefillOutput {
    fn new(
        backend: &CudaBackend,
        template: &ModelOutputHeadTemplate,
        rows: usize,
        hidden: usize,
        vocab: usize,
        epsilon: f32,
        logit_softcap: Option<f32>,
    ) -> Result<Self> {
        let activation_elements = elements(rows, hidden, "prefill output activation overflow")?;
        let logits_elements = elements(rows, vocab, "prefill output logits overflow")?;
        Ok(Self {
            gather: GatherRowsBf16::compile(&backend.inner.compiler, hidden)?,
            indices: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            index_staging: backend.inner.context.allocate_pinned(rows)?,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            final_norm: backend.prepare_rms_norm_bf16(rows, hidden, epsilon)?,
            output_head: template.instantiate_batch(backend, rows)?,
            sampler: backend.prepare_device_batch_sampler_bf16(vocab, rows)?,
            selected: backend.inner.pool.allocate(&backend.inner.stream, activation_elements)?,
            normalized: backend.inner.pool.allocate(&backend.inner.stream, activation_elements)?,
            logits: backend.inner.pool.allocate(&backend.inner.stream, logits_elements)?,
            logit_softcap: logit_softcap
                .map(|cap| {
                    crate::kernels::LogitSoftcap::compile(
                        &backend.inner.compiler,
                        logits_elements,
                        cap,
                    )
                })
                .transpose()?,
        })
    }

    fn execute(
        &mut self,
        stream: &Stream,
        source: &DeviceBuffer<bf16>,
        source_rows: usize,
        rows: &[u32],
        policies: &[SamplingLogits],
        final_norm_weight: &CudaTensor,
    ) -> Result<()> {
        if rows.len() != self.indices.len() || policies.len() != rows.len() {
            return Err(Error::InvalidDecoderKernel("packed output rows differ from bucket"));
        }
        self.index_staging.copy_from_slice(rows)?;
        stream.copy_to_device(&mut self.index_staging, &mut self.indices)?;
        self.gather
            .execute(stream, source, &self.indices, &mut self.selected, source_rows)?;
        self.final_norm
            .execute(&self.selected, final_norm_weight, &mut self.normalized)?;
        self.output_head.execute(&self.normalized, &mut self.logits)?;
        if let Some(softcap) = &self.logit_softcap {
            softcap.execute(stream, &mut self.logits)?;
        }
        self.sampler.sample(&self.logits, policies)?;
        Ok(())
    }

    fn read_sampled(&mut self, stream: &Stream) -> Result<Vec<u32>> {
        stream.copy_to_host(self.sampler.selected(), &mut self.token_staging)?;
        Ok(self.token_staging.to_vec()?)
    }
}

impl CudaMoeModelSession {
    pub(crate) fn prepare_packed_output_buckets(&mut self, maximum: usize) -> Result<()> {
        for rows in bucket_sizes(maximum) {
            if self.packed_outputs.contains_key(&rows) {
                continue;
            }
            let output = PackedPrefillOutput::new(
                &self.backend,
                &self.output_template,
                rows,
                self.hidden_size,
                self.logits.len(),
                self.final_norm.epsilon(),
                self.logit_softcap_cap,
            )?;
            self.packed_outputs.insert(rows, output);
        }
        Ok(())
    }

    pub(crate) fn finish_packed_prefill_rows(
        &mut self,
        rows: &[usize],
        tokens: usize,
        policies: &[SamplingLogits],
    ) -> Result<Vec<u32>> {
        if rows.is_empty() || rows.len() != policies.len() {
            return Err(Error::InvalidDecoderKernel("invalid packed output batch"));
        }
        if rows.iter().any(|row| *row >= tokens) {
            return Err(Error::InvalidDecoderKernel("packed output row exceeds activations"));
        }
        let rows = rows
            .iter()
            .map(|row| u32::try_from(*row))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let count = rows.len();
        let bucket = self
            .packed_outputs
            .keys()
            .copied()
            .filter(|rows| *rows >= count)
            .min()
            .unwrap_or(count);
        let mut output = if let Some(output) = self.packed_outputs.remove(&bucket) {
            output
        } else {
            PackedPrefillOutput::new(
                &self.backend,
                &self.output_template,
                bucket,
                self.hidden_size,
                self.logits.len(),
                self.final_norm.epsilon(),
                self.logit_softcap_cap,
            )?
        };
        let mut padded_rows = rows;
        let last_row = *padded_rows
            .last()
            .ok_or(Error::InvalidDecoderKernel("packed output has no final row"))?;
        padded_rows.resize(bucket, last_row);
        let mut padded_policies = policies.to_vec();
        padded_policies.resize(bucket, SamplingLogits::None);
        let source = if self.layers.len().is_multiple_of(2) {
            &self.prefill_first
        } else {
            &self.prefill_second
        };
        let result = output.execute(
            &self.stream,
            source,
            tokens,
            &padded_rows,
            &padded_policies,
            &self.final_norm_weight,
        );
        self.packed_outputs.insert(bucket, output);
        result?;
        let mut tokens = self
            .packed_outputs
            .get_mut(&bucket)
            .ok_or(Error::InvalidDecoderKernel("missing packed output bucket"))?
            .read_sampled(&self.stream)?;
        tokens.truncate(count);
        Ok(tokens)
    }
}

fn elements(rows: usize, columns: usize, message: &'static str) -> Result<usize> {
    rows.checked_mul(columns).ok_or(Error::InvalidDecoderKernel(message))
}

fn bucket_sizes(maximum: usize) -> Vec<usize> {
    let mut sizes = std::iter::successors(Some(2_usize), |size| size.checked_mul(2))
        .take_while(|size| *size <= maximum)
        .collect::<Vec<_>>();
    sizes.extend([5, 10, maximum].into_iter().filter(|size| (2..=maximum).contains(size)));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

#[cfg(test)]
mod tests {
    use super::bucket_sizes;

    #[test]
    fn prepares_canonical_output_buckets() {
        assert_eq!(bucket_sizes(10), [2, 4, 5, 8, 10]);
        assert_eq!(bucket_sizes(3), [2, 3]);
        assert!(bucket_sizes(1).is_empty());
    }
}
