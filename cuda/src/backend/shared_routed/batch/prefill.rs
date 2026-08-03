use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};

use super::{
    graph::{bf16_tensor, checked},
    layer::SharedRoutedBatchLayer,
    states::{full_states, linear_states},
};
use crate::{
    CudaBackend, CudaSharedRoutedModelSession, DeviceBatchSamplerBf16, Error, PagedPrefillBatch,
    Result,
    backend::shared_routed::{
        CudaSharedRoutedModelTemplate,
        boundary::{SharedRoutedEmbedding, SharedRoutedOutputHead},
    },
    kernels::{GatherRowsBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub(crate) struct CudaSharedRoutedPrefillBatch {
    backend: CudaBackend,
    rows: usize,
    row_tokens: usize,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    embedding: SharedRoutedEmbedding,
    layers: Vec<SharedRoutedBatchLayer>,
    paging: PagedPrefillBatch,
    output: PackedOutput,
}

impl CudaSharedRoutedPrefillBatch {
    pub(crate) fn new(
        template: &CudaSharedRoutedModelTemplate,
        rows: usize,
        row_tokens: usize,
    ) -> Result<Self> {
        if rows < 2 || row_tokens == 0 {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill batch is empty"));
        }
        let backend = &template.backend;
        let tokens = checked(rows, row_tokens)?;
        let hidden = template.decoder.hidden_size;
        let elements = checked(tokens, hidden)?;
        let allocate_bf16 = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        let layers = template
            .layers
            .iter()
            .map(|layer| SharedRoutedBatchLayer::new(layer, tokens))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backend: backend.clone(),
            rows,
            row_tokens,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            position_staging: backend.inner.context.allocate_pinned(checked(3, tokens)?)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, checked(3, tokens)?)?,
            first: allocate_bf16(elements)?,
            second: allocate_bf16(elements)?,
            embedding: template.prepare_embedding()?,
            layers,
            paging: backend.prepare_paged_prefill_batch(
                template.cache_spec()?,
                template.max_sequence_blocks,
                rows,
                tokens,
            )?,
            output: PackedOutput::new(template, rows)?,
        })
    }

    pub(crate) fn execute(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        policies: Option<&[SamplingLogits]>,
    ) -> Result<Option<Vec<u32>>> {
        self.validate(sessions, tokens, tables, starts)?;
        let counts = vec![self.row_tokens; self.rows];
        self.paging.prepare(tables, starts, &counts)?;
        self.token_staging.copy_from_slice(tokens)?;
        self.position_staging
            .copy_from_slice(&self.paging_positions(starts).repeat(3))?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        self.embedding.execute_batch(&self.token_ids, tokens.len(), &mut self.first)?;
        for index in 0..self.layers.len() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            match &mut self.layers[index] {
                SharedRoutedBatchLayer::Linear(layer) => {
                    layer.execute_packed(input, &mut linear_states(sessions, index)?, output)?;
                },
                SharedRoutedBatchLayer::Full(layer) => {
                    layer.execute_packed_prefill(
                        input,
                        &self.positions,
                        &mut full_states(sessions, index)?,
                        &self.paging,
                        output,
                    )?;
                },
            }
        }
        for session in sessions.iter_mut() {
            session.position = session
                .position
                .checked_add(self.row_tokens)
                .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        }
        let hidden = if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        };
        policies
            .map(|policies| self.output.execute(hidden, tokens.len(), policies))
            .transpose()
    }

    fn paging_positions(&self, starts: &[usize]) -> Vec<u32> {
        starts
            .iter()
            .flat_map(|start| (*start..*start + self.row_tokens).map(|value| value as u32))
            .collect()
    }

    fn validate(
        &self,
        sessions: &[&mut CudaSharedRoutedModelSession],
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
    ) -> Result<()> {
        if sessions.len() != self.rows
            || tables.len() != self.rows
            || starts.len() != self.rows
            || tokens.len() != self.rows * self.row_tokens
        {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill batch shape mismatch"));
        }
        for (session, start) in sessions.iter().zip(starts) {
            if session.position != *start {
                return Err(Error::InvalidPagedKv("shared-routed prefill position mismatch"));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PackedOutput {
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
    fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
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

    fn execute(
        &mut self,
        hidden: &DeviceBuffer<bf16>,
        source_rows: usize,
        policies: &[SamplingLogits],
    ) -> Result<Vec<u32>> {
        let row_tokens = source_rows / policies.len();
        let indices = (1..=policies.len())
            .map(|row| u32::try_from(row * row_tokens - 1))
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
