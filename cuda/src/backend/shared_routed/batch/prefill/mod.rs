mod output;
use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use output::PackedOutput;
use runtime::{backend::SamplingLogits, kv::BlockTable};

use super::{
    graph::checked,
    layer::SharedRoutedBatchLayer,
    states::{full_states, linear_states},
};
use crate::{
    CudaBackend, CudaSharedRoutedModelSession, Error, ExecutionPhase, PagedPrefillBatch, Result,
    backend::shared_routed::{CudaSharedRoutedModelTemplate, boundary::SharedRoutedEmbedding},
};

#[derive(Debug)]
pub struct CudaSharedRoutedPrefillBatch {
    backend: CudaBackend,
    rows: usize,
    counts: Vec<usize>,
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
        Self::new_ragged(template, &vec![row_tokens; rows])
    }

    pub(crate) fn new_ragged(
        template: &CudaSharedRoutedModelTemplate,
        counts: &[usize],
    ) -> Result<Self> {
        let tokens = counts
            .iter()
            .try_fold(0_usize, |sum, count| sum.checked_add(*count))
            .ok_or(Error::InvalidDecoderKernel("shared-routed prefill size overflow"))?;
        Self::with_capacity(template, counts, tokens)
    }

    pub(crate) fn with_capacity(
        template: &CudaSharedRoutedModelTemplate,
        counts: &[usize],
        tokens: usize,
    ) -> Result<Self> {
        let total = counts.iter().try_fold(0_usize, |sum, count| sum.checked_add(*count));
        if counts.is_empty() || counts.contains(&0) || total.is_none_or(|total| total > tokens) {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill batch is empty"));
        }
        let rows = counts.len();
        let backend = &template.backend;
        let hidden = template.decoder.hidden_size;
        let elements = checked(tokens, hidden)?;
        let allocate_bf16 = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        let layers = template
            .layers
            .iter()
            .map(|layer| SharedRoutedBatchLayer::new(layer, tokens, ExecutionPhase::Prefill, None))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backend: backend.clone(),
            rows,
            counts: counts.to_vec(),
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

    pub(crate) fn reconfigure(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
        counts: &[usize],
    ) -> Result<()> {
        let total = counts.iter().try_fold(0_usize, |sum, count| sum.checked_add(*count));
        if counts.is_empty()
            || counts.contains(&0)
            || total.is_none_or(|total| total > self.token_ids.len())
        {
            return Err(Error::InvalidDecoderKernel(
                "ragged reconfiguration exceeds token capacity",
            ));
        }
        if counts.len() != self.rows {
            let paging = self.backend.prepare_paged_prefill_batch(
                template.cache_spec()?,
                template.max_sequence_blocks,
                counts.len(),
                self.token_ids.len(),
            )?;
            let output = PackedOutput::new(template, counts.len())?;
            self.paging = paging;
            self.output = output;
            self.rows = counts.len();
        }
        self.counts = counts.to_vec();
        Ok(())
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
        if policies.is_some_and(|policies| policies.len() != self.rows) {
            return Err(Error::InvalidDecoderKernel("packed sampling row mismatch"));
        }
        self.paging.prepare(tables, starts, &self.counts)?;
        // Padding participates only in row-independent projections. Attention,
        // recurrence, K/V writes and session positions use the logical counts.
        let mut padded_tokens = vec![0; self.token_ids.len()];
        padded_tokens[..tokens.len()].copy_from_slice(tokens);
        self.token_staging.copy_from_slice(&padded_tokens)?;
        let mut positions = self.paging_positions(starts)?;
        positions.resize(self.token_ids.len(), 0);
        self.position_staging.copy_from_slice(&positions.repeat(3))?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        self.embedding
            .execute_batch(&self.token_ids, self.token_ids.len(), &mut self.first)?;
        for index in 0..self.layers.len() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            match &mut self.layers[index] {
                SharedRoutedBatchLayer::Linear(layer) => {
                    let mut states = linear_states(sessions, index)?;
                    // A row with an armed checkpoint splits its recurrence,
                    // which only the ragged path does.
                    if self.rows > 1
                        && tokens.len() == self.token_ids.len()
                        && self.counts.iter().all(|count| *count == self.counts[0])
                        && states.iter().all(|state| state.armed_checkpoint().is_none())
                    {
                        layer.execute_packed(input, &mut states, output)?;
                    } else {
                        layer.execute_ragged(input, &mut states, &self.counts, output)?;
                    }
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
        for (session, count) in sessions.iter_mut().zip(&self.counts) {
            session.position = session
                .position
                .checked_add(*count)
                .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        }
        let hidden = if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        };
        policies
            .map(|policies| self.output.execute(hidden, &self.counts, policies))
            .transpose()
    }

    fn paging_positions(&self, starts: &[usize]) -> Result<Vec<u32>> {
        starts
            .iter()
            .zip(&self.counts)
            .flat_map(|(start, count)| *start..*start + *count)
            .map(|value| {
                u32::try_from(value)
                    .map_err(|_| Error::InvalidPagedKv("shared-routed position exceeds u32"))
            })
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
            || tokens.len() != self.counts.iter().sum::<usize>()
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
