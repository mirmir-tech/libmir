//! Buffers and layers of one verification row count, at fixed addresses.

use mircuda::{DeviceBuffer, PinnedBuffer, Stream, bf16};
use runtime::kv::BlockTable;

use super::super::{
    super::{
        CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate,
        boundary::SharedRoutedEmbedding,
    },
    graph::{bf16_tensor, checked},
    layer::SharedRoutedBatchLayer,
    workspace,
};
use crate::{
    Error, ExecutionPhase, PagedDecodeBatch, Result,
    backend::scratch_pool::SharedScratch,
    kernels::{BatchedSplitAttentionWorkspace, ShiftedRmsNorm},
};

#[derive(Debug)]
pub(super) struct VerifyResources {
    stream: Stream,
    pub(super) rows: usize,
    _workspace: SharedScratch<BatchedSplitAttentionWorkspace>,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    paging: PagedDecodeBatch,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    embedding: SharedRoutedEmbedding,
    layers: Vec<SharedRoutedBatchLayer>,
    final_norm: ShiftedRmsNorm,
    final_norm_weight: crate::CudaTensor,
    /// The final norm of every row: the prediction module's and the output
    /// head's input.
    pub(super) normalized: DeviceBuffer<bf16>,
}

impl VerifyResources {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        let backend = &template.backend;
        let elements = checked(rows, template.decoder.hidden_size)?;
        let allocate = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        let (workspace_handle, attention_workspace) =
            workspace::attention_workspace(template, rows)?;
        Ok(Self {
            stream: backend.inner.stream.clone(),
            rows,
            _workspace: workspace_handle,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            paging: backend.prepare_paged_decode_batch(
                template.cache_spec()?,
                template.max_sequence_blocks,
                rows,
            )?,
            first: allocate(elements)?,
            second: allocate(elements)?,
            embedding: template.prepare_embedding()?,
            layers: template
                .layers
                .iter()
                .map(|layer| {
                    SharedRoutedBatchLayer::new(
                        layer,
                        rows,
                        rows,
                        ExecutionPhase::Decode,
                        Some(attention_workspace.clone()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                rows,
                template.decoder.hidden_size,
                template.decoder.rms_norm_eps.to_string().parse()?,
                template.norm_shift,
            )?,
            final_norm_weight: template.final_norm.clone(),
            normalized: allocate(elements)?,
        })
    }

    /// Host-side preparation: the fed token, the session's device drafts
    /// after it, and one decode row per position of `table`'s session.
    pub(super) fn prepare(
        &mut self,
        session: &CudaSharedRoutedModelSession,
        token: u32,
        table: &BlockTable,
    ) -> Result<()> {
        session.embedding.validate_token(token)?;
        let drafts = session
            .drafts
            .as_ref()
            .filter(|_| session.drafted + 1 >= self.rows)
            .ok_or(Error::InvalidDecoderKernel("verification rows exceed the drafts"))?;
        self.token_staging.copy_from_slice(&vec![token; self.rows])?;
        self.stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        self.stream
            .copy_device_range(drafts, 0..self.rows - 1, &mut self.token_ids, 1)?;
        let tables = (1..=self.rows)
            .map(|rows| {
                let mut row = table.clone();
                row.set_token_len(session.position + rows);
                row
            })
            .collect::<Vec<_>>();
        self.paging.prepare(&tables.iter().collect::<Vec<_>>())?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.prepare_verify(session, index, &self.paging)?;
        }
        Ok(())
    }

    /// Device work of the pass; no allocation or host synchronization.
    pub(super) fn execute(&mut self) -> Result<()> {
        self.embedding.execute_batch(&self.token_ids, self.rows, &mut self.first)?;
        for index in 0..self.layers.len() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            self.layers[index].execute_prepared_verify(input, &self.paging, output)?;
        }
        let hidden = if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        };
        self.final_norm.execute(
            &self.stream,
            hidden,
            bf16_tensor(&self.final_norm_weight)?,
            &mut self.normalized,
        )
    }

    pub(super) fn commit(
        &mut self,
        session: &mut CudaSharedRoutedModelSession,
        kept: usize,
    ) -> Result<()> {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.commit_verify(session, index, kept)?;
        }
        Ok(())
    }

    /// The pages the attention layers address, and their split partitions:
    /// a graph replays both.
    pub(super) fn capture_key(&self, block_count: u32) -> (u32, usize) {
        let partitions = self
            .layers
            .iter()
            .map(|layer| layer.capture_partitions(&self.paging))
            .max()
            .unwrap_or_default();
        (block_count, partitions)
    }
}
