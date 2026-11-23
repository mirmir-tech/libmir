//! Packed execution: prefill of concatenated sequences and decode of one
//! token per session, over the sessions' shared latent pages.

use mircuda::{DeviceBuffer, bf16};

use super::{CudaLatentAttentionExecution, Placement};
use crate::{
    Error, LatentAttentionState, PagedDecodeBatch, PagedPrefillBatch, Result,
    kernels::{LatentExpandedMetadata, LatentExpandedRow, LatentExpandedRows},
};

impl CudaLatentAttentionExecution {
    /// Prefills several sequences whose rows are concatenated in `input`.
    pub(crate) fn execute_packed_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &LatentAttentionState,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // Sequences prefill expanded when their contexts are the rows the
        // batch's attention metadata describes.
        let rows = batch
            .rows()
            .iter()
            .map(|row| LatentExpandedRow { start: row.start(), tokens: row.tokens() })
            .collect::<Vec<_>>();
        let described = batch
            .rows()
            .iter()
            .all(|row| row.table().token_len() == row.start() + row.tokens());
        let placement = Placement {
            positions: batch.positions(),
            slot_mapping: batch.slot_mapping(false),
            sequences: batch.request_indices(),
            tables: batch.tables(),
            table_stride: batch.max_blocks(),
            expanded: (described && rows.len() == batch.active()).then_some(
                LatentExpandedRows::Packed {
                    rows: &rows,
                    metadata: LatentExpandedMetadata {
                        query_starts: batch.query_starts(),
                        contexts: batch.token_counts(),
                        context_starts: batch.context_starts(),
                    },
                },
            ),
        };
        let mut pages = state.cache.clone();
        self.run(input, &mut pages, &placement, output)
    }

    /// Host-side preparation of a packed decode step: binds the sessions'
    /// shared pages.
    pub(crate) fn prepare_packed(
        &mut self,
        states: &[&mut LatentAttentionState],
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        let [first, ..] = states else {
            return Err(Error::InvalidDecoderKernel("latent packed step has no rows"));
        };
        self.prepare_rows(first, paging)
    }

    /// Host-side preparation of rows that all use the pages of `state`.
    pub(crate) fn prepare_rows(
        &mut self,
        state: &LatentAttentionState,
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        if paging.active() != self.absorption.tokens() {
            return Err(Error::InvalidDecoderKernel("latent packed rows differ from the plan"));
        }
        self.pages = Some(state.cache.clone());
        Ok(())
    }

    /// Device work of a packed decode step; no allocation or host sync.
    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut pages = self
            .pages
            .take()
            .ok_or(Error::InvalidDecoderKernel("latent packed step was not prepared"))?;
        let placement = Placement {
            positions,
            slot_mapping: paging.slot_mapping(),
            // A decode row's sequence is its row: `query_starts` is 0, 1, ….
            sequences: paging.query_starts(),
            tables: paging.tables(),
            table_stride: paging.max_blocks(),
            expanded: None,
        };
        let result = self.run(input, &mut pages, &placement, output);
        self.pages = Some(pages);
        result
    }
}
