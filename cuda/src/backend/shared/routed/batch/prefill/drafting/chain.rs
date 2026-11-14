//! Drafting for several sessions at once: each further draft runs the
//! prediction module on every session's last output and draft, one position
//! later, in one packed pass.

use mircuda::DeviceBuffer;
use runtime::{backend::SamplingLogits, kv::BlockTable};

use super::head::RowHead;
use crate::{
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Error, PagedPrefillBatch, Result,
    backend::shared::routed::{boundary::SharedRoutedEmbedding, prediction::PredictionExecution},
};

/// A session that drafts: its index in the step and the position of the
/// module output its first draft comes from.
pub(super) struct ChainRow<'a> {
    pub(super) session: usize,
    pub(super) position: usize,
    pub(super) table: &'a BlockTable,
}

#[derive(Debug)]
pub(super) struct DraftChain {
    rows: usize,
    head: RowHead,
    prediction: PredictionExecution,
    paging: PagedPrefillBatch,
}

impl DraftChain {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        let weights = template
            .prediction
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("model has no prediction module loaded"))?;
        Ok(Self {
            rows,
            head: RowHead::new(template, rows)?,
            prediction: PredictionExecution::new(template, weights, rows)?,
            paging: template.backend.prepare_paged_prefill_batch(
                weights.layer().storage(template.cache),
                template.max_sequence_blocks,
                rows,
                rows,
            )?,
        })
    }

    /// Writes `count` drafts into every member's device drafts, starting
    /// from rows `sources` of the `source_rows` module outputs in `source`.
    pub(super) fn run(
        &mut self,
        embedding: &SharedRoutedEmbedding,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        members: &[ChainRow<'_>],
        (source, source_rows, sources): (&DeviceBuffer<mircuda::bf16>, usize, &[u32]),
        count: usize,
    ) -> Result<()> {
        if members.len() != self.rows || sources.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("draft chain rows differ from the plan"));
        }
        let stream = self.prediction.backend.inner.stream.clone();
        let greedy = vec![SamplingLogits::None; self.rows];
        self.head.gather(source, source_rows, sources)?;
        for index in 0..count {
            let tokens = self.head.sampler.sample(&self.head.logits, &greedy)?;
            for (row, member) in members.iter().enumerate() {
                let drafts = sessions[member.session].draft_buffer(count)?;
                stream.copy_device_range(tokens, row..row + 1, drafts, index)?;
            }
            if index + 1 == count {
                break;
            }
            stream.copy_device_range(tokens, 0..self.rows, &mut self.prediction.token_ids, 0)?;
            let selected = &self.head.selected;
            stream.copy_device_range(
                selected,
                0..selected.len(),
                &mut self.prediction.hidden,
                0,
            )?;
            let at = members.iter().map(|member| member.position + index + 1).collect::<Vec<_>>();
            let tables = members
                .iter()
                .zip(&at)
                .map(|(member, at)| {
                    let mut table = member.table.clone();
                    table.set_token_len(at + 1);
                    table
                })
                .collect::<Vec<_>>();
            let tables = tables.iter().collect::<Vec<_>>();
            self.paging.prepare(&tables, &at, &vec![1; self.rows])?;
            let state = sessions[members[0].session]
                .prediction
                .as_mut()
                .ok_or(Error::InvalidDecoderKernel("session has no prediction module state"))?;
            self.prediction.execute_packed(embedding, state, &self.paging)?;
            let drafted = &self.prediction.drafted;
            stream.copy_device_range(drafted, 0..drafted.len(), &mut self.head.selected, 0)?;
            self.head.logits_of_selected()?;
        }
        for member in members {
            sessions[member.session].drafted = count;
        }
        Ok(())
    }
}
