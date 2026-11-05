//! A packed pass with the prediction module: prompt chunks and decode rows
//! that verify their session's drafts, then the drafts after them.
//!
//! The decoder runs once over every row. The output head reads only the rows
//! that need tokens: a final chunk's last row and every row of a decode
//! session. The module then runs over all rows with the token after each,
//! decode sessions keep their accepted prefix, and one chain drafts for every
//! session that continues.

mod accept;
mod chain;
mod head;

use std::collections::HashMap;

use runtime::{backend::SamplingLogits, kv::BlockTable, speculation::DraftBounds};

use self::{
    accept::{Verdicts, following_tokens},
    chain::{ChainRow, DraftChain},
    head::RowHead,
};
use super::CudaSharedRoutedPrefillBatch;
use crate::{
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Error, Result, SpeculativeStep,
    backend::shared_routed::prediction::PredictionExecution,
};

/// One row of a drafting pass.
#[derive(Clone, Debug)]
pub enum DraftingRow<'a> {
    /// A prompt chunk and the token after each of its tokens; a final chunk
    /// has no token after its last and samples it with `sample`.
    Prompt {
        tokens: &'a [u32],
        following: &'a [u32],
        sample: Option<SamplingLogits>,
    },
    /// The fed token of a decode session; a speculative row also verifies
    /// the drafts its batch count leaves room for, kept within its bounds.
    Decode {
        token: u32,
        sampling: SamplingLogits,
        speculation: Option<DraftBounds>,
    },
}

/// The module and heads of a batch's drafting passes, built on first use.
#[derive(Debug)]
pub(super) struct BatchDrafting {
    prediction: PredictionExecution,
    heads: HashMap<usize, RowHead>,
    chains: HashMap<usize, DraftChain>,
}

impl CudaSharedRoutedPrefillBatch {
    /// Runs `rows` and drafts up to `draft_tokens` tokens for every session
    /// that continues. A final chunk returns its sampled token, a decode row
    /// its verified drafts and the token after them.
    pub fn execute_drafting(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        rows: &[DraftingRow<'_>],
        tables: &[&BlockTable],
        draft_tokens: usize,
    ) -> Result<Vec<Option<SpeculativeStep>>> {
        if rows.len() != self.rows || sessions.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("drafting rows differ from the batch"));
        }
        let starts = sessions.iter().map(|session| session.position).collect::<Vec<_>>();
        let offsets = self.offsets();
        let mut drafted = Vec::new();
        let mut tokens = Vec::with_capacity(offsets[self.rows]);
        for (index, row) in rows.iter().enumerate() {
            match row {
                DraftingRow::Prompt { tokens: prompt, .. } => tokens.extend_from_slice(prompt),
                DraftingRow::Decode { token, .. } => {
                    let drafts = self.counts[index] - 1;
                    sessions[index].arm_rollback(drafts)?;
                    tokens.push(*token);
                    tokens.extend(std::iter::repeat_n(0, drafts));
                    if drafts > 0 {
                        drafted.push(index);
                    }
                },
            }
        }
        // A decode row's table holds its fed token; the pass reaches its drafts.
        let tables = tables
            .iter()
            .zip(&starts)
            .zip(&self.counts)
            .map(|((table, start), count)| {
                let mut table = (*table).clone();
                table.set_token_len(start + count);
                table
            })
            .collect::<Vec<_>>();
        let tables = tables.iter().collect::<Vec<_>>();
        self.run(sessions, &tokens, &tables, &starts, &drafted)?;
        let hidden = if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        };
        let drafting = self
            .drafting
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("batch drafting is missing"))?;
        drafting.prediction.normalize_decoder(hidden, &template.final_norm)?;
        let verdicts = Verdicts::compute(drafting, template, sessions, rows, &offsets, &starts)?;
        let following = following_tokens(rows, &self.counts, &verdicts, self.token_ids.len());
        drafting.prediction.upload_tokens(&following)?;
        let state = sessions[0]
            .prediction
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("session has no prediction module state"))?;
        drafting.prediction.execute_packed(&self.embedding, state, &self.paging)?;
        let mut members = Vec::new();
        let mut sources = Vec::new();
        for (index, verdict) in verdicts.rows.iter().enumerate() {
            let Some((accepted, _)) = verdict.step else {
                continue;
            };
            if matches!(rows[index], DraftingRow::Decode { .. }) {
                sessions[index].roll_back(starts[index], accepted + 1)?;
            }
            if verdict.drafts {
                let row = offsets[index] + verdict.source;
                members.push(ChainRow {
                    session: index,
                    position: starts[index] + verdict.source,
                    table: tables[index],
                });
                sources.push(u32::try_from(row)?);
            } else {
                sessions[index].forget_drafts();
            }
        }
        let count = members
            .iter()
            .map(|member| {
                let capacity = member.table.capacity(template.cache.block_size);
                draft_tokens.min(capacity.saturating_sub(member.position + 1))
            })
            .min()
            .unwrap_or(0);
        if count > 0 {
            let chain = match drafting.chains.entry(members.len()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(DraftChain::new(template, members.len())?)
                },
            };
            let source = (&drafting.prediction.drafted, self.token_ids.len(), sources.as_slice());
            chain.run(&self.embedding, sessions, &members, source, count)?;
        } else {
            for member in &members {
                sessions[member.session].forget_drafts();
            }
        }
        Ok(verdicts.steps())
    }

    /// Where each row's tokens start in the packed pass, and their total.
    fn offsets(&self) -> Vec<usize> {
        std::iter::once(0)
            .chain(self.counts.iter().scan(0, |total, count| {
                *total += count;
                Some(*total)
            }))
            .collect()
    }
}

impl BatchDrafting {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, tokens: usize) -> Result<Self> {
        let weights = template
            .prediction
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("model has no prediction module loaded"))?;
        Ok(Self {
            prediction: PredictionExecution::new(template, weights, tokens)?,
            heads: HashMap::new(),
            chains: HashMap::new(),
        })
    }
}
