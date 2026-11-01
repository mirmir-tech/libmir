//! Packed drafting passes for three sessions at once against plain greedy
//! decoding of each prompt alone: tokens may part only at near-ties.

use std::time::Instant;

use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable},
};

use super::{BLOCK, DRAFT_TOKENS};
use crate::{
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, PagedKvCache, Result,
    backend::{CudaSharedRoutedPrefillBatch, DraftingRow},
};

/// Blocks each session's table owns, apart from the others'.
const SESSION_BLOCKS: u32 = 20;

pub(super) struct Batched {
    pub(super) tokens: Vec<Vec<u32>>,
    pub(super) seconds: f64,
    pub(super) steps: usize,
}

pub(super) fn batched(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
    prompts: &[Vec<u32>],
    count: usize,
) -> Result<Batched> {
    let mut owned = prompts
        .iter()
        .map(|_| template.instantiate_with_caches(caches))
        .collect::<Result<Vec<_>>>()?;
    let mut tables = (0..prompts.len())
        .map(|row| {
            let mut table = BlockTable::with_block_size(BLOCK);
            for block in 0..SESSION_BLOCKS {
                table.push(BlockId(u32::try_from(row).unwrap_or(0) * SESSION_BLOCKS + block));
            }
            table
        })
        .collect::<Vec<_>>();
    let capacity = prompts
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .max(prompts.len() * (DRAFT_TOKENS + 1));
    let counts = prompts.iter().map(Vec::len).collect::<Vec<_>>();
    let mut batch = template.prepare_padded_prefill_batch(&counts, capacity)?;
    let started = Instant::now();
    let rows = prompts
        .iter()
        .map(|prompt| DraftingRow::Prompt {
            tokens: prompt,
            following: &prompt[1..],
            sample: Some(SamplingLogits::None),
        })
        .collect::<Vec<_>>();
    for (table, prompt) in tables.iter_mut().zip(prompts) {
        table.set_token_len(prompt.len());
    }
    let steps = run(&mut batch, template, &mut owned, &rows, &tables)?;
    let mut tokens = prompts.iter().map(|_| Vec::new()).collect::<Vec<_>>();
    let mut next = steps.iter().map(|step| step.1).collect::<Vec<_>>();
    let mut passes = 0;
    while tokens.iter().any(|tokens| tokens.len() < count) {
        for ((tokens, next), (table, session)) in
            tokens.iter_mut().zip(&next).zip(tables.iter_mut().zip(&owned))
        {
            tokens.push(*next);
            table.set_token_len(session.position() + 1);
        }
        let counts = owned.iter().map(|session| 1 + session.pending_drafts()).collect::<Vec<_>>();
        batch.reconfigure(template, &counts)?;
        let rows = next
            .iter()
            .map(|token| DraftingRow::Decode {
                token: *token,
                sampling: SamplingLogits::None,
                speculation: Some(runtime::speculation::DraftBounds::default()),
            })
            .collect::<Vec<_>>();
        let steps = run(&mut batch, template, &mut owned, &rows, &tables)?;
        passes += 1;
        for ((tokens, next), (accepted, token)) in tokens.iter_mut().zip(&mut next).zip(steps) {
            tokens.extend(accepted);
            *next = token;
        }
    }
    for tokens in &mut tokens {
        tokens.truncate(count);
    }
    Ok(Batched {
        tokens,
        seconds: started.elapsed().as_secs_f64(),
        steps: passes,
    })
}

fn run(
    batch: &mut CudaSharedRoutedPrefillBatch,
    template: &CudaSharedRoutedModelTemplate,
    owned: &mut [CudaSharedRoutedModelSession],
    rows: &[DraftingRow<'_>],
    tables: &[BlockTable],
) -> Result<Vec<(Vec<u32>, u32)>> {
    let mut sessions = owned.iter_mut().collect::<Vec<_>>();
    let tables = tables.iter().collect::<Vec<_>>();
    let steps = batch.execute_drafting(template, &mut sessions, rows, &tables, DRAFT_TOKENS)?;
    steps
        .into_iter()
        .map(|step| {
            step.map(|step| (step.accepted, step.next))
                .ok_or(crate::Error::InvalidExecutionPlan("drafting row produced no token"))
        })
        .collect()
}
