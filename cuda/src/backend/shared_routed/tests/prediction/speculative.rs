//! Greedy speculative decoding against plain greedy decoding on the same
//! checkpoint: every speculative token is the verification argmax by
//! construction, so the sequences may part only where plain decoding's top
//! two logits nearly tie.

use std::time::Instant;

use mircuda::bf16;
use runtime::kv::{BlockId, BlockTable};
use uuid::Uuid;

use super::{BLOCK, argmax, read};
use crate::{CudaSharedRoutedModelTemplate, PagedKvCache, Result};

pub(super) struct Run {
    pub(super) tokens: Vec<u32>,
    /// Top-1 minus top-2 logit of plain decoding at every generated token.
    pub(super) margins: Vec<f32>,
    pub(super) seconds: f64,
    pub(super) steps: usize,
}

pub(super) fn plain(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
    prompt: &[u32],
    count: usize,
) -> Result<Run> {
    let mut session = template.instantiate_with_caches(caches)?;
    let id = Uuid::new_v4();
    let mut table = table(prompt.len() + count + 1);
    table.set_token_len(prompt.len());
    let started = Instant::now();
    let mut logits = read(template, session.prefill(id, prompt, &table)?)?;
    let (mut tokens, mut margins) = (Vec::new(), Vec::new());
    while tokens.len() < count {
        let token = argmax(&logits);
        margins.push(margin(&logits));
        tokens.push(token);
        table.set_token_len(prompt.len() + tokens.len());
        logits = read(template, session.decode(id, token, &table)?)?;
    }
    Ok(Run {
        tokens,
        margins,
        seconds: started.elapsed().as_secs_f64(),
        steps: count,
    })
}

pub(super) fn speculative(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
    prompt: &[u32],
    count: usize,
    draft_tokens: usize,
) -> Result<Run> {
    let mut session = template.instantiate_with_caches(caches)?;
    let id = Uuid::new_v4();
    let rows = prompt.len();
    let table = table(rows + count + 3 * draft_tokens + 2);
    let mut pass = table.clone();
    pass.set_token_len(rows);
    let started = Instant::now();
    let first = argmax(&read(template, session.prefill_rows(id, prompt, &pass)?)?);
    let next_tokens = prompt[1..].iter().copied().chain([first]).collect::<Vec<_>>();
    session.predict(id, rows, &next_tokens, &pass)?;
    session.draft(id, rows, rows - 1, rows - 1, draft_tokens, &table)?;
    let (mut tokens, mut next, mut steps) = (Vec::new(), first, 0);
    while tokens.len() < count {
        tokens.push(next);
        let mut fed = table.clone();
        fed.set_token_len(rows + tokens.len());
        let greedy = runtime::backend::SamplingLogits::None;
        let bounds = runtime::speculation::DraftBounds::default();
        let step = session.decode_speculative(id, next, draft_tokens, &greedy, &bounds, &fed)?;
        steps += 1;
        tokens.extend(&step.accepted);
        next = step.next;
    }
    tokens.truncate(count);
    Ok(Run {
        tokens,
        margins: Vec::new(),
        seconds: started.elapsed().as_secs_f64(),
        steps,
    })
}

fn table(tokens: usize) -> BlockTable {
    let mut table = BlockTable::with_block_size(BLOCK);
    for block in 0..tokens.div_ceil(BLOCK) {
        table.push(BlockId(u32::try_from(block).unwrap_or(u32::MAX)));
    }
    table
}

fn margin(values: &[bf16]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for value in values.iter().map(|value| value.to_f32()) {
        if value > first {
            second = first;
            first = value;
        } else if value > second {
            second = value;
        }
    }
    first - second
}
