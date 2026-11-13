//! Tokens of a drafting pass: sampled final chunks, verified decode rows,
//! and the token after every row for the prediction module.

use std::collections::hash_map::Entry;

use runtime::{backend::SamplingLogits, sampling::verify_drafts};

use super::{BatchDrafting, DraftingRow, head::RowHead};
use crate::{
    CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, Result, SpeculativeStep,
    backend::shared::routed::session::{draft_sampling, seed},
};

/// The outcome of one row.
#[derive(Debug, Default)]
pub(super) struct RowVerdict {
    /// Kept drafts and the token after them, for rows that produce a token.
    pub(super) step: Option<(usize, u32)>,
    /// The kept drafts themselves.
    pub(super) kept: Vec<u32>,
    /// The session drafts on from the module output at this row offset.
    pub(super) drafts: bool,
    pub(super) source: usize,
}

#[derive(Debug)]
pub(super) struct Verdicts {
    pub(super) rows: Vec<RowVerdict>,
}

impl Verdicts {
    /// Reads the tokens the pass produced: one wait for the device.
    pub(super) fn compute(
        drafting: &mut BatchDrafting,
        template: &CudaSharedRoutedModelTemplate,
        sessions: &[&mut CudaSharedRoutedModelSession],
        rows: &[DraftingRow<'_>],
        offsets: &[usize],
        starts: &[usize],
    ) -> Result<Self> {
        let (mut indices, mut policies, sampled) = selection(rows, offsets)?;
        if indices.is_empty() {
            return Ok(Self {
                rows: rows.iter().map(|_| RowVerdict::default()).collect(),
            });
        }
        // Power-of-two row counts bound the head shapes a batch builds; the
        // padding repeats the last row and is ignored.
        let rows_needed = indices.len();
        let bucket = rows_needed.next_power_of_two();
        indices.resize(bucket, indices[rows_needed - 1]);
        policies.resize(bucket, SamplingLogits::None);
        let head = match drafting.heads.entry(bucket) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(RowHead::new(template, bucket)?),
        };
        let prediction = &drafting.prediction;
        head.gather(&prediction.hidden, prediction.tokens, &indices)?;
        let tokens = template.backend.read_tokens(head.sample(&policies)?)?;
        let logits = sampled.then(|| template.backend.read_logits(&head.logits)).transpose()?;
        let vocab = template.decoder.vocab_size;
        let mut cursor = 0;
        let mut verdicts = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let count = offsets[index + 1] - offsets[index];
            verdicts.push(match row {
                DraftingRow::Prompt { sample: Some(_), .. } => {
                    cursor += 1;
                    let step = Some((0, tokens[cursor - 1]));
                    RowVerdict {
                        step,
                        kept: Vec::new(),
                        drafts: true,
                        source: count - 1,
                    }
                },
                DraftingRow::Prompt { sample: None, .. } => RowVerdict::default(),
                DraftingRow::Decode { sampling, speculation, .. } => {
                    let span = cursor..cursor + count;
                    cursor += count;
                    if let Some(bounds) = speculation {
                        let drafts = match &sessions[index].drafts {
                            Some(buffer) if count > 1 => {
                                let mut drafts = template.backend.read_tokens(buffer)?;
                                drafts.truncate(count - 1);
                                drafts
                            },
                            _ => Vec::new(),
                        };
                        let sampled = draft_sampling(sampling)?.zip(logits.as_ref());
                        let (accepted, next) =
                            if let Some(((target, draw, bound), logits)) = sampled {
                                let rows = span
                                    .map(|row| &logits[row * vocab..row * vocab + bound.min(vocab)])
                                    .collect::<Vec<_>>();
                                verify_drafts(&rows, &drafts, target, seed(draw, starts[index])?)?
                            } else {
                                let target = &tokens[span];
                                let kept = drafts.iter().zip(target).take_while(|(d, t)| d == t);
                                let accepted = kept.count();
                                (accepted, target[accepted])
                            };
                        let (accepted, next) = bounds.apply(&drafts, accepted, next);
                        let kept = drafts[..accepted].to_vec();
                        RowVerdict {
                            step: Some((accepted, next)),
                            kept,
                            drafts: true,
                            source: accepted,
                        }
                    } else {
                        let step = Some((0, tokens[span.start]));
                        RowVerdict {
                            step,
                            kept: Vec::new(),
                            drafts: false,
                            source: 0,
                        }
                    }
                },
            });
        }
        Ok(Self { rows: verdicts })
    }

    pub(super) fn steps(self) -> Vec<Option<SpeculativeStep>> {
        self.rows
            .into_iter()
            .map(|row| {
                row.step.map(|(_, next)| SpeculativeStep {
                    accepted: row.kept,
                    next,
                    drafts: Vec::new(),
                })
            })
            .collect()
    }
}

/// The packed rows the head reads, their policies, and whether a sampled
/// verification needs logits on the host: a final chunk's last row and every
/// row of a decode session.
fn selection(
    rows: &[DraftingRow<'_>],
    offsets: &[usize],
) -> Result<(Vec<u32>, Vec<SamplingLogits>, bool)> {
    let (mut indices, mut policies, mut sampled) = (Vec::new(), Vec::new(), false);
    for (index, row) in rows.iter().enumerate() {
        match row {
            DraftingRow::Prompt { sample: Some(policy), .. } => {
                indices.push(u32::try_from(offsets[index + 1] - 1)?);
                policies.push(policy.clone());
            },
            DraftingRow::Prompt { sample: None, .. } => {},
            DraftingRow::Decode { sampling, speculation, .. } => {
                let rows = offsets[index]..offsets[index + 1];
                let speculative = speculation.is_some();
                // Sampled verification reads logits; greedy rows take the argmax.
                let policy = if speculative {
                    SamplingLogits::None
                } else {
                    sampling.clone()
                };
                sampled |= speculative && draft_sampling(sampling)?.is_some();
                for row in rows {
                    indices.push(u32::try_from(row)?);
                    policies.push(policy.clone());
                }
            },
        }
    }
    Ok((indices, policies, sampled))
}

/// The token after every packed row, padded to the pass's `capacity`.
pub(super) fn following_tokens(
    rows: &[DraftingRow<'_>],
    counts: &[usize],
    verdicts: &Verdicts,
    capacity: usize,
) -> Vec<u32> {
    let mut following = Vec::with_capacity(capacity);
    for ((row, count), verdict) in rows.iter().zip(counts).zip(&verdicts.rows) {
        let next = verdict.step.map_or(0, |(_, next)| next);
        match row {
            DraftingRow::Prompt { following: known, .. } => {
                following.extend(known.iter().copied().take(*count));
                following.extend(std::iter::repeat_n(next, count - known.len().min(*count)));
            },
            DraftingRow::Decode { .. } => {
                following.extend(&verdict.kept);
                following.extend(std::iter::repeat_n(next, count - verdict.kept.len()));
            },
        }
    }
    following.resize(capacity, 0);
    following
}
