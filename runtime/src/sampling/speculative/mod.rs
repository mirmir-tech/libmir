//! Rejection sampling of speculative drafts.
//!
//! Each draft is the prediction module's argmax, a point mass `q = δ(d)`.
//! Row `j` keeps its draft with the target probability `p_j(d)`; the first
//! row that rejects samples from `p_j` without `d`, renormalized; when every
//! draft is kept, the row after them samples from its own `p`. Every emitted
//! token then has the target distribution, as without drafts.

use super::{
    SplitMix64, WeightedCandidate,
    candidate::{top_k_limit, truncate_top_p},
    draw,
};
use crate::error::{Result, RuntimeError};

/// Tokens more than this many temperature-scaled nats below the top carry
/// under `e^-30` of the mass each and are left out before sorting.
const NEGLIGIBLE_NATS: f32 = 30.0;

/// The target distribution of a sampled decode step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftSampling {
    pub temperature: f32,
    pub top_p: f32,
    /// `0` keeps every token.
    pub top_k: usize,
}

/// Verifies `drafts` against target logit `rows` (one more than the
/// drafts). Returns how many drafts are kept and the sampled token after
/// them.
pub fn verify_drafts(
    rows: &[&[f32]],
    drafts: &[u32],
    sampling: DraftSampling,
    seed: u64,
) -> Result<(usize, u32)> {
    if rows.len() != drafts.len() + 1 {
        return Err(RuntimeError::Backend("draft rows differ from drafts".into()));
    }
    let mut rng = SplitMix64::new(seed);
    for (row, (logits, &draft)) in rows.iter().zip(drafts).enumerate() {
        let weights = distribution(logits, sampling)?;
        let total = weights.iter().map(|candidate| candidate.weight).sum::<f64>();
        let kept = weights
            .iter()
            .find(|candidate| candidate.token_id == draft)
            .map_or(0.0, |candidate| candidate.weight / total);
        if rng.next_unit() < kept {
            continue;
        }
        let rest = weights
            .into_iter()
            .filter(|candidate| candidate.token_id != draft)
            .collect::<Vec<_>>();
        return Ok((row, draw(&mut rng, &rest)?));
    }
    let last = rows.last().ok_or_else(|| RuntimeError::Backend("no target rows".into()))?;
    Ok((drafts.len(), draw(&mut rng, &distribution(last, sampling)?)?))
}

/// Temperature, then top-p, then top-k over the ranked logits, as
/// [`super::Sampler`] filters them.
fn distribution(logits: &[f32], sampling: DraftSampling) -> Result<Vec<WeightedCandidate>> {
    let max = logits
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .fold(f32::MIN, f32::max);
    let floor = NEGLIGIBLE_NATS.mul_add(-sampling.temperature, max);
    let mut ranked = logits
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite() && **score >= floor)
        .map(|(index, score)| Ok((u32::try_from(index)?, *score)))
        .collect::<Result<Vec<_>>>()?;
    if ranked.is_empty() {
        return Err(RuntimeError::Backend("logits contain no finite values".into()));
    }
    ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    let temperature = f64::from(sampling.temperature);
    let mut total = 0.0;
    let weights = ranked
        .into_iter()
        .map(|(token_id, score)| {
            let weight = (f64::from(score - max) / temperature).exp();
            total += weight;
            WeightedCandidate { token_id, weight }
        })
        .collect();
    let weights = truncate_top_p(weights, total, f64::from(sampling.top_p));
    let limit = top_k_limit(weights.len(), sampling.top_k);
    Ok(weights.into_iter().take(limit).collect())
}

#[cfg(test)]
mod tests;
