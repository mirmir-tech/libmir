use crate::{
    backend::CandidateLogitsTrace,
    error::{Result, RuntimeError},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate {
    pub token_id: u32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WeightedCandidate {
    pub token_id: u32,
    pub weight: f64,
}

pub(super) fn compact_candidates(
    trace: &CandidateLogitsTrace,
    history: &[u32],
    repetition_penalty: f32,
) -> Result<Vec<Candidate>> {
    if trace.token_ids.len() != trace.scores.len() {
        return Err(RuntimeError::Backend(format!(
            "candidate logits have {} ids but {} scores",
            trace.token_ids.len(),
            trace.scores.len()
        )));
    }
    let candidates = trace
        .token_ids
        .iter()
        .copied()
        .zip(trace.scores.iter().copied())
        .filter(|(_, score)| score.is_finite())
        .map(|(token_id, score)| Candidate {
            token_id,
            score: penalized_score(score, token_id, history, repetition_penalty),
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(RuntimeError::Backend("candidate logits contain no finite values".into()));
    }
    Ok(candidates)
}

pub(super) fn candidates(
    logits: &[f32],
    history: &[u32],
    repetition_penalty: f32,
) -> Result<Vec<Candidate>> {
    let mut candidates = Vec::with_capacity(logits.len());
    for (index, score) in logits.iter().copied().enumerate() {
        if score.is_finite() {
            let token_id = u32::try_from(index)?;
            candidates.push(Candidate {
                token_id,
                score: penalized_score(score, token_id, history, repetition_penalty),
            });
        }
    }
    if candidates.is_empty() {
        return Err(RuntimeError::Backend("logits contain no finite values".into()));
    }
    Ok(candidates)
}

pub(super) fn top_k_limit(len: usize, top_k: usize) -> usize {
    if top_k == 0 {
        len
    } else {
        len.min(top_k.max(1))
    }
}

pub(super) fn truncate_top_p(
    weights: Vec<WeightedCandidate>,
    total: f64,
    top_p: f64,
) -> Vec<WeightedCandidate> {
    let mut kept = Vec::new();
    let mut cumulative = 0.0;
    for candidate in weights {
        cumulative += candidate.weight / total;
        kept.push(candidate);
        if cumulative >= top_p {
            break;
        }
    }
    kept
}

fn penalized_score(score: f32, token_id: u32, history: &[u32], repetition_penalty: f32) -> f32 {
    if repetition_penalty <= 1.0 || !history.contains(&token_id) {
        return score;
    }
    if score.is_sign_positive() {
        score / repetition_penalty
    } else {
        score * repetition_penalty
    }
}
