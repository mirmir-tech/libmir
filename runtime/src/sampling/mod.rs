use crate::{
    backend::{CandidateLogitsTrace, LogitsTrace},
    error::{Result, RuntimeError},
};

mod candidate;
use candidate::{
    Candidate, WeightedCandidate, candidates, compact_candidates, top_k_limit, truncate_top_p,
};

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub vocab_size: Option<usize>,
    pub seed: u64,
}

#[derive(Debug)]
pub struct Sampler {
    config: SamplerConfig,
    rng: SplitMix64,
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            rng: SplitMix64::new(config.seed),
        })
    }

    pub fn sample(&mut self, logits: &LogitsTrace) -> Result<u32> {
        self.sample_with_history(logits, &[])
    }

    pub fn sample_with_history(&mut self, logits: &LogitsTrace, history: &[u32]) -> Result<u32> {
        let logits = self.logits(logits)?;
        let mut candidates = candidates(logits, history, self.config.repetition_penalty)?;
        self.sample_ranked(&mut candidates)
    }

    pub fn sample_candidates_with_history(
        &mut self,
        trace: &CandidateLogitsTrace,
        history: &[u32],
    ) -> Result<u32> {
        let mut candidates = compact_candidates(trace, history, self.config.repetition_penalty)?;
        self.sample_ranked(&mut candidates)
    }

    pub fn draw_unit_f32(&mut self) -> f32 {
        self.rng.next_unit_f32()
    }

    fn sample_ranked(&mut self, candidates: &mut [Candidate]) -> Result<u32> {
        candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
        if self.config.temperature <= f32::EPSILON || self.config.top_k == 1 {
            return Ok(candidates[0].token_id);
        }
        let weights = self.filtered_weights(candidates);
        draw(&mut self.rng, &weights)
    }

    fn filtered_weights(&self, candidates: &[Candidate]) -> Vec<WeightedCandidate> {
        let max = candidates[0].score;
        let mut weights = Vec::with_capacity(candidates.len());
        let mut total = 0.0;
        for candidate in candidates {
            let scaled =
                (f64::from(candidate.score - max) / f64::from(self.config.temperature)).exp();
            total += scaled;
            weights.push(WeightedCandidate {
                token_id: candidate.token_id,
                weight: scaled,
            });
        }
        let weights = truncate_top_p(weights, total, f64::from(self.config.top_p));
        let limit = top_k_limit(weights.len(), self.config.top_k);
        weights.into_iter().take(limit).collect()
    }

    fn logits<'a>(&self, trace: &'a LogitsTrace) -> Result<&'a [f32]> {
        let logits = last_logits(trace)?;
        Ok(match self.config.vocab_size {
            Some(limit) => logits.get(..limit).ok_or_else(|| {
                RuntimeError::Backend(format!(
                    "sampler vocab limit {limit} exceeds logits length {}",
                    logits.len()
                ))
            })?,
            None => logits,
        })
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            vocab_size: None,
            seed: 0x4d49_524d_4952,
        }
    }
}

impl SamplerConfig {
    fn validate(self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RuntimeError::Config("temperature must be finite and >= 0".into()));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(RuntimeError::Config("top_p must be in (0, 1]".into()));
        }
        if self.vocab_size == Some(0) {
            return Err(RuntimeError::Config(
                "sampler vocab_size must be greater than zero".into(),
            ));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty < 1.0 {
            return Err(RuntimeError::Config("repetition_penalty must be finite and >= 1".into()));
        }
        Ok(())
    }
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        let bytes = self.next_u64().to_le_bytes();
        let upper = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        f64::from(upper) / (f64::from(u32::MAX) + 1.0)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let bytes = self.next_u64().to_le_bytes();
        let upper = u16::from_le_bytes([bytes[6], bytes[7]]);
        f32::from(upper) / (f32::from(u16::MAX) + 1.0)
    }
}

fn last_logits(trace: &LogitsTrace) -> Result<&[f32]> {
    if trace.shape.len() != 3 || trace.shape[0] != 1 || trace.shape[1] < 1 {
        return Err(RuntimeError::Backend(format!(
            "logits trace must be [1, S, V], got {:?}",
            trace.shape
        )));
    }
    let seq = usize::try_from(trace.shape[1])?;
    let vocab = usize::try_from(trace.shape[2])?;
    let start = (seq - 1) * vocab;
    let end = start + vocab;
    trace
        .values
        .get(start..end)
        .ok_or_else(|| RuntimeError::Backend("logits trace is truncated".into()))
}

fn draw(rng: &mut SplitMix64, weights: &[WeightedCandidate]) -> Result<u32> {
    let total: f64 = weights.iter().map(|candidate| candidate.weight).sum();
    if total <= 0.0 {
        return Err(RuntimeError::Backend("sample weights sum to zero".into()));
    }
    let mut threshold = rng.next_unit() * total;
    for candidate in weights {
        threshold -= candidate.weight;
        if threshold <= 0.0 {
            return Ok(candidate.token_id);
        }
    }
    weights
        .last()
        .map(|candidate| candidate.token_id)
        .ok_or_else(|| RuntimeError::Backend("no sample candidates".into()))
}

#[cfg(test)]
mod tests;
