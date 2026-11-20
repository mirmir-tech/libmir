//! Dense multi-head latent attention in FP32, the way the checkpoint defines
//! it: every head's keys and values are expanded from the latent and nothing
//! is absorbed.

use models::layout::RotaryPairing;

use super::super::LatentAttentionSpec;

pub(super) struct Model {
    pub spec: LatentAttentionSpec,
    pub latent_norm: Vec<f32>,
    pub key_value_up: Vec<f32>,
    pub inverse_frequencies: Vec<f32>,
    pub epsilon: f32,
    pub rope_scale: f32,
    pub softmax_scale: f32,
}

pub(super) struct Token {
    /// `[heads, nope + rope]`.
    pub query: Vec<f32>,
    /// `[latent_rank + rope]`.
    pub latent: Vec<f32>,
}

struct Expanded {
    nope_keys: Vec<f32>,
    rope_key: Vec<f32>,
    values: Vec<f32>,
}

impl Model {
    /// `[heads · value]` output of every token of `sequence` attending
    /// causally to the tokens before it.
    pub fn attend(&self, sequence: &[Token]) -> Vec<Vec<f32>> {
        let expanded = sequence
            .iter()
            .enumerate()
            .map(|(position, token)| self.expand(token, position))
            .collect::<Vec<_>>();
        sequence
            .iter()
            .enumerate()
            .map(|(position, token)| self.query(token, position, &expanded[..=position]))
            .collect()
    }

    fn expand(&self, token: &Token, position: usize) -> Expanded {
        let spec = self.spec;
        let rank = spec.latent_rank;
        let latent = &token.latent[..rank];
        let mean = latent.iter().map(|value| value * value).sum::<f32>() / float(rank);
        let inverse_rms = (mean + self.epsilon).sqrt().recip();
        let normalized = latent
            .iter()
            .zip(&self.latent_norm)
            .map(|(value, weight)| value * inverse_rms * weight)
            .collect::<Vec<_>>();
        let row = |index: usize| -> f32 {
            let weights = &self.key_value_up[index * rank..(index + 1) * rank];
            weights.iter().zip(&normalized).map(|(w, c)| w * c).sum()
        };
        let projected = spec.nope_dim + spec.value_dim;
        let mut nope_keys = Vec::with_capacity(spec.heads * spec.nope_dim);
        let mut values = Vec::with_capacity(spec.heads * spec.value_dim);
        for head in 0..spec.heads {
            nope_keys.extend((0..spec.nope_dim).map(|d| row(head * projected + d)));
            values.extend((0..spec.value_dim).map(|d| row(head * projected + spec.nope_dim + d)));
        }
        Expanded {
            nope_keys,
            rope_key: self.rotate(&token.latent[rank..], position),
            values,
        }
    }

    fn query(&self, token: &Token, position: usize, visible: &[Expanded]) -> Vec<f32> {
        let spec = self.spec;
        let width = spec.nope_dim + spec.rope_dim;
        let mut output = Vec::with_capacity(spec.heads * spec.value_dim);
        for head in 0..spec.heads {
            let query = &token.query[head * width..(head + 1) * width];
            let rope_query = self.rotate(&query[spec.nope_dim..], position);
            let scores = visible
                .iter()
                .map(|key| {
                    let nope = &key.nope_keys[head * spec.nope_dim..(head + 1) * spec.nope_dim];
                    let dot = dot(&query[..spec.nope_dim], nope) + dot(&rope_query, &key.rope_key);
                    dot * self.softmax_scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            output.extend((0..spec.value_dim).map(|d| {
                visible
                    .iter()
                    .zip(&weights)
                    .map(|(key, weight)| weight * key.values[head * spec.value_dim + d])
                    .sum::<f32>()
                    / denominator
            }));
        }
        output
    }

    fn rotate(&self, input: &[f32], position: usize) -> Vec<f32> {
        let half = input.len() / 2;
        let mut output = input.to_vec();
        for (pair, frequency) in self.inverse_frequencies.iter().enumerate() {
            let (first, second) = match self.spec.rotary_pairing {
                RotaryPairing::Interleaved => (2 * pair, 2 * pair + 1),
                RotaryPairing::Halves => (pair, pair + half),
            };
            let (sine, cosine) = (float(position) * frequency).sin_cos();
            let (sine, cosine) = (sine * self.rope_scale, cosine * self.rope_scale);
            output[first] = input[first].mul_add(cosine, -input[second] * sine);
            output[second] = input[second].mul_add(cosine, input[first] * sine);
        }
        output
    }
}

fn float(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
