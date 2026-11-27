//! A routed-expert block's routing evaluated on the host in `f32`: the
//! reference backend routers are held to, not a runtime path.

use super::{RoutedExpertsSpec, RouterNormalization};
use crate::error::{ModelsError, Result};

#[cfg(test)]
mod tests;

/// One selected expert and the weight on its output (or, for an
/// input-scaled router, on its input).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedExpert {
    pub expert: usize,
    pub weight: f32,
}

impl RoutedExpertsSpec {
    /// The `top_k` experts one token's router `logits` select, in selection
    /// order (best first, the lower index on ties), weighted by the spec's
    /// normalization, group limit and output scale. A sigmoid-biased router
    /// takes its `correction_bias`.
    // Expert counts and scales are far below `f32`'s exact range.
    #[allow(clippy::cast_possible_truncation)]
    pub fn route(
        &self,
        logits: &[f32],
        correction_bias: Option<&[f32]>,
    ) -> Result<Vec<RoutedExpert>> {
        let invalid =
            |detail: &str| ModelsError::InvalidConfig(format!("reference routing: {detail}"));
        if logits.len() != self.expert_count || self.top_k > logits.len() {
            return Err(invalid("one logit per expert and at most that many selections"));
        }
        let scale = self.output_scale as f32;
        let gate = |expert: usize| (1.0 + (-logits[expert]).exp()).recip();
        let routing = self.router_normalization;
        let choice: Box<dyn Fn(usize) -> f32> = match (routing, correction_bias) {
            (RouterNormalization::SigmoidBiasedTopK, Some(bias)) if bias.len() == logits.len() => {
                Box::new(move |expert| gate(expert) + bias[expert])
            },
            (RouterNormalization::SigmoidBiasedTopK, _) => {
                return Err(invalid("a sigmoid-biased router needs one bias per expert"));
            },
            (RouterNormalization::SparseMixer { jitter_eps }, _) => {
                return Ok(sparse_mixer(logits, self.top_k, jitter_eps as f32, scale));
            },
            _ => Box::new(|expert| logits[expert]),
        };
        let selected = self.select(logits.len(), &choice);
        let weight = |expert: usize| -> f32 {
            match routing {
                RouterNormalization::SigmoidInputScaledTopK => gate(expert),
                RouterNormalization::SigmoidBiasedTopK => {
                    gate(expert) / selected.iter().map(|&chosen| gate(chosen)).sum::<f32>()
                },
                _ => {
                    let best = logits[selected[0]];
                    let exponent = |index: usize| (logits[index] - best).exp();
                    let total: f32 = if routing == RouterNormalization::SoftmaxTopK {
                        selected.iter().map(|&chosen| exponent(chosen)).sum()
                    } else {
                        (0..logits.len()).map(exponent).sum()
                    };
                    exponent(expert) / total
                },
            }
        };
        Ok(selected
            .iter()
            .map(|&expert| RoutedExpert { expert, weight: weight(expert) * scale })
            .collect())
    }

    /// The best `top_k` experts by `choice` within the best groups.
    fn select(&self, experts: usize, choice: &dyn Fn(usize) -> f32) -> Vec<usize> {
        let mut order: Vec<usize> = (0..experts).collect();
        if let Some(limit) = self.group_limit {
            let size = limit.group_size(experts);
            // A sigmoid router ranks a group by its two best scores.
            let ranked = if self.router_normalization == RouterNormalization::SigmoidBiasedTopK {
                2
            } else {
                1
            };
            let score = |group: usize| {
                let mut scores: Vec<f32> = (group * size..(group + 1) * size).map(choice).collect();
                scores.sort_by(|a, b| b.total_cmp(a));
                scores.iter().take(ranked).sum::<f32>()
            };
            let mut groups: Vec<usize> = (0..limit.groups).collect();
            groups.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
            groups.truncate(limit.selected);
            order.retain(|expert| groups.contains(&(expert / size)));
        }
        order.sort_by(|&a, &b| choice(b).total_cmp(&choice(a)).then(a.cmp(&b)));
        order.truncate(self.top_k);
        order
    }
}

/// Phi-3.5-MoE's sparsemixer at inference (MLX-LM `phimoe.sparsemixer`):
/// each round takes the best remaining expert and weighs it by its softmax
/// over the experts within `2 * jitter` of it, relative to
/// `max(|logit|, best)`.
fn sparse_mixer(logits: &[f32], top_k: usize, jitter: f32, scale: f32) -> Vec<RoutedExpert> {
    let mut masked = logits.to_vec();
    let mut selected = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        let (expert, best) = masked.iter().copied().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |kept, (index, value)| {
                if value > kept.1 {
                    (index, value)
                } else {
                    kept
                }
            },
        );
        let total: f32 = logits
            .iter()
            .zip(&masked)
            // A NaN distance stays in the band.
            .filter(|&(&logit, _)| {
                let distance = (best - logit) / logit.abs().max(best);
                distance.partial_cmp(&(2.0 * jitter)) != Some(std::cmp::Ordering::Greater)
            })
            .map(|(_, &value)| (value - best).exp())
            .sum();
        selected.push(RoutedExpert { expert, weight: scale / total });
        masked[expert] = f32::NEG_INFINITY;
    }
    selected
}
