//! Phi-3.5-MoE's sparsemixer at inference (transformers `sparsemixer`,
//! MLX-LM `phimoe.sparsemixer`): each round takes the best remaining expert
//! and weighs it by its softmax over the experts within `2 * jitter` of it,
//! measured relative to `max(|logit|, best)`.

use mirtal::{DType, Graph, Shape};

use crate::engine::{Array, Result, RouterOutput};

/// Large enough that a masked logit's softmax weight is zero, finite so the
/// masks combine without `inf - inf`.
const MASKED: f32 = 1.0e30;

pub(super) fn route(
    logits: &Array,
    top_k: usize,
    jitter: f32,
    graph: Graph<'_>,
) -> Result<RouterOutput> {
    let logits = graph.astype(logits.native(), DType::Float32)?;
    let experts = *logits.shape()?.dimensions().last().unwrap_or(&0);
    let expert_ids = graph.arange(0.0, f32::from(u16::try_from(experts)?), 1.0, DType::Float32)?;
    let magnitude = graph.maximum(&logits, &graph.negative(&logits)?)?;
    let threshold = graph.full(&Shape::new([])?, 2.0 * jitter, DType::Float32)?;
    let mut masked = logits.clone();
    let (mut indices, mut weights) = (Vec::with_capacity(top_k), Vec::with_capacity(top_k));
    for _ in 0..top_k {
        let best = graph.reduce_max(&masked, -1, true)?;
        let index = graph.argmax_axis(&masked, -1, true)?;
        let factor = graph.maximum(&magnitude, &best)?;
        let distance = graph.divide(&graph.subtract(&best, &logits)?, &factor)?;
        let far = graph.astype(&graph.less(&threshold, &distance)?, DType::Float32)?;
        let band = graph.subtract(&masked, &graph.multiply_scalar(&far, MASKED)?)?;
        let band = graph.softmax(&band, -1, true)?;
        weights.push(graph.take_along_axis(&band, &index, -1)?);
        // Drop the chosen expert from later rounds.
        let chosen = graph.astype(&index, DType::Float32)?;
        let selected = graph.logical_and(
            &graph.greater_equal(&expert_ids, &chosen)?,
            &graph.less(&expert_ids, &graph.add_scalar(&chosen, 1.0)?)?,
        )?;
        let selected = graph.astype(&selected, DType::Float32)?;
        masked = graph.subtract(&masked, &graph.multiply_scalar(&selected, MASKED)?)?;
        indices.push(index);
    }
    let indices = graph.concatenate(&indices.iter().collect::<Vec<_>>(), -1)?;
    let weights = graph.concatenate(&weights.iter().collect::<Vec<_>>(), -1)?;
    Ok(RouterOutput {
        indices: Array::from_native(graph.astype(&indices, DType::Uint32)?)?,
        weights: Array::from_native(weights)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Stream;

    /// MLX-LM `phimoe.sparsemixer` on the host.
    fn host(logits: &[f32], top_k: usize, jitter: f32) -> (Vec<u32>, Vec<f32>) {
        let mut masked = logits.to_vec();
        let (mut indices, mut weights) = (Vec::new(), Vec::new());
        for _ in 0..top_k {
            let (index, best) =
                masked.iter().copied().enumerate().fold((0, f32::NEG_INFINITY), |kept, (i, v)| {
                    if v > kept.1 {
                        (i, v)
                    } else {
                        kept
                    }
                });
            let band = logits
                .iter()
                .zip(&masked)
                .map(|(&logit, &value)| {
                    let factor = logit.abs().max(best);
                    if (best - logit) / factor > 2.0 * jitter {
                        f32::NEG_INFINITY
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>();
            let total = band.iter().map(|value| (value - best).exp()).sum::<f32>();
            weights.push((band[index] - best).exp() / total);
            indices.push(u32::try_from(index).unwrap_or(u32::MAX));
            masked[index] = f32::NEG_INFINITY;
        }
        (indices, weights)
    }

    #[test]
    fn matches_mlx_lm_sparsemixer() -> Result<()> {
        let stream = Stream::new_gpu()?;
        // Near-ties keep several experts in the band; a clear winner keeps one.
        let rows = [[1.00_f32, 0.995, 0.2, -0.4, 0.99, 0.1], [3.0, -1.0, 0.5, 2.9, 0.0, -2.0]];
        let flat = rows.concat();
        let logits = Array::from_f32(&flat, &[2, 6])?;
        let output = route(&logits, 2, 0.01, stream.native().graph())?;
        let indices = output.indices.astype(crate::engine::Dtype::Float32, &stream)?;
        let indices = indices.to_vec_f32(&stream)?;
        let weights = output.weights.to_vec_f32(&stream)?;
        for (row, logits) in rows.iter().enumerate() {
            let (expected_indices, expected_weights) = host(logits, 2, 0.01);
            for slot in 0..2 {
                let expected = f32::from(u16::try_from(expected_indices[slot])?);
                assert!((indices[row * 2 + slot] - expected).abs() < 0.5);
                assert!((weights[row * 2 + slot] - expected_weights[slot]).abs() < 1.0e-5);
            }
        }
        Ok(())
    }
}
