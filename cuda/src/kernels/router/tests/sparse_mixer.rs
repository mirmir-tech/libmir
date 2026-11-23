//! Phi-3.5-MoE's sparsemixer against MLX-LM `phimoe.sparsemixer`.

use mircuda::bf16;

use super::Runtime;
use crate::{
    Result,
    kernels::{
        RouterJitter, RouterUnitScoring, RouterUnitSpec, RouterUnitTopK, RouterUnitWeighting,
    },
};

const EXPERTS: usize = 16;
const TOP_K: usize = 2;
const JITTER: f32 = 0.01;

/// Near-ties keep several experts in the band, a clear winner keeps one, and
/// zero logits measure a NaN distance that stays in the band.
fn rows() -> [[f32; EXPERTS]; 3] {
    let mut rows = [[-1.5_f32; EXPERTS]; 3];
    rows[0][..6].copy_from_slice(&[1.0, 0.996_093_75, 0.2, -0.4, 0.992_187_5, 0.1]);
    rows[1][..6].copy_from_slice(&[3.0, -1.0, 0.5, 2.875, 0.0, -2.0]);
    rows[2][..3].copy_from_slice(&[0.0, 0.0, -0.5]);
    rows
}

#[test]
fn weighs_each_round_by_its_softmax_band() -> Result<()> {
    let runtime = Runtime::new()?;
    let rows = rows();
    let scores: Vec<bf16> = rows.iter().flatten().map(|&value| bf16::from_f32(value)).collect();
    let spec = RouterUnitSpec {
        tokens: rows.len(),
        experts: EXPERTS,
        top_k: TOP_K,
    };
    let scoring = RouterUnitScoring {
        weighting: RouterUnitWeighting::SparseMixer(RouterJitter::new(JITTER)?),
        group_limit: None,
        scale: 1.0,
    };
    let router = RouterUnitTopK::compile_scored(&runtime.compiler, spec, scoring)?;
    let scores = runtime.copy(&scores)?;
    let mut selected = runtime.pool.allocate::<u32>(&runtime.stream, rows.len() * TOP_K)?;
    let mut weights = runtime.pool.allocate::<bf16>(&runtime.stream, rows.len() * TOP_K)?;
    router.execute(&runtime.stream, &scores, None, &mut selected, &mut weights)?;
    let (selected, weights) = (runtime.read(&selected)?, runtime.read(&weights)?);
    let mut banded = false;
    for (row, logits) in rows.iter().enumerate() {
        let (indices, expected) = host(logits);
        let at = row * TOP_K..(row + 1) * TOP_K;
        assert_eq!(selected[at.clone()], indices, "row {row}");
        for (weight, expected) in weights[at].iter().zip(expected) {
            banded |= expected < 0.99;
            let error = (weight.to_f32() - expected).abs();
            assert!(error <= expected / 128.0, "row {row}: {weight} against {expected}");
        }
    }
    assert!(banded, "no band held more than its best expert");
    Ok(())
}

/// MLX-LM `phimoe.sparsemixer` on the host.
fn host(logits: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut masked = logits.to_vec();
    let (mut indices, mut weights) = (Vec::new(), Vec::new());
    for _ in 0..TOP_K {
        let (index, best) =
            masked.iter().copied().enumerate().fold((0, f32::NEG_INFINITY), |kept, (i, v)| {
                if v > kept.1 {
                    (i, v)
                } else {
                    kept
                }
            });
        let total = logits
            .iter()
            .zip(&masked)
            // A NaN distance stays in the band.
            .filter(|&(&logit, _)| {
                let distance = (best - logit) / logit.abs().max(best);
                distance.partial_cmp(&(2.0 * JITTER)) != Some(std::cmp::Ordering::Greater)
            })
            .map(|(_, &value)| (value - best).exp())
            .sum::<f32>();
        weights.push(total.recip());
        indices.push(u32::try_from(index).unwrap_or(u32::MAX));
        masked[index] = f32::NEG_INFINITY;
    }
    (indices, weights)
}
