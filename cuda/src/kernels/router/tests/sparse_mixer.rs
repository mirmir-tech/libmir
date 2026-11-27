//! Phi-3.5-MoE's sparsemixer against the `models` host routing (MLX-LM
//! `phimoe.sparsemixer`).

use mircuda::bf16;
use models::semantic::{ActivationSpec, RoutedExpertsSpec, RouterNormalization};

use super::Runtime;
use crate::{
    Result,
    kernels::{RouterUnitScoring, RouterUnitSpec, RouterUnitTopK},
};

const EXPERTS: usize = 16;
const TOP_K: usize = 2;
const JITTER: f64 = 0.01;

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
    let experts = RoutedExpertsSpec {
        expert_count: EXPERTS,
        top_k: TOP_K,
        intermediate_size: 1,
        activation: ActivationSpec::SwiGlu,
        router_normalization: RouterNormalization::SparseMixer { jitter_eps: JITTER },
        group_limit: None,
        output_scale: 1.0,
    };
    let scoring = RouterUnitScoring::of(&experts)?;
    let router = RouterUnitTopK::compile_scored(&runtime.compiler, spec, scoring)?;
    let scores = runtime.copy(&scores)?;
    let mut selected = runtime.pool.allocate::<u32>(&runtime.stream, rows.len() * TOP_K)?;
    let mut weights = runtime.pool.allocate::<bf16>(&runtime.stream, rows.len() * TOP_K)?;
    router.execute(&runtime.stream, &scores, None, &mut selected, &mut weights)?;
    let (selected, weights) = (runtime.read(&selected)?, runtime.read(&weights)?);
    let mut banded = false;
    for (row, logits) in rows.iter().enumerate() {
        let route = experts.route(logits, None)?;
        let indices: Vec<u32> = route
            .iter()
            .map(|selected| u32::try_from(selected.expert))
            .collect::<std::result::Result<_, _>>()?;
        let expected = route.iter().map(|selected| selected.weight);
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
