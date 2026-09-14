use super::*;
use crate::engine::kernels::expert_group::aligned::{AlignedGroup, PaddingBudget};

#[test]
fn padded_affine_mlp_restores_only_real_routes_in_original_order() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let projections = weights(8, 64, 128, &stream)?;
    let input = Array::from_f32(&values(2 * 37 * 64), &[2, 37, 64])?;
    let ids = (0..222).map(|i| (i * 5) % 8).collect::<Vec<u32>>();
    let indices = Array::from_u32(&ids, &[2, 37, 3])?;
    let routing = Array::from_f32(&values(222), &[2, 37, 3])?;
    for (budget, capacity) in [(PaddingBudget::Four, 256), (PaddingBudget::Eight, 288)] {
        let padded =
            input.align_expert_inputs(&indices, 8, &AlignedGroup::with_budget(budget)?, &stream)?;
        assert_eq!(padded.input.shape()?, [capacity, 1, 64]);
        let output = mlp(&projections, &padded.input, &padded.indices, true, &stream)?;
        let restored = padded.restore_weighted(&output, &routing, &stream)?;
        let baseline = input.group_expert_inputs(&indices, 8, &stream)?;
        let expected = mlp(&projections, &baseline.input, &baseline.indices, true, &stream)?;
        let expected = baseline.restore_weighted(&expected, &routing, &stream)?;
        assert_close(&restored, &expected, &stream)?;
        assert!(restored.to_vec_f32(&stream)?.iter().any(|v| v.abs() > 1e-5));
        // Graph restoration also supports padding (used by clamped/GPT-OSS).
        let graph = padded.restore(&output, &stream)?.weighted_sum(&routing, -2, &stream)?;
        assert_close(&restored, &graph, &stream)?;
    }
    Ok(())
}
