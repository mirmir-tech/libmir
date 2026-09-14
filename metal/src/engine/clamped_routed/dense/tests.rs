use std::sync::Arc;

use super::*;
use crate::{
    config::MoePrefill,
    engine::{
        DenseLinear,
        kernels::expert_group::aligned::{AlignedGroup, PaddingBudget},
    },
};

#[test]
fn sorted_clamped_prefill_preserves_bias_clamp_routes_and_interleaving() -> Result<()> {
    let shape = MxFp4Shape {
        tokens: 4,
        top_k: 2,
        hidden: 4,
        intermediate: 4,
    };
    let input_values =
        [8.0, -9.0, 2.0, -1.0, -3.0, 7.0, 0.5, 2.0, 1.0, -2.0, 9.0, -8.0, 4.0, 3.0, -5.0, 6.0];
    let input = Array::from_f32(&input_values, &[4, 4])?;
    let indices = [2, 0, 1, 2, 0, 1, 2, 1];
    let weights = [0.2, 0.8, 0.4, 0.6, 0.7, 0.3, 0.55, 0.45];
    let routing = RouterOutput {
        indices: Array::from_u32(&indices, &[4, 2])?,
        weights: Array::from_f32(&weights, &[4, 2])?,
    };
    let limit = Array::from_f32(&[2.0], &[])?;
    let mut reference = vec![0.0_f32; 16];
    for token in 0..4 {
        for choice in 0..2 {
            let expert = indices[token * 2 + choice];
            for dim in 0..4 {
                let x = input_values[token * 4 + dim];
                let gate = (x + bias(expert, 0)).min(2.0);
                let up = (x + bias(expert, 1)).clamp(-2.0, 2.0) + 1.0;
                let value = (gate / (1.0 + (-1.702 * gate).exp())).mul_add(up, bias(expert, 2));
                reference[token * 4 + dim] =
                    weights[token * 2 + choice].mul_add(value, reference[token * 4 + dim]);
            }
        }
    }
    for mode in [MoePrefill::Default, MoePrefill::ClampedSorted] {
        let mut config = crate::MetalConfig::default();
        config.diagnostics.moe_prefill = mode;
        let stream = Stream::new_gpu_with_config(Arc::new(config))?;
        let gate = projection(0, &stream)?;
        let up = projection(1, &stream)?;
        let down = projection(2, &stream)?;
        let output = dense_experts(&input, &routing, [&gate, &up, &down], &limit, shape, &stream)?;
        compare(&output, &reference, &stream)?;
        for budget in [PaddingBudget::Four, PaddingBudget::Eight] {
            let aligned = input.reshape(&[1, 4, 4], &stream)?.align_expert_inputs(
                &routing.indices.reshape(&[1, 4, 2], &stream)?,
                3,
                &AlignedGroup::with_budget(budget)?,
                &stream,
            )?;
            let activated = activate(
                &gate.gather(&aligned.input, &aligned.indices, true, &stream)?,
                &up.gather(&aligned.input, &aligned.indices, true, &stream)?,
                &limit,
                &stream,
            )?;
            let padded_output = down.gather(&activated, &aligned.indices, true, &stream)?;
            let restored = aligned.restore(&padded_output, &stream)?.weighted_sum(
                &routing.weights.reshape(&[1, 4, 2], &stream)?,
                -2,
                &stream,
            )?;
            compare(&restored, &reference, &stream)?;
        }
        for interleaved in [false, true] {
            let fused = fused_projection(interleaved, &stream)?;
            let output = fused_dense_experts(
                &input, &routing, &fused, &down, interleaved, &limit, shape, &stream,
            )?;
            compare(&output, &reference, &stream)?;
        }
    }
    Ok(())
}

fn bias(expert: u32, projection: usize) -> f32 {
    const BIASES: [[f32; 3]; 3] = [[0.2, -0.5, 1.0], [-0.3, 0.75, -2.0], [0.6, -0.25, 3.0]];
    BIASES[expert as usize][projection]
}

fn projection(kind: usize, stream: &Stream) -> Result<BoundLinear> {
    let weights = (0..3 * 4 * 4)
        .map(|i| {
            if i % 4 == (i / 4) % 4 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let biases = (0..3_u32).flat_map(|expert| vec![bias(expert, kind); 4]).collect::<Vec<_>>();
    Ok(BoundLinear::Dense(DenseLinear::from_arrays(
        &Array::from_f32(&weights, &[3, 4, 4])?,
        Some(Array::from_f32(&biases, &[3, 4])?),
        stream,
    )?))
}

fn fused_projection(interleaved: bool, stream: &Stream) -> Result<BoundLinear> {
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for expert in 0..3 {
        for row in 0..8 {
            let (kind, dim) = if interleaved {
                (row % 2, row / 2)
            } else {
                (row / 4, row % 4)
            };
            biases.push(bias(expert, kind));
            weights.extend((0..4).map(|column| {
                if column == dim {
                    1.0
                } else {
                    0.0
                }
            }));
        }
    }
    Ok(BoundLinear::Dense(DenseLinear::from_arrays(
        &Array::from_f32(&weights, &[3, 8, 4])?,
        Some(Array::from_f32(&biases, &[3, 8])?),
        stream,
    )?))
}

fn compare(output: &Array, reference: &[f32], stream: &Stream) -> Result<()> {
    let actual = output.to_vec_f32(stream)?;
    assert_eq!(actual.len(), reference.len());
    assert!(actual.iter().zip(reference).all(|(a, b)| (a - b).abs() < 1e-5));
    Ok(())
}

#[test]
fn clamped_activation_preserves_projection_dtype_with_f32_limit() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let graph = stream.native().graph();
    let limit = Array::from_f32(&[7.0], &[])?;
    for dtype in [mirtal::DType::Bfloat16, mirtal::DType::Float32] {
        let gate = Array::from_native(
            graph.astype(Array::from_f32(&[20.0, -20.0, 0.0, 1.0], &[1, 1, 4])?.native(), dtype)?,
        )?;
        let up = Array::from_native(
            graph.astype(Array::from_f32(&[20.0, -20.0, 2.0, 0.0], &[1, 1, 4])?.native(), dtype)?,
        )?;
        let output = activate(&gate, &up, &limit, &stream)?;
        assert_eq!(output.dtype()?, gate.dtype()?);
        let values = output.to_vec_f32(&stream)?;
        assert!((values[0] - 56.0).abs() < 0.01);
        assert!(values[1].abs() < 1.0e-10);
        assert!(values[2].abs() < f32::EPSILON);
        assert!((values[3] - 0.8458).abs() < 0.003);
    }
    Ok(())
}
