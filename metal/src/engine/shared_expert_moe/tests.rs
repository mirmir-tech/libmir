use super::*;
use crate::engine::{DenseLinear, QuantizedArrays, QuantizedLinear};

#[test]
fn executes_routed_and_shared_experts_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for fused in [false, true] {
        execute(fused, &stream)?;
    }
    Ok(())
}

#[test]
fn splits_interleaved_fused_expert_projection() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let weight = Array::from_f32(
        &[1.0, 0.0, 10.0, 0.0, 0.0, 2.0, 0.0, 20.0, 2.0, 0.0, 20.0, 0.0, 0.0, 3.0, 0.0, 30.0],
        &[2, 4, 2],
    )?;
    let gate_up = RoutedGateUp::Fused {
        projection: BoundLinear::Dense(DenseLinear::from_arrays(&weight, None, &stream)?),
        width: 2,
        interleaved: true,
    };
    let input = Array::from_f32(&[3.0, 4.0], &[1, 1, 2])?;
    let indices = Array::from_u32(&[0], &[1])?;

    let (gate, up) = gate_up.gather(&input, &indices, false, &stream)?;
    assert_eq!(gate.to_vec_f32(&stream)?, [3.0, 8.0]);
    assert_eq!(up.to_vec_f32(&stream)?, [30.0, 80.0]);
    Ok(())
}

fn execute(fused: bool, stream: &Stream) -> Result<()> {
    let routed_gate_up = if fused {
        RoutedGateUp::Fused {
            projection: linear(&[2, 128, 64], stream)?,
            width: 64,
            interleaved: false,
        }
    } else {
        RoutedGateUp::Separate {
            gate: linear(&[2, 64, 64], stream)?,
            up: linear(&[2, 64, 64], stream)?,
            fused: None,
        }
    };
    let moe = SharedExpertMoe {
        config: SharedExpertMoeConfig::new(2, 1, 64)?,
        router: linear(&[2, 64], stream)?,
        routed_gate_up,
        routed_down: linear(&[2, 64, 64], stream)?,
        shared_gate: linear(&[64, 64], stream)?,
        shared_up: linear(&[64, 64], stream)?,
        fused_shared_gate_up: None,
        shared_down: linear(&[64, 64], stream)?,
        shared_output_gate: Some(linear(&[1, 64], stream)?),
        shared_gate_up_mode: FusionMode::Disabled,
        routing: ExpertRouter::Unit,
    };
    let input = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = moe.forward(&input, stream)?;

    output.async_eval(stream)?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 1, 64]);
    assert!(output.to_vec_f32(stream)?.iter().all(|value| *value == 0.0));
    Ok(())
}

fn linear(shape: &[i32], stream: &Stream) -> Result<BoundLinear> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
    })?;
    let dense = Array::from_f32(&vec![0.0; elements], shape)?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}

/// The experts and weights a configured router selects, on the host.
fn routed_reference(
    logits: &[f32],
    router: &ScoredRouter,
    bias: &[f32],
    top_k: usize,
) -> Vec<(usize, f32)> {
    let softmax = matches!(router.routing, models::layout::ExpertRouting::Softmax { .. });
    let scores: Vec<f32> = if softmax {
        let total = logits.iter().map(|value| value.exp()).sum::<f32>();
        logits.iter().map(|value| value.exp() / total).collect()
    } else {
        logits.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect()
    };
    let mut selection = scores
        .iter()
        .zip(bias.iter().chain(std::iter::repeat(&0.0)))
        .map(|(s, b)| s + b)
        .collect::<Vec<_>>();
    if let Some(limit) = router.group_limit {
        let size = scores.len() / limit.groups;
        let mut groups = (0..limit.groups)
            .map(|group| {
                let mut members = selection[group * size..(group + 1) * size].to_vec();
                members.sort_by(|a, b| b.total_cmp(a));
                (
                    group,
                    if softmax {
                        members[0]
                    } else {
                        members[0] + members[1]
                    },
                )
            })
            .collect::<Vec<_>>();
        groups.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (group, _) in &groups[limit.selected..] {
            for value in &mut selection[group * size..(group + 1) * size] {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| selection[*b].total_cmp(&selection[*a]));
    let chosen = &order[..top_k];
    let renormalize =
        !matches!(router.routing, models::layout::ExpertRouting::Softmax { renormalize: false });
    let total = if renormalize {
        chosen.iter().map(|&expert| scores[expert]).sum()
    } else {
        1.0
    };
    let mut pairs = chosen
        .iter()
        .map(|&expert| (expert, scores[expert] / total * router.output_scale))
        .collect::<Vec<_>>();
    pairs.sort_by_key(|pair| pair.0);
    pairs
}

#[test]
fn configured_routers_select_from_the_best_groups() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = [0.3, -1.2, 2.0, 0.1, -0.4, 1.5, 0.9, -2.0];
    let bias = [0.0, 0.5, -0.2, 0.0, 0.3, 0.0, 0.0, 1.0];
    let cases = [
        (models::layout::ExpertRouting::SigmoidWithCorrectionBias, 4, 2, 2.5, true),
        (models::layout::ExpertRouting::Softmax { renormalize: false }, 2, 1, 1.0, false),
    ];
    for (routing, groups, selected, output_scale, biased) in cases {
        let router = ScoredRouter {
            routing,
            correction_bias: biased.then(|| Array::from_f32(&bias, &[8])).transpose()?,
            group_limit: Some(models::semantic::ExpertGroupLimit { groups, selected }),
            output_scale,
        };
        let output = router.route(&Array::from_f32(&logits, &[1, 1, 8])?, 2, &stream)?;
        let indices = output.indices.to_vec_u32(&stream)?;
        let weights = output.weights.to_vec_f32(&stream)?;
        let mut actual = indices
            .iter()
            .zip(&weights)
            .map(|(&index, &weight)| (usize::try_from(index).unwrap_or(0), weight))
            .collect::<Vec<_>>();
        actual.sort_by_key(|pair| pair.0);
        let expected = routed_reference(
            &logits,
            &router,
            if biased {
                &bias
            } else {
                &[]
            },
            2,
        );
        assert_eq!(
            actual.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            expected.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            "{routing:?}"
        );
        for ((_, actual), (_, expected)) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1.0e-5, "{routing:?}: {actual} != {expected}");
        }
    }
    Ok(())
}
